"""Maximum Causal Entropy IRL (Ziebart et al., 2010) — tabular, goal-conditioned.

Reward model:   r_theta(s) = theta . phi(s)   (per-episode phi includes
goal-relative and vessel-static blocks, so theta is shared/pooled across all
demonstrations — the "single reward function" assumption examined in the paper).

For each *episode group* (same goal, year, month and — when vessel statics are
enabled — the same MMSI feature vector), we:

  1. run **finite-horizon backward soft value iteration** with the goal state
     absorbing:   Q_t(s,a) = r(s) + V_{t+1}(s'),
                  V_t(s)   = logsumexp_a Q_t(s,a)   (masked),
                  V_t(goal)= 0
  2. compute the maximum-causal-entropy policy pi_t(a|s) = exp(Q_t - V_t)
  3. propagate **expected state visitations** D_t forward from the start
  4. accumulate the model feature expectations  E_pi[ sum_t phi(s_t) ]

The gradient of the demonstration log-likelihood is the classic

    g = E_expert[phi] - E_pi_theta[phi]            (+ L2)

ascended with Adam. Per-decision test log-likelihood uses the same soft-VI
policy (this is also the metric where PEMIRL's context conditioning gives the
~29% improvement reported in the paper).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.dataset import Episode
from ..features import FeatureBuilder
from ..env.graph_mdp import GraphMDP
from ..utils.logging import get_logger

log = get_logger(__name__)

NEG_INF = -1e18


def _logsumexp_masked(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q = np.where(mask, q, NEG_INF)
    m = q.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(q - m).sum(axis=-1, keepdims=True)))[..., 0]


@dataclass
class SoftVIResult:
    log_pi: np.ndarray   # (T, S, A) log policy (time-varying, masked)
    V: np.ndarray        # (T+1, S)


def soft_value_iteration(mdp: GraphMDP, r: np.ndarray, goal: int, horizon: int,
                         temperature: float = 1.0) -> SoftVIResult:
    """Finite-horizon soft VI with absorbing goal. r: (S,) state reward."""
    S, A = mdp.n_states, mdp.n_actions
    ns, mask = mdp.next_state, mdp.action_mask
    ns_safe = np.where(ns >= 0, ns, 0)
    r = r / max(temperature, 1e-8)

    V = np.zeros((horizon + 1, S), dtype=np.float64)
    V[horizon] = 0.0
    log_pi = np.zeros((horizon, S, A), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        q = r[:, None] + V[t + 1][ns_safe]          # (S, A)
        q = np.where(mask, q, NEG_INF)
        v = _logsumexp_masked(q, mask)               # (S,)
        v[goal] = 0.0                                # absorbing goal
        V[t] = v
        lp = q - v[:, None]
        lp[goal, :] = np.where(mask[goal], -np.log(mask[goal].sum()), NEG_INF)
        log_pi[t] = lp
    return SoftVIResult(log_pi=log_pi, V=V)


def expected_visitation(mdp: GraphMDP, log_pi: np.ndarray, start: int,
                        goal: int, horizon: int) -> np.ndarray:
    """Forward pass: expected state-visitation counts D (S,), goal absorbing."""
    S, A = mdp.n_states, mdp.n_actions
    ns, mask = mdp.next_state, mdp.action_mask
    d = np.zeros(S, dtype=np.float64)
    d[start] = 1.0
    D = np.zeros(S, dtype=np.float64)
    for t in range(horizon):
        D += d
        pi_t = np.exp(np.where(mask, log_pi[t], NEG_INF))   # (S, A)
        flow = d[:, None] * pi_t
        flow[goal, :] = 0.0                                  # absorb at goal
        d_next = np.zeros(S, dtype=np.float64)
        valid = mask & (ns >= 0)
        np.add.at(d_next, ns[valid], flow[valid])
        d = d_next
        if d.sum() < 1e-12:
            break
    return D


# --------------------------------------------------------------------------- #

class MCEIRL:
    """Pooled linear-reward MCE-IRL over grouped demonstration episodes."""

    def __init__(self, mdp: GraphMDP, features: FeatureBuilder,
                 lr: float = 0.05, l2: float = 1e-4, temperature: float = 1.0,
                 group_by_goal: bool = True, seed: int = 0):
        self.mdp = mdp
        self.features = features
        self.lr, self.l2, self.temperature = lr, l2, temperature
        self.group_by_goal = group_by_goal
        rng = np.random.default_rng(seed)
        self.theta = 0.01 * rng.standard_normal(features.dim)
        # Adam state
        self._m = np.zeros_like(self.theta)
        self._v = np.zeros_like(self.theta)
        self._step = 0

    # ----------------------------------------------------------- grouping
    def _group(self, episodes: list[Episode]):
        """Group episodes that share an IDENTICAL soft-VI solve.

        Key is (goal, year, month, mmsi). It must include mmsi: vessel-static
        features add a state-independent offset to r(s), but soft-VI here uses an
        absorbing goal with V[goal]=0 pinned, and that boundary condition makes
        the policy NOT invariant to a uniform reward offset (verified: adding a
        constant changes log_pi). So vessels with different statics get different
        policies and cannot share a VI solve. Episodes that DO share the full key
        (same goal, season, AND vessel) share an identical solve, which is exact.
        """
        groups: dict[tuple, list[Episode]] = defaultdict(list)
        for ep in episodes:
            key = (ep.goal, ep.year, ep.month,
                   ep.mmsi if self.features.spec.use_vessel_static else None) \
                if self.group_by_goal else (id(ep),)
            groups[key].append(ep)
        return groups

    # -------------------------------------------------------------- train
    def fit(self, episodes: list[Episode], n_iters: int = 200,
            log_every: int = 10, callback=None) -> np.ndarray:
        groups = self._group(episodes)
        log.info("MCE-IRL: %d episodes in %d (goal, season, vessel) groups",
                 len(episodes), len(groups))
        n_eps = len(episodes)

        # ---- precompute, ONCE, everything that does not depend on theta ------
        # phi_all is a function of (goal, year, month, mmsi) only — IDENTICAL to
        # the group key — and is CONSTANT across iterations (only theta changes).
        # Caching it per group here, instead of rebuilding it inside every
        # iteration, removes ~200x redundant feature construction (the dominant
        # cost: 285ms/call x n_groups x n_iters). This is a pure hoist of a
        # loop-invariant; results are bit-for-bit identical to recomputing.
        f_expert = np.zeros(self.features.dim)
        phi_cache: dict[tuple, np.ndarray] = {}
        horizon: dict[tuple, int] = {}
        for key, eps in groups.items():
            ep0 = eps[0]
            phi_cache[key] = self.features.phi_all_states(
                ep0.goal, ep0.year, ep0.month, ep0.mmsi)
            horizon[key] = max(len(ep) for ep in eps)
            for ep in eps:
                # expert feature expectation: sum phi over the demonstrated path
                pe = self.features.episode_matrix(ep.states[:-1], ep.goal,
                                                  ep.year, ep.month, ep.mmsi)
                f_expert += pe.sum(axis=0)
        f_expert /= n_eps

        for it in range(n_iters):
            f_model = np.zeros(self.features.dim)
            ll = 0.0
            n_dec = 0
            for key, eps in groups.items():
                phi_all = phi_cache[key]          # loop-invariant, cached once
                r = phi_all @ self.theta
                vi = soft_value_iteration(self.mdp, r, eps[0].goal,
                                          horizon[key], self.temperature)
                for ep in eps:
                    D = expected_visitation(self.mdp, vi.log_pi,
                                            int(ep.states[0]), ep.goal, len(ep))
                    f_model += (D[:, None] * phi_all).sum(axis=0)
                    ll += vi.log_pi[np.arange(len(ep)), ep.states[:-1],
                                    ep.actions].sum()
                    n_dec += len(ep)
            f_model /= n_eps
            grad = f_expert - f_model - self.l2 * self.theta
            self._adam(grad)
            if callback:
                callback(it, ll / max(n_dec, 1), self.theta)
            if it % log_every == 0:
                log.info("iter %4d | per-decision LL %.4f | |grad| %.4f",
                         it, ll / max(n_dec, 1), np.linalg.norm(grad))
        return self.theta

    def _adam(self, grad: np.ndarray, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._step += 1
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * grad ** 2
        mh = self._m / (1 - b1 ** self._step)
        vh = self._v / (1 - b2 ** self._step)
        self.theta += self.lr * mh / (np.sqrt(vh) + eps)   # ascent

    # ---------------------------------------------------------- evaluate
    def log_likelihood(self, episodes: list[Episode]) -> tuple[float, int]:
        """Total and per-decision log-likelihood under the soft-VI policy."""
        total, n_dec = 0.0, 0
        for key, eps in self._group(episodes).items():
            ep0 = eps[0]
            phi_all = self.features.phi_all_states(ep0.goal, ep0.year,
                                                   ep0.month, ep0.mmsi)
            r = phi_all @ self.theta
            H = max(len(ep) for ep in eps)
            vi = soft_value_iteration(self.mdp, r, ep0.goal, H, self.temperature)
            for ep in eps:
                total += vi.log_pi[np.arange(len(ep)), ep.states[:-1],
                                   ep.actions].sum()
                n_dec += len(ep)
        return float(total), n_dec

    def greedy_route(self, start: int, goal: int, year, month, mmsi,
                     horizon: int = 512) -> list[int]:
        phi_all = self.features.phi_all_states(goal, year, month, mmsi)
        r = phi_all @ self.theta
        vi = soft_value_iteration(self.mdp, r, goal, horizon, self.temperature)
        s, path = start, [start]
        for t in range(horizon):
            if s == goal:
                break
            a = int(np.argmax(np.where(self.mdp.action_mask[s],
                                       vi.log_pi[t, s], NEG_INF)))
            s = int(self.mdp.next_state[s, a])
            path.append(s)
        return path

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.theta,
                 mu=self.features._mu if self.features._mu is not None else np.zeros(0),
                 sd=self.features._sd if self.features._sd is not None else np.zeros(0))
        log.info("MCE-IRL theta saved to %s", path)

    def load(self, path: str | Path) -> None:
        z = np.load(path)
        self.theta = z["theta"]
        if z["mu"].size:
            self.features._mu, self.features._sd = z["mu"], z["sd"]
