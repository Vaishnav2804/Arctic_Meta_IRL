"""Deep-MCE — maximum-causal-entropy IRL with a *deep* state reward
(Wulfmeier et al., 2015: "Maximum Entropy Deep Inverse Reinforcement
Learning", adapted to the finite-horizon goal-absorbing soft VI of
``mce_irl.py``).

Reward model: r_theta(s) = MLP(phi(s)) — the same per-episode feature vector
the linear MCE-IRL reward sees, through a (256, 256) MLP matching the AIRL
discriminator's capacity. Everything else is *identical* to MCE-IRL: the same
(goal, year, month, vessel) episode groups, the same exact finite-horizon
backward soft value iteration and forward expected-visitation pass, and the
same likelihood gradient — which for a deep reward becomes

    dL/dtheta = sum_s ( mu_expert(s) - mu_model(s) ) * dr_theta(s)/dtheta,

i.e. the visitation-difference vector back-propagated through the reward
network (the planner is treated as the exact inference layer; no sampling, no
discriminator).

Position in the ladder: isolates *reward capacity* under the *exact* MaxEnt
objective. MCE-IRL -> Deep-MCE changes only linear -> nonlinear; Deep-MCE ->
AIRL changes only exact soft-VI partition -> adversarial/sampled estimation.
Evaluation is the same soft-VI policy log-likelihood as MCE-IRL.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import Episode
from ..env.graph_mdp import GraphMDP
from ..features import FeatureBuilder
from ..models.discriminator import mlp
from ..utils.logging import get_logger
from .mce_irl import expected_visitation, soft_value_iteration

log = get_logger(__name__)


class DeepMCEIRL:
    """Grouped deep-reward MCE-IRL; mirrors :class:`MCEIRL`'s interface."""

    def __init__(self, mdp: GraphMDP, features: FeatureBuilder,
                 hidden=(256, 256), lr: float = 1e-3,
                 weight_decay: float = 1e-4, grad_clip: float = 10.0,
                 temperature: float = 1.0, device: str = "cpu",
                 seed: int = 0):
        self.mdp = mdp
        self.features = features
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.net = mlp([features.dim, *tuple(hidden), 1]).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr,
                                      weight_decay=weight_decay)

    # ----------------------------------------------------------- grouping
    def _group(self, episodes: list[Episode]):
        """Same exact-solve grouping as MCEIRL (see its docstring on why the
        key must include mmsi when vessel statics are in phi)."""
        groups: dict[tuple, list[Episode]] = defaultdict(list)
        for ep in episodes:
            key = (ep.goal, ep.year, ep.month,
                   ep.mmsi if self.features.spec.use_vessel_static else None)
            groups[key].append(ep)
        return groups

    def _reward(self, phi_t: torch.Tensor) -> torch.Tensor:
        return self.net(phi_t).squeeze(-1)                    # (S,)

    # -------------------------------------------------------------- train
    def fit(self, episodes: list[Episode], n_iters: int = 150,
            log_every: int = 10, callback=None) -> None:
        groups = self._group(episodes)
        n_eps = len(episodes)
        log.info("Deep-MCE: %d episodes in %d (goal, season, vessel) groups",
                 n_eps, len(groups))

        # loop-invariant caches (exactly as in MCEIRL.fit): per-group features
        # on device, horizons, and per-group expert state-visit counts.
        phi_cache: dict[tuple, torch.Tensor] = {}
        horizon: dict[tuple, int] = {}
        expert_counts: dict[tuple, np.ndarray] = {}
        S = self.mdp.n_states
        for key, eps in groups.items():
            ep0 = eps[0]
            phi = self.features.phi_all_states(ep0.goal, ep0.year, ep0.month,
                                               ep0.mmsi)
            phi_cache[key] = torch.as_tensor(phi, dtype=torch.float32,
                                             device=self.device)
            horizon[key] = max(len(ep) for ep in eps)
            cnt = np.zeros(S, dtype=np.float64)
            for ep in eps:
                np.add.at(cnt, ep.states[:-1], 1.0)
            expert_counts[key] = cnt

        for it in range(n_iters):
            ll, n_dec = 0.0, 0
            self.optim.zero_grad()
            for key, eps in groups.items():
                phi_t = phi_cache[key]
                r_t = self._reward(phi_t)
                r_np = r_t.detach().cpu().numpy().astype(np.float64)
                vi = soft_value_iteration(self.mdp, r_np, eps[0].goal,
                                          horizon[key], self.temperature)
                g = expert_counts[key].copy()
                for ep in eps:
                    g -= expected_visitation(self.mdp, vi.log_pi,
                                             int(ep.states[0]), ep.goal,
                                             len(ep))
                    ll += vi.log_pi[np.arange(len(ep)), ep.states[:-1],
                                    ep.actions].sum()
                    n_dec += len(ep)
                # ascend LL: loss = -(mu_E - mu_model) . r  (grads accumulate
                # across groups; one optimizer step per iteration)
                g_t = torch.as_tensor(g / n_eps, dtype=torch.float32,
                                      device=self.device)
                loss = -(g_t @ r_t)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           self.grad_clip)
            self.optim.step()
            if callback:
                callback(it, ll / max(n_dec, 1))
            if it % log_every == 0:
                log.info("iter %4d | per-decision LL %.4f", it,
                         ll / max(n_dec, 1))

    # ---------------------------------------------------------- evaluate
    @torch.no_grad()
    def _reward_np(self, key_ep: Episode) -> np.ndarray:
        phi = self.features.phi_all_states(key_ep.goal, key_ep.year,
                                           key_ep.month, key_ep.mmsi)
        phi_t = torch.as_tensor(phi, dtype=torch.float32, device=self.device)
        return self._reward(phi_t).cpu().numpy().astype(np.float64)

    def log_likelihood(self, episodes: list[Episode]) -> tuple[float, int]:
        total, n_dec = 0.0, 0
        for _, ll in self.per_episode_ll(episodes).items():
            total += ll
        n_dec = sum(len(ep) for ep in episodes)
        return float(total), n_dec

    def per_episode_ll(self, episodes: list[Episode]) -> dict[int, float]:
        """{voyage_index: total LL} under the soft-VI policy (as MCE-IRL)."""
        out = {}
        for key, eps in self._group(episodes).items():
            r = self._reward_np(eps[0])
            H = max(len(ep) for ep in eps)
            vi = soft_value_iteration(self.mdp, r, eps[0].goal, H,
                                      self.temperature)
            for ep in eps:
                out[ep.voyage_index] = float(
                    vi.log_pi[np.arange(len(ep)), ep.states[:-1],
                              ep.actions].sum())
        return out

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict()}, path)
        log.info("Deep-MCE reward saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
