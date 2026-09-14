"""Route generation and PEMIRL test-time conditioning.

* :func:`pemirl_support_sets` — group test episodes by vessel and take a small
  support set per vessel to infer z (paper §II: "route generation is
  conditioned on a small support set per vessel to infer context").
* :func:`pemirl_log_likelihood` — per-decision test log-likelihood of the
  PEMIRL policy with support-set-inferred contexts (the metric on which the
  paper reports a ~29% improvement over MCE-IRL in the tabular setting).
* :func:`generate_route_pemirl` / :func:`generate_route_policy` — decode O–D
  routes from the conditioned policy.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..algos.pemirl import PEMIRL
from ..algos.mce_irl import soft_value_iteration
from ..data.dataset import Episode
from ..features import FeatureBuilder
from ..env.gcrl_env import GCRLNavEnv, Task
from ..env.graph_mdp import GraphMDP


def _episode_tensors(ep: Episode, features: FeatureBuilder):
    phi = features.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month,
                                  ep.mmsi)
    return (torch.from_numpy(phi),
            torch.from_numpy(ep.actions.astype(np.int64)))


def pemirl_support_sets(episodes: list[Episode], support_size: int = 3,
                        seed: int = 0):
    """Split each vessel's episodes into (support, query). Vessels with fewer
    than support_size+1 episodes contribute all-but-one as support."""
    rng = np.random.default_rng(seed)
    by_vessel: dict[int, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_vessel[ep.mmsi].append(ep)
    out = {}
    for mmsi, eps in by_vessel.items():
        if len(eps) < 2:
            continue
        idx = rng.permutation(len(eps))
        k = min(support_size, len(eps) - 1)
        out[mmsi] = ([eps[i] for i in idx[:k]], [eps[i] for i in idx[k:]])
    return out


@torch.no_grad()
def pemirl_log_likelihood(model: PEMIRL, episodes: list[Episode],
                          features: FeatureBuilder, mdp: GraphMDP,
                          support_size: int = 3, seed: int = 0
                          ) -> tuple[float, int]:
    """Per-vessel support-conditioned per-decision test log-likelihood."""
    total, n_dec = 0.0, 0
    for mmsi, (support, query) in pemirl_support_sets(
            episodes, support_size, seed).items():
        z = model.infer_context_support(
            [_episode_tensors(ep, features) for ep in support], fixed=True)
        for ep in query:
            phi, act = _episode_tensors(ep, features)
            phi = phi.to(model.device)
            act = act.to(model.device)
            mask = torch.from_numpy(
                mdp.action_mask[ep.states[:-1]]).to(model.device)
            zT = z.expand(phi.size(0), -1)
            lp = model.policy.log_prob_action(phi, mask, act, zT)
            total += float(lp.sum().item())
            n_dec += len(ep)
    return total, n_dec


@torch.no_grad()
def generate_route_pemirl(model: PEMIRL, env: GCRLNavEnv, task: Task,
                          support: list[Episode], features: FeatureBuilder,
                          deterministic: bool = True) -> list[int]:
    z = model.infer_context_support(
        [_episode_tensors(ep, features) for ep in support], fixed=True)
    return generate_route_policy(model.policy, env, task, z=z,
                                 deterministic=deterministic,
                                 device=str(model.device))


@torch.no_grad()
def generate_route_pemirl_softvi(model: PEMIRL, mdp: GraphMDP, task: Task,
                                 support: list[Episode],
                                 features: FeatureBuilder,
                                 horizon: int = 512,
                                 temperature: float = 1.0) -> list[int]:
    """Planner-matched decode of the frozen adversarial reward.

    To remove the decoder as a confound in the route-fidelity comparison
    (MCE-IRL is decoded by a global soft-VI planner, PEMIRL by its native
    policy), we decode f with the *same* soft-VI planner used for MCE-IRL:
    infer the per-vessel context z from the support set, marginalize the
    reward to a state value ``r~(s) = logsumexp_{a valid} f(s,a,z)``, and run
    the identical finite-horizon soft VI + greedy rollout as ``greedy_route``.
    Works for the pooled AIRL control too (context_dim=0 -> empty z)."""
    z = model.infer_context_support(
        [_episode_tensors(ep, features) for ep in support], fixed=True)  # (1,Z)
    phi_all = features.phi_all_states(task.goal, task.year, task.month,
                                      task.mmsi)                          # (S,D)
    S, A = mdp.n_states, mdp.n_actions
    obs = torch.as_tensor(phi_all, dtype=torch.float32, device=model.device)
    zT = z.to(model.device).expand(S, -1)
    F = np.empty((S, A), dtype=np.float64)
    for a in range(A):
        act = torch.full((S,), a, dtype=torch.long, device=model.device)
        F[:, a] = model.discriminator.get_unnormed_d(obs, act, zT
                                                     ).squeeze(-1).cpu().numpy()
    Fm = np.where(mdp.action_mask, F, -np.inf)             # mask invalid actions
    m = np.max(Fm, axis=1, keepdims=True)
    r_tilde = (m.squeeze(1) + np.log(np.exp(Fm - m).sum(axis=1)))
    r_tilde = np.where(np.isfinite(r_tilde), r_tilde, 0.0)  # (S,) state reward
    vi = soft_value_iteration(mdp, r_tilde, task.goal, horizon, temperature)
    s, path = int(task.start), [int(task.start)]
    for t in range(horizon):
        if s == int(task.goal):
            break
        a = int(np.argmax(np.where(mdp.action_mask[s], vi.log_pi[t, s],
                                   -np.inf)))
        s = int(mdp.next_state[s, a])
        path.append(s)
    return path


@torch.no_grad()
def generate_route_policy(policy, env: GCRLNavEnv, task: Task,
                          z: torch.Tensor | None = None,
                          deterministic: bool = True,
                          device: str = "cpu") -> list[int]:
    obs = env.reset(task)
    path = [env.s]
    done = False
    while not done:
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(env.action_mask(), dtype=torch.bool,
                            device=device).unsqueeze(0)
        a, _, _ = policy.act(o, m, z, deterministic=deterministic)
        obs, _, done, _ = env.step(int(a.item()))
        path.append(env.s)
    return path
