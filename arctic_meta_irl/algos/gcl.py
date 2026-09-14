"""GCL — Guided Cost Learning (Finn et al., 2016), sample-based MaxEnt IRL.

Learns a deep cost c_theta(s, a) by directly optimizing the MaxEnt IRL
likelihood with an importance-sampled partition estimate:

    L(theta) = E_demo[ c_theta(tau) ]
             + log ( (1/M) * sum_{tau in fusion} exp(-c_theta(tau)) / q(tau) )

where q(tau) = prod_t pi(a_t|s_t) is the current sampler policy's trajectory
probability and the fusion set contains both generated rollouts and the demo
minibatch (Finn et al.'s fusion distribution; demos are scored under the same
q). The policy is improved on r = -c via the same action-masked PPO as the
adversarial arms, closing the sample-refinement loop.

Position in the ladder: a *nonlinear* reward trained by the *MaxEnt-IRL*
objective (like Deep-MCE) but with the partition estimated by *sampling*
(like AIRL/GAIL) instead of exact soft VI. AIRL is GCL's discriminator-based
successor, so GCL vs AIRL isolates the effect of the adversarial formulation
while holding "deep reward + sampled partition" fixed.

Protocol parity with the AIRL arm: cost MLP (256, 256) == f's capacity,
same PPO hyperparameters and rollout budget, same gradient-norm clip and
per-batch reward standardization for the policy signal.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..models.discriminator import mlp
from ..models.policy import MaskedPolicy
from ..utils.logging import get_logger
from .sampler import Rollout

log = get_logger(__name__)


class GCL(torch.nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict,
                 device: str = "cpu"):
        super().__init__()
        p = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.grad_clip = float(p.get("cost_grad_clip", 1.0))
        self.reward_norm = bool(p.get("reward_norm", True))
        self.demo_batch = int(p.get("demo_batch", 32))
        self.c_clamp = float(p.get("c_clamp", 50.0))
        self.device = torch.device(device)

        self.cost = mlp([obs_dim + n_actions,
                         *tuple(p.get("cost_hidden", (256, 256))), 1])
        self.policy = MaskedPolicy(
            obs_dim, n_actions, context_dim=0,
            hidden=tuple(p.get("policy_hidden", (256, 256))))
        self.optim = torch.optim.Adam(
            self.cost.parameters(),
            lr=float(p.get("optimizer_lr_cost", 3e-4)),
            weight_decay=float(p.get("weight_decay", 3e-5)))
        self.to(self.device)

    # ------------------------------------------------------------------ cost
    def _c(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Per-transition cost c(s, a): (N, 1). Clamped so a runaway partition
        term cannot drive exp(-c) to inf/0 (same spirit as AIRL's f_clamp)."""
        a1h = torch.nn.functional.one_hot(act.long(), self.n_actions).float()
        c = self.cost(torch.cat([obs, a1h], dim=-1))
        return torch.clamp(c, -self.c_clamp, self.c_clamp)

    def gcl_reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Policy reward r = -c(s, a), optionally standardized per batch."""
        with torch.no_grad():
            r = -self._c(obs, act)
            if self.reward_norm and r.numel() > 1:
                r = (r - r.mean()) / (r.std() + 1e-6)
            return r

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def _traj_logq(self, obs: torch.Tensor, act: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """log q(tau) under the CURRENT policy (the importance sampler)."""
        return self.policy.log_prob_action(obs, mask, act).sum()

    def step(self, sample_rollouts: list[Rollout],
             demo: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
             n_step: int = 10, rng=None) -> dict:
        """n_step IOC updates. demo entries are (obs, act, action_mask)
        tensors; the action mask is needed to score demos under q."""
        import numpy as np
        rng = rng or np.random.default_rng(0)

        samples = []
        for ro in sample_rollouts:
            t = ro.tensors(self.device)
            m = t["masks"]
            samples.append((t["obs"], t["actions"], m))

        loss_total, n_upd = 0.0, 0
        for _ in range(n_step):
            di = rng.choice(len(demo), min(self.demo_batch, len(demo)),
                            replace=False)
            demo_mb = [demo[i] for i in di]
            # demo term: mean trajectory cost over the demo minibatch
            c_demo = torch.stack([self._c(o.to(self.device),
                                          a.to(self.device)).sum()
                                  for o, a, m in demo_mb])
            # fusion partition: generated rollouts + the demo minibatch,
            # each weighted by 1/q(tau) under the current policy
            ws = []
            for o, a, m in samples + [(o.to(self.device), a.to(self.device),
                                       m.to(self.device))
                                      for o, a, m in demo_mb]:
                logq = self._traj_logq(o, a, m)
                ws.append(-self._c(o, a).sum() - logq)
            w = torch.stack(ws)
            partition = torch.logsumexp(w, dim=0) - torch.log(
                torch.tensor(float(w.numel()), device=self.device))
            loss = c_demo.mean() + partition
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost.parameters(),
                                           self.grad_clip)
            self.optim.step()
            loss_total += float(loss.item())
            n_upd += 1
        return {"ioc_loss": loss_total / max(n_upd, 1)}

    # --------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cost": self.cost.state_dict(),
                    "policy": self.policy.state_dict()}, path)
        log.info("GCL checkpoint saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.cost.load_state_dict(ckpt["cost"])
        self.policy.load_state_dict(ckpt["policy"])
