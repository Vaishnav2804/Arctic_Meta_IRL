"""GAIL — Generative Adversarial Imitation Learning (Ho & Ermon, 2016).

Occupancy-measure matching with an *unstructured* discriminator: D(s, a)
classifies expert vs generated transitions and the generator (action-masked
PPO) is rewarded with the non-saturating surrogate r = -log(1 - D). Unlike
AIRL, f is not interpreted as a reward — nothing transferable is recovered —
which is exactly why GAIL is the control separating "adversarial imitation
matches the expert" from "the adversarial *reward* is meaningful".

Capacity/protocol parity with the AIRL arm (``configs/pemirl_noctx.yaml``):
same Discriminator MLP (256, 256) with WGAN gradient penalty, same
MaskedPolicy generator, same PPO hyperparameters, rollout budget, and
discriminator steps per outer iteration, same gradient-norm clipping, and the
same per-batch reward standardization (``reward_norm``) so the policy's
learning signal is scaled identically. The only changed factor is the reward
surrogate (-log(1-D) instead of the AIRL logit f).
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..models.discriminator import Discriminator
from ..models.policy import MaskedPolicy
from ..utils.logging import get_logger
from .sampler import Rollout

log = get_logger(__name__)


class GAIL(torch.nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict,
                 device: str = "cpu"):
        super().__init__()
        p = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.mini_bs = int(p.get("mini_batch_size", 1024))
        self.disc_grad_clip = float(p.get("disc_grad_clip", 1.0))
        self.reward_norm = bool(p.get("reward_norm", True))
        self.grad_penalty = float(p.get("grad_penalty", 10.0))
        self.device = torch.device(device)

        self.discriminator = Discriminator(
            obs_dim, n_actions, context_dim=0,
            hidden=tuple(p.get("disc_hidden", (256, 256))),
            state_only=bool(p.get("state_only", False)))
        self.policy = MaskedPolicy(
            obs_dim, n_actions, context_dim=0,
            hidden=tuple(p.get("policy_hidden", (256, 256))))
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(p.get("optimizer_lr_discriminator", 3e-4)),
            weight_decay=float(p.get("weight_decay", 3e-5)))
        self.to(self.device)

    def _z(self, n: int) -> torch.Tensor:
        """Zero-width context: reuses the context-conditioned Discriminator
        with context_dim=0, exactly as the pooled-AIRL ablation does."""
        return torch.zeros(n, 0, device=self.device)

    def gail_reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """r = -log(1 - D(s,a)) — Ho & Ermon's non-saturating surrogate.
        D is clamped to [1e-3, 1-1e-3] (as in the BCE step), which bounds r to
        (0, log 1000], then optionally standardized per batch (reward_norm)."""
        with torch.no_grad():
            f = self.discriminator.get_unnormed_d(obs, act, self._z(obs.size(0)))
            d = torch.clamp(torch.sigmoid(f), 1e-3, 1 - 1e-3)
            r = -torch.log(1.0 - d)
            if self.reward_norm and r.numel() > 1:
                r = (r - r.mean()) / (r.std() + 1e-6)
            return r

    def step(self, sample_rollouts: list[Rollout],
             demo: list[tuple[torch.Tensor, torch.Tensor]],
             n_step: int = 10) -> dict:
        """Discriminator BCE on transitions (expert=1, generated=0) with
        gradient penalties on both populations — the AIRL-arm protocol minus
        everything context-related."""
        ts = [ro.tensors(self.device) for ro in sample_rollouts]
        sp = torch.cat([t["obs"] for t in ts])
        ap = torch.cat([t["actions"] for t in ts])
        se = torch.cat([o.to(self.device) for o, a in demo])
        ae = torch.cat([a.to(self.device) for o, a in demo])

        bce_total, acc_e, acc_p, n_upd = 0.0, 0.0, 0.0, 0
        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                f_b = self.discriminator.get_unnormed_d(
                    sp[ind_p], ap[ind_p], self._z(ind_p.size(0)))
                f_e = self.discriminator.get_unnormed_d(
                    se[ind_e], ae[ind_e], self._z(ind_e.size(0)))
                d_b = torch.clamp(torch.sigmoid(f_b), 1e-3, 1 - 1e-3)
                d_e = torch.clamp(torch.sigmoid(f_e), 1e-3, 1 - 1e-3)
                loss = (self.criterion(d_b, torch.zeros_like(d_b))
                        + self.criterion(d_e, torch.ones_like(d_e)))
                loss = loss + self.discriminator.gradient_penalty(
                    sp[ind_p], ap[ind_p], self._z(ind_p.size(0)),
                    lam=self.grad_penalty)
                loss = loss + self.discriminator.gradient_penalty(
                    se[ind_e], ae[ind_e], self._z(ind_e.size(0)),
                    lam=self.grad_penalty)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.disc_grad_clip)
                self.optim.step()
                bce_total += float(loss.item())
                acc_e += float((d_e > 0.5).float().mean().item())
                acc_p += float((d_b < 0.5).float().mean().item())
                n_upd += 1
        return {"disc_loss": bce_total / max(n_upd, 1),
                "disc_acc_expert": acc_e / max(n_upd, 1),
                "disc_acc_policy": acc_p / max(n_upd, 1)}

    # --------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"discriminator": self.discriminator.state_dict(),
                    "policy": self.policy.state_dict()}, path)
        log.info("GAIL checkpoint saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.policy.load_state_dict(ckpt["policy"])
