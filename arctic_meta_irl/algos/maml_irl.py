"""MAML-IRL — Model-Agnostic Meta-Learning for Inverse Reinforcement Learning.

Learns an initial reward parameter vector theta (meta-prior) that can be fast-adapted
to a specific vessel v using inner gradient descent steps on a small support set:
    theta_v' = theta - alpha * grad_theta L_disc(theta; D_v^supp)

Capacity/protocol parity with the AIRL arm (configs/pemirl_noctx.yaml):
same Discriminator MLP (256, 256) with WGAN gradient penalty, same MaskedPolicy
generator, same PPO hyperparameters, rollout budget, and gradient-norm clipping.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

from ..models.discriminator import Discriminator
from ..models.policy import MaskedPolicy
from ..utils.logging import get_logger
from .sampler import Rollout

log = get_logger(__name__)


class MAML_IRL(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, device: str = "cpu"):
        super().__init__()
        p = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.mini_bs = int(p.get("mini_batch_size", 1024))
        self.disc_grad_clip = float(p.get("disc_grad_clip", 1.0))
        self.reward_norm = bool(p.get("reward_norm", True))
        self.grad_penalty = float(p.get("grad_penalty", 10.0))
        self.inner_lr = float(p.get("inner_lr", 0.01))
        self.inner_steps = int(p.get("inner_steps", 1))
        self.first_order = bool(p.get("first_order", True))
        self.device = torch.device(device)

        # Meta-discriminator (capacity-matched to AIRL, context_dim=0)
        self.discriminator = Discriminator(
            obs_dim, n_actions, context_dim=0,
            hidden=tuple(p.get("disc_hidden", (256, 256))),
            state_only=bool(p.get("state_only", False))
        )
        # Action-masked policy generator
        self.policy = MaskedPolicy(
            obs_dim, n_actions, context_dim=0,
            hidden=tuple(p.get("policy_hidden", (256, 256)))
        )
        self.criterion = nn.BCELoss()
        self.optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(p.get("optimizer_lr_discriminator", 3e-4)),
            weight_decay=float(p.get("weight_decay", 3e-5))
        )
        self.to(self.device)

    def _z(self, n: int) -> torch.Tensor:
        """Zero-width context: matching context_dim=0."""
        return torch.zeros(n, 0, device=self.device)

    def adapt_vessel_discriminator(
        self,
        supp_obs: torch.Tensor,
        supp_act: torch.Tensor,
        policy_obs: torch.Tensor,
        policy_act: torch.Tensor,
    ) -> Discriminator:
        """Inner loop: produce fast-adapted Discriminator for a specific vessel."""
        fast_disc = copy.deepcopy(self.discriminator)
        fast_disc.train()
        fast_opt = torch.optim.SGD(fast_disc.parameters(), lr=self.inner_lr)

        z_s = self._z(supp_obs.size(0))
        z_p = self._z(policy_obs.size(0))

        for _ in range(self.inner_steps):
            f_exp = fast_disc.get_unnormed_d(supp_obs, supp_act, z_s)
            f_pol = fast_disc.get_unnormed_d(policy_obs, policy_act, z_p)
            d_exp = torch.clamp(torch.sigmoid(f_exp), 1e-3, 1 - 1e-3)
            d_pol = torch.clamp(torch.sigmoid(f_pol), 1e-3, 1 - 1e-3)

            loss = self.criterion(d_exp, torch.ones_like(d_exp)) + \
                   self.criterion(d_pol, torch.zeros_like(d_pol))
            loss = loss + fast_disc.gradient_penalty(supp_obs, supp_act, z_s, lam=self.grad_penalty)
            loss = loss + fast_disc.gradient_penalty(policy_obs, policy_act, z_p, lam=self.grad_penalty)

            fast_opt.zero_grad()
            loss.backward()
            fast_opt.step()

        return fast_disc

    def step(
        self,
        sample_rollouts: List[Rollout],
        vessel_demos: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        n_step: int = 10,
    ) -> Dict[str, float]:
        """Outer meta-learning step across vessel batches."""
        ts = [ro.tensors(self.device) for ro in sample_rollouts]
        sp = torch.cat([t["obs"] for t in ts])
        ap = torch.cat([t["actions"] for t in ts])

        vessel_ids = list(vessel_demos.keys())
        if not vessel_ids:
            return {"disc_loss": 0.0, "disc_acc_expert": 0.0, "disc_acc_policy": 0.0}

        bce_total, acc_e, acc_p, n_upd = 0.0, 0.0, 0.0, 0
        z_p = self._z(sp.size(0))

        for _ in range(n_step):
            self.optim.zero_grad()
            # Meta-batch sampling
            meta_batch_ids = torch.randperm(len(vessel_ids))[:16].tolist()
            meta_loss = 0.0

            for vid_idx in meta_batch_ids:
                vid = vessel_ids[vid_idx]
                se, ae = vessel_demos[vid]
                se, ae = se.to(self.device), ae.to(self.device)

                if se.size(0) <= 3:
                    continue

                # Split vessel data into 3 support episodes & query set
                supp_n = min(3, se.size(0) // 2)
                supp_s, supp_a = se[:supp_n], ae[:supp_n]
                query_s, query_a = se[supp_n:], ae[supp_n:]

                # 1. Inner adaptation step
                adapted_disc = self.adapt_vessel_discriminator(supp_s, supp_a, sp[:supp_n], ap[:supp_n])

                # 2. Outer query loss on adapted parameters
                z_q = self._z(query_s.size(0))
                f_exp = adapted_disc.get_unnormed_d(query_s, query_a, z_q)
                f_pol = adapted_disc.get_unnormed_d(sp[:query_s.size(0)], ap[:query_s.size(0)], z_q)
                d_exp = torch.clamp(torch.sigmoid(f_exp), 1e-3, 1 - 1e-3)
                d_pol = torch.clamp(torch.sigmoid(f_pol), 1e-3, 1 - 1e-3)

                v_loss = self.criterion(d_exp, torch.ones_like(d_exp)) + \
                         self.criterion(d_pol, torch.zeros_like(d_pol))

                meta_loss = meta_loss + v_loss.item()
                acc_e += float((d_exp > 0.5).float().mean().item())
                acc_p += float((d_pol < 0.5).float().mean().item())
                n_upd += 1

                if self.first_order:
                    v_loss.backward()
                    # Accumulate first-order meta-gradients onto self.discriminator
                    for p_meta, p_adapted in zip(self.discriminator.parameters(), adapted_disc.parameters()):
                        if p_adapted.grad is not None:
                            if p_meta.grad is None:
                                p_meta.grad = p_adapted.grad.detach().clone() / len(meta_batch_ids)
                            else:
                                p_meta.grad += p_adapted.grad.detach().clone() / len(meta_batch_ids)

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_grad_clip)
            self.optim.step()
            bce_total += float(meta_loss.item() if isinstance(meta_loss, torch.Tensor) else meta_loss)

        return {
            "disc_loss": bce_total / max(n_upd, 1),
            "disc_acc_expert": acc_e / max(n_upd, 1),
            "disc_acc_policy": acc_p / max(n_upd, 1),
        }

    # --------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "policy": self.policy.state_dict()
        }, path)
        log.info("MAML-IRL checkpoint saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.policy.load_state_dict(ckpt["policy"])
