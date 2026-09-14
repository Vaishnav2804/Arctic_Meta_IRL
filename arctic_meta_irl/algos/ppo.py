"""Action-masked PPO with GAE — generator for PEMIRL and the hand-crafted-
reward baseline (prior goal-conditioned navigation setup cited in the paper)."""
from __future__ import annotations

import numpy as np
import torch

from .sampler import Rollout
from ..models.policy import MaskedPolicy


class PPO:
    def __init__(self, policy: MaskedPolicy, lr: float = 3e-4, clip: float = 0.2,
                 epochs: int = 8, gae_lambda: float = 0.95, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 mini_batch_size: int = 1024, device: str = "cpu",
                 max_grad_norm: float = 0.5):
        self.policy = policy.to(device)
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip, self.epochs = clip, epochs
        self.lam, self.gamma = gae_lambda, gamma
        self.ent_c, self.val_c = entropy_coef, value_coef
        self.mbs = mini_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm

    # -------------------------------------------------------------- update
    def step(self, rollouts: list[Rollout], reward_override=None,
             lr_mult: float = 1.0) -> dict:
        """One PPO update over a batch of rollouts.

        reward_override(ro) -> (T,) tensor lets PEMIRL substitute the AIRL
        reward for the (zero / environment) reward stored in the rollout.
        """
        OBS, ACT, MSK, LP, ADV, RET, Z = [], [], [], [], [], [], []
        for ro in rollouts:
            t = ro.tensors(self.device)
            rew = (reward_override(ro).to(self.device).flatten()
                   if reward_override is not None else t["rewards"])
            adv, ret = self._gae(rew, t["values"])
            OBS.append(t["obs"]); ACT.append(t["actions"]); MSK.append(t["masks"])
            LP.append(t["logps"]); ADV.append(adv); RET.append(ret)
            if ro.z is not None:
                Z.append(ro.z.to(self.device).unsqueeze(0).expand(len(ro), -1))
        obs = torch.cat(OBS); act = torch.cat(ACT); msk = torch.cat(MSK)
        lp_old = torch.cat(LP); adv = torch.cat(ADV); ret = torch.cat(RET)
        z = torch.cat(Z) if Z else None
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for g in self.optim.param_groups:
            g["lr"] = g.get("initial_lr", g["lr"]) * lr_mult

        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        for _ in range(self.epochs):
            idx = torch.randperm(obs.size(0), device=self.device)
            for mb in idx.split(self.mbs):
                zz = z[mb] if z is not None else None
                d = self.policy.dist(obs[mb], msk[mb], zz)
                lp = d.log_prob(act[mb])
                ratio = torch.exp(lp - lp_old[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v = self.policy.value(obs[mb], zz)
                v_loss = torch.nn.functional.mse_loss(v, ret[mb])
                ent = d.entropy().mean()
                loss = pi_loss + self.val_c * v_loss - self.ent_c * ent
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               self.max_grad_norm)
                self.optim.step()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                n_updates += 1
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    def _gae(self, rew: torch.Tensor, val: torch.Tensor):
        T = rew.size(0)
        adv = torch.zeros(T, device=self.device)
        last = 0.0
        for t in reversed(range(T)):
            v_next = val[t + 1] if t + 1 < T else torch.tensor(0.0, device=self.device)
            delta = rew[t] + self.gamma * v_next - val[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        return adv, adv + val
