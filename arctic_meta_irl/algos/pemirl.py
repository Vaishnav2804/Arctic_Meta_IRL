"""PEMIRL — Probabilistic Embeddings for Meta-IRL (Yu et al., 2019), AIRL-style.

Adapted from ``model/pemirl_airl.py`` of
https://github.com/LucasCJYSDL/Multi-task-Hierarchical-AIRL (the codebase run
as ``python run_metaIRL_baselines.py --env_name XXX --n_traj 1000 --algo PEMIRL
--seed YYY``) to the discrete, action-masked, goal-conditioned Arctic graph
MDP. The three training signals are kept intact:

1. **Context posterior training** — bi-LSTM q(z|tau) maximizes log q(z|tau) on
   generated rollouts whose true conditioning z is known (starts after
   ``cnt_starting_iter`` outer iterations, as in the reference).
2. **Info-max (Lemma 2)** — couples the posterior and the discriminator:
   maximize E[ log q(z|tau) * (f_sum(tau) - baseline) ], where the baseline is
   the mean f_sum over ``context_repeat_num`` rollouts sharing a context.
3. **AIRL BCE** — discriminator f(s,a,z) classifies expert vs generated
   transitions through D = exp(f)/(exp(f)+1), with gradient penalties on both.

Expert demonstrations carry *no* task labels: their z is inferred by the
posterior (``convert_demo``). The generator is action-masked PPO conditioned
on z, rewarded with the AIRL reward (see fix (iv) below).

Stability fixes (methodological contribution of the paper)
----------------------------------------------------------
A naive port of the released PEMIRL code diverges on this MDP: the info-max
term back-propagates into the discriminator through f_sum(tau), and because f
is unbounded, maximizing E[log q(z|tau) * f_sum] drives f -> infinity. In a
500-iteration run this blew the discriminator BCE loss up to ~1e11, saturated
sigmoid(f) rewards, and collapsed the policy (validation log-likelihood
frozen). Six coupled fixes prevent this; each MUST be preserved exactly:

(i)   **f_clamp** — the per-transition f used inside the info-max target is
      clamped to +-``f_clamp`` (default 10.0) BEFORE summing over the
      trajectory, so f_sum — and hence the incentive to inflate f — is bounded.
(ii)  **Per-batch advantage standardization** — the info-max advantage
      (f_sum - baseline) is divided by its batch std, so only its SHAPE (which
      rollouts score higher than their context-mates), not its absolute scale,
      drives the posterior/discriminator gradient.
(iii) **Gradient-norm clipping** (``disc_grad_clip``, default 1.0) on both the
      discriminator and posterior optimizers, in the info-max step AND the
      AIRL BCE step (PPO already clips the policy).
(iv)  **reward_logit** — the generator reward is the AIRL logit r = f
      (clamped, then per-batch standardized when ``reward_norm``) instead of
      the released-code r = sigmoid(f), which saturates as |f| grows and
      starves the policy of learning signal.
(v)   **info_coeff** — the info-max loss is scaled by ``info_coeff``
      (default 0.1), plumbed through the config, so the coupling term cannot
      dominate the discriminator's BCE objective.
(vi)  **cnt_starting_iter** — posterior maximum-likelihood training is delayed
      until after ``cnt_starting_iter`` outer iterations (default 5), so
      q(z|tau) is not fit to rollouts from a still-random policy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..models.context_net import ContextPosterior
from ..models.discriminator import Discriminator
from ..models.policy import MaskedPolicy
from ..utils.logging import get_logger
from .sampler import Rollout

log = get_logger(__name__)


class PEMIRL(torch.nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict,
                 device: str = "cpu"):
        super().__init__()
        p = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.dim_cnt = int(p["context_dim"])
        self.cnt_limit = float(p.get("context_limit", 2.0))
        self.info_coeff = float(p.get("info_coeff", 0.1))
        self.cnt_sampling_fixed = bool(p.get("cnt_sampling_fixed", False))
        self.cnt_training_iters = int(p.get("cnt_training_iterations", 2))
        self.cnt_starting_iter = int(p.get("cnt_starting_iter", 5))
        self.info_training_iters = int(p.get("info_training_iters", 2))
        self.repeat_num = int(p.get("context_repeat_num", 4))
        self.mini_bs = int(p.get("mini_batch_size", 1024))
        self.device = torch.device(device)

        # ---- stability knobs (added after a 500-iter run diverged: the
        # info-max term trained the discriminator to maximize an UNBOUNDED f,
        # blowing disc loss to ~1e11 and freezing val LL). ------------------
        # clamp |f| in the info-max target so f_sum can't run to infinity
        self.f_clamp = float(p.get("f_clamp", 10.0))
        # gradient-norm clip on the discriminator optimizer (PPO already clips)
        self.disc_grad_clip = float(p.get("disc_grad_clip", 1.0))
        # standardize the AIRL reward per batch before PPO consumes it, so a
        # saturating sigmoid(f) can't collapse the policy's learning signal
        self.reward_norm = bool(p.get("reward_norm", True))
        # use the principled AIRL logit reward r = f - log pi(a|s,z) instead of
        # the released-code r = sigmoid(f) (which saturates as f grows)
        self.reward_logit = bool(p.get("reward_logit", True))

        self.discriminator = Discriminator(
            obs_dim, n_actions, self.dim_cnt,
            hidden=tuple(p.get("disc_hidden", (256, 256))),
            state_only=bool(p.get("state_only", False)))
        self.policy = MaskedPolicy(
            obs_dim, n_actions, context_dim=self.dim_cnt,
            hidden=tuple(p.get("policy_hidden", (256, 256))))
        # optional static vessel-metadata prior on the context posterior: the
        # metadata is the last ``meta_dim`` features of the observation (the
        # vessel-static block of phi), given a dedicated encoder pathway into z.
        self.meta_prior = bool(p.get("meta_prior", False)) and bool(self.dim_cnt)
        meta_slice = ((obs_dim - int(p["meta_dim"]), obs_dim)
                      if self.meta_prior else None)
        self.context_posterior = ContextPosterior(
            input_dim=obs_dim + n_actions,
            hidden_dim=int(p.get("bi_lstm_hidden", 128)),
            context_dim=self.dim_cnt, context_limit=self.cnt_limit,
            meta_slice=meta_slice, meta_hidden=int(p.get("meta_hidden", 32)))

        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(p.get("optimizer_lr_discriminator", 3e-4)),
            weight_decay=float(p.get("weight_decay", 3e-5)))
        self.context_optim = torch.optim.Adam(
            self.context_posterior.parameters(), weight_decay=1e-3)
        self.grad_penalty = float(p.get("grad_penalty", 10.0))
        self.to(self.device)

    # ------------------------------------------------------------ helpers
    def _posterior_input(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        a1h = F.one_hot(act.long(), self.n_actions).float()
        return torch.cat([obs, a1h], dim=-1).unsqueeze(0)  # (1, T, D+A)

    def sample_prior(self, n: int) -> torch.Tensor:
        """z ~ N(0, I), clamped — contexts to condition generator rollouts on."""
        z = torch.randn(n, self.dim_cnt, device=self.device)
        return torch.clamp(z, -self.cnt_limit, self.cnt_limit)

    @torch.no_grad()
    def infer_context(self, obs: torch.Tensor, act: torch.Tensor,
                      fixed: bool | None = None) -> torch.Tensor:
        """Posterior context for one trajectory. obs (T, D), act (T,) -> (1, Z)."""
        fixed = self.cnt_sampling_fixed if fixed is None else fixed
        x = self._posterior_input(obs.to(self.device), act.to(self.device))
        return self.context_posterior.sample_context(x, fixed=fixed)

    @torch.no_grad()
    def infer_context_support(self, episodes_tensors: list[tuple[torch.Tensor, torch.Tensor]],
                              fixed: bool = True) -> torch.Tensor:
        """Test-time inference from a small support set: average posterior means.

        An empty support set (the k=0 / no-adaptation arm of the support-size
        ablation) falls back to the prior mean z = 0."""
        if not episodes_tensors:
            return torch.zeros(1, self.dim_cnt, device=self.device)
        means = []
        for obs, act in episodes_tensors:
            x = self._posterior_input(obs.to(self.device), act.to(self.device))
            mean, _ = self.context_posterior(x)
            means.append(mean)
        z = torch.stack(means).mean(dim=0)
        return torch.clamp(z, -self.cnt_limit, self.cnt_limit)

    def airl_reward(self, obs: torch.Tensor, act: torch.Tensor,
                    z: torch.Tensor) -> torch.Tensor:
        """Generator reward.

        Two forms (config ``reward_logit``):
          - logit (default, principled AIRL): r = f  (clamped). The full AIRL
            reward is f - log pi(a|s,z); the -log pi entropy bonus is already
            supplied by PPO's entropy_coef, so r = f is the transferable part.
            Unlike sigmoid(f), this does NOT saturate as f grows.
          - sigmoid (released-code): r = sigmoid(f). Kept for ablation.
        Reward is clamped to +-f_clamp and (optionally) standardized per batch
        so the policy always gets a finite, well-scaled signal.
        """
        with torch.no_grad():
            f = self.discriminator.get_unnormed_d(obs, act, z)
            f = torch.clamp(f, -self.f_clamp, self.f_clamp)
            r = f if self.reward_logit else torch.sigmoid(f)
            if self.reward_norm and r.numel() > 1:
                r = (r - r.mean()) / (r.std() + 1e-6)
            return r

    # ---------------------------------------------------------- convert_demo
    @torch.no_grad()
    def convert_demo(self, demo: list[tuple[torch.Tensor, torch.Tensor]]
                     ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Expert (obs, act) -> (obs, act, z) with posterior-inferred contexts."""
        out = []
        for obs, act in demo:
            obs, act = obs.to(self.device), act.to(self.device)
            z = self.infer_context(obs, act)                 # (1, Z)
            out.append((obs, act, z.expand(obs.size(0), -1)))
        return out

    # ----------------------------------------------------------------- step
    def step(self, sample_rollouts: list[Rollout],
             demo: list[tuple[torch.Tensor, torch.Tensor]],
             training_itr: int, n_step: int = 10) -> dict:
        stats = {}

        # rollouts -> tensors, grouped with their conditioning contexts
        samples = []
        for ro in sample_rollouts:
            t = ro.tensors(self.device)
            z = ro.z.to(self.device).unsqueeze(0)            # (1, Z) true context
            samples.append((t["obs"], t["actions"], z))

        # ---- (1) context posterior training on generated rollouts ----------
        # With context_dim == 0 (the pooled-AIRL ablation) the posterior and
        # the info-max term are vacuous (zero-width z) — skip both entirely.
        if self.dim_cnt and training_itr > self.cnt_starting_iter:
            for _ in range(self.cnt_training_iters):
                cnt_loss = torch.tensor(0.0, device=self.device)
                for obs, act, z in samples:
                    x = self._posterior_input(obs, act)
                    cnt_loss = cnt_loss - self.context_posterior.log_prob_context(x, z).mean()
                cnt_loss = cnt_loss / max(len(samples), 1)
                self.context_optim.zero_grad()
                cnt_loss.backward()
                self.context_optim.step()
            stats["cnt_loss"] = float(cnt_loss.item())

        # ---- (2) info-max objective (Lemma 2 of the PEMIRL paper) ----------
        # NOTE: this term back-props into the discriminator via f_sum. Without a
        # bound on f it drives f -> inf (the 500-iter divergence). We clamp the
        # per-transition f used here to +-f_clamp BEFORE summing, and clip the
        # discriminator grad norm, so the info-max signal stays finite.
        for _ in range(self.info_training_iters if self.dim_cnt else 0):
            logp_list, f_sum_list = [], []
            for obs, act, z in samples:
                x = self._posterior_input(obs, act)
                logp_list.append(self.context_posterior.log_prob_context(x, z))
                zT = z.expand(obs.size(0), -1)
                f = self.discriminator.get_unnormed_d(obs, act, zT)
                f = torch.clamp(f, -self.f_clamp, self.f_clamp)
                f_sum_list.append(f.sum(dim=0, keepdim=True))
            logp = torch.cat(logp_list, dim=0)               # (B, 1)
            f_sum = torch.cat(f_sum_list, dim=0)             # (B, 1)
            R = self.repeat_num
            usable = (f_sum.size(0) // R) * R
            if usable >= R:
                f_u, lp_u = f_sum[:usable], logp[:usable]
                base = f_u.view(-1, R, 1).mean(dim=1, keepdim=True) \
                          .expand(-1, R, -1).reshape(-1, 1)
                # standardize the advantage (f_sum - base) per batch so its
                # scale can't grow with f; only its SHAPE drives the gradient.
                adv = f_u - base
                adv = adv / (adv.std() + 1e-6)
                info_loss = -self.info_coeff * (lp_u * adv).mean()
                self.optim.zero_grad()
                self.context_optim.zero_grad()
                info_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.disc_grad_clip)
                torch.nn.utils.clip_grad_norm_(
                    self.context_posterior.parameters(), self.disc_grad_clip)
                self.optim.step()
                self.context_optim.step()
                stats["info_loss"] = float(info_loss.item())

        # ---- (3) AIRL BCE on transitions ------------------------------------
        demo_z = self.convert_demo(demo)
        sp = torch.cat([o for o, a, z in samples])
        ap = torch.cat([a for o, a, z in samples])
        zp = torch.cat([z.expand(o.size(0), -1) for o, a, z in samples])
        se = torch.cat([o for o, a, z in demo_z])
        ae = torch.cat([a for o, a, z in demo_z])
        ze = torch.cat([z for o, a, z in demo_z])

        bce_total, n_upd = 0.0, 0
        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                f_b = self.discriminator.get_unnormed_d(sp[ind_p], ap[ind_p], zp[ind_p])
                f_e = self.discriminator.get_unnormed_d(se[ind_e], ae[ind_e], ze[ind_e])
                d_b = torch.clamp(torch.sigmoid(f_b), 1e-3, 1 - 1e-3)
                d_e = torch.clamp(torch.sigmoid(f_e), 1e-3, 1 - 1e-3)
                loss = (self.criterion(d_b, torch.zeros_like(d_b))
                        + self.criterion(d_e, torch.ones_like(d_e)))
                loss = loss + self.discriminator.gradient_penalty(
                    sp[ind_p], ap[ind_p], zp[ind_p], lam=self.grad_penalty)
                loss = loss + self.discriminator.gradient_penalty(
                    se[ind_e], ae[ind_e], ze[ind_e], lam=self.grad_penalty)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.disc_grad_clip)
                self.optim.step()
                bce_total += float(loss.item())
                n_upd += 1
        stats["disc_loss"] = bce_total / max(n_upd, 1)
        return stats

    # --------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"discriminator": self.discriminator.state_dict(),
                    "policy": self.policy.state_dict(),
                    "context_posterior": self.context_posterior.state_dict()},
                   path)
        log.info("PEMIRL checkpoint saved to %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.policy.load_state_dict(ckpt["policy"])
        self.context_posterior.load_state_dict(ckpt["context_posterior"])
