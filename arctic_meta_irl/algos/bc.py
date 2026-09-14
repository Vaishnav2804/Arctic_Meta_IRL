"""Behavior cloning — supervised maximum-likelihood baselines.

Two policy families share one trainer:

* ``arch: mlp`` — :class:`~arctic_meta_irl.models.policy.MaskedPolicy` with
  ``context_dim=0``: per-state pi(a | phi(s)), the classic BC baseline. Same
  input features, masking, and hidden sizes as the AIRL/PEMIRL generator, so
  the comparison isolates the *training signal* (supervised vs adversarial),
  not the architecture.
* ``arch: lstm | transformer`` —
  :class:`~arctic_meta_irl.models.seq_policy.SequencePolicy`: teacher-forced
  next-action prediction with the episode's own prefix as context.

Training maximizes exactly the evaluation metric (per-decision LL) on the
train split, with early stopping on the val split — no reward is learned, so
this is the "how far does pure imitation get you" reference for the IRL
ladder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.dataset import Episode, make_dataloader
from ..env.graph_mdp import GraphMDP
from ..features import FeatureBuilder
from ..models.policy import MaskedPolicy
from ..models.seq_policy import SequencePolicy
from ..utils.logging import get_logger

log = get_logger(__name__)


def build_policy(arch: str, obs_dim: int, n_actions: int, cfg: dict):
    if arch == "mlp":
        return MaskedPolicy(obs_dim, n_actions, context_dim=0,
                            hidden=tuple(cfg.get("hidden", (256, 256))))
    return SequencePolicy(obs_dim, n_actions, arch=arch,
                          hidden=int(cfg.get("seq_hidden", 128)),
                          layers=int(cfg.get("seq_layers", 1)),
                          nhead=int(cfg.get("seq_nhead", 4)),
                          dropout=float(cfg.get("seq_dropout", 0.1)))


def batch_log_probs(policy, batch: dict, mdp: GraphMDP,
                    device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T) per-step log pi(a_t|.) and the boolean pad mask, either arch."""
    phi = batch["phi"].to(device)
    act = batch["actions"].to(device)
    pad = batch["mask"].to(device)
    amask = torch.from_numpy(
        mdp.action_mask[batch["states"].numpy()]).to(device)
    if isinstance(policy, SequencePolicy):
        lp = policy.log_probs(phi, act, amask)
    else:
        B, T, D = phi.shape
        lp = policy.dist(phi.reshape(B * T, D),
                         amask.reshape(B * T, -1)).log_prob(
            act.reshape(B * T)).reshape(B, T)
    return lp, pad


@torch.no_grad()
def eval_ll(policy, episodes: list[Episode], features: FeatureBuilder,
            mdp: GraphMDP, device: str, batch_size: int = 32
            ) -> tuple[float, int]:
    """Total and decision-count LL over a split (teacher-forced for seq)."""
    policy.eval()
    loader = make_dataloader(episodes, features, batch_size=batch_size,
                             shuffle=False)
    total, n = 0.0, 0
    for batch in loader:
        lp, pad = batch_log_probs(policy, batch, mdp, device)
        total += float(lp[pad].sum().item())
        n += int(pad.sum().item())
    return total, n


@torch.no_grad()
def per_episode_ll(policy, episodes: list[Episode], features: FeatureBuilder,
                   mdp: GraphMDP, device: str) -> dict[int, float]:
    """{voyage_index: total LL} — the eval-script per-episode dump."""
    policy.eval()
    out = {}
    for ep in episodes:
        phi = torch.from_numpy(features.episode_matrix(
            ep.states[:-1], ep.goal, ep.year, ep.month, ep.mmsi)
        ).unsqueeze(0).to(device)
        act = torch.from_numpy(ep.actions.astype(np.int64)
                               ).unsqueeze(0).to(device)
        amask = torch.from_numpy(
            mdp.action_mask[ep.states[:-1]]).unsqueeze(0).to(device)
        if isinstance(policy, SequencePolicy):
            lp = policy.log_probs(phi, act, amask)
        else:
            lp = policy.dist(phi[0], amask[0]).log_prob(act[0])
        out[ep.voyage_index] = float(lp.sum().item())
    return out


def train_bc(policy, train_eps: list[Episode], val_eps: list[Episode],
             features: FeatureBuilder, mdp: GraphMDP, cfg: dict,
             device: str, ckpt: Path, tb=None) -> dict:
    """Early-stopped supervised training; saves the best-val checkpoint."""
    policy.to(device)
    optim = torch.optim.Adam(policy.parameters(),
                             lr=float(cfg.get("lr", 3e-4)),
                             weight_decay=float(cfg.get("weight_decay", 1e-5)))
    epochs = int(cfg.get("epochs", 200))
    patience = int(cfg.get("patience", 20))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    loader = make_dataloader(train_eps, features,
                             batch_size=int(cfg.get("batch_size", 32)),
                             shuffle=True)
    best = {"val_ll_per_decision": -float("inf"), "epoch": -1}
    since_best = 0
    for epoch in range(epochs):
        policy.train()
        tot, n = 0.0, 0
        for batch in loader:
            lp, pad = batch_log_probs(policy, batch, mdp, device)
            loss = -(lp[pad]).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optim.step()
            tot += float(lp[pad].sum().item())
            n += int(pad.sum().item())
        train_ll = tot / max(n, 1)
        vll, vn = eval_ll(policy, val_eps, features, mdp, device)
        v = vll / max(vn, 1)
        if tb:
            tb.scalar("bc/train_ll_per_decision", train_ll, epoch)
            tb.scalar("bc/val_ll_per_decision", v, epoch)
        if epoch % 5 == 0:
            log.info("epoch %3d | train LL/dec %.4f | val LL/dec %.4f",
                     epoch, train_ll, v)
        if v > best["val_ll_per_decision"]:
            best.update(val_ll_per_decision=v, epoch=epoch)
            since_best = 0
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"policy": policy.state_dict()}, ckpt)
        else:
            since_best += 1
            if since_best >= patience:
                log.info("early stop at epoch %d (best %.4f @ %d)",
                         epoch, best["val_ll_per_decision"], best["epoch"])
                break
    return best
