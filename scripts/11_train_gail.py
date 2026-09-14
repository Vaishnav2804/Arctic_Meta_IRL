"""11 — Train GAIL (adversarial imitation, no reward recovery).

    python scripts/11_train_gail.py --config configs/gail.yaml --seed 0

Mirrors scripts/04_train_pemirl.py minus everything context-related: per
outer iteration (1) roll the masked policy on train tasks, (2) discriminator
BCE with gradient penalties (expert=1 vs generated=0), (3) PPO on the
non-saturating surrogate reward -log(1 - D). Budget and hyperparameters are
protocol-matched to the AIRL arm.

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/gail/gail.pt (+ tb logs, history.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.gail import GAIL
from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.eval.rollout import _episode_tensors
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def policy_val_ll(model: GAIL, episodes, fb, mdp) -> tuple[float, int]:
    """Per-decision LL of the generator policy on a split (no context)."""
    import torch
    total, n = 0.0, 0
    with torch.no_grad():
        for ep in episodes:
            phi, act = _episode_tensors(ep, fb)
            phi, act = phi.to(model.device), act.to(model.device)
            mask = torch.from_numpy(
                mdp.action_mask[ep.states[:-1]]).to(model.device)
            lp = model.policy.log_prob_action(phi, mask, act)
            total += float(lp.sum().item())
            n += len(ep)
    return total, n


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n_traj", type=int, default=None)
    p.add_argument("--n_epoch", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=25)
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    gc = cfg["gail"]
    n_traj = args.n_traj or int(gc["n_traj"])
    n_epoch = args.n_epoch or int(gc["n_epoch"])

    mdp, fb, eps = load_pipeline(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    train_eps = list(eps["train"])
    if len(train_eps) > n_traj:
        train_eps = [train_eps[i] for i in
                     rng.choice(len(train_eps), n_traj, replace=False)]
    demo = [_episode_tensors(ep, fb) for ep in train_eps]
    tasks = tasks_from_episodes(train_eps)
    log.info("GAIL: %d expert demos, obs_dim=%d, n_actions=%d, device=%s",
             len(demo), fb.dim, mdp.n_actions, device)

    model = GAIL(obs_dim=fb.dim, n_actions=mdp.n_actions, cfg=gc,
                 device=device)
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    sampler = Sampler(env, device=device)
    ppo = PPO(model.policy,
              lr=float(gc["optimizer_lr_policy"]), clip=float(gc["ppo_clip"]),
              epochs=int(gc["ppo_epochs"]), gae_lambda=float(gc["gae_lambda"]),
              gamma=float(cfg["mdp"]["gamma"]),
              entropy_coef=float(gc["entropy_coef"]),
              value_coef=float(gc["value_coef"]),
              mini_batch_size=int(gc["mini_batch_size"]), device=device)

    ckpt_dir = seeded_dir(Path(gc["ckpt_dir"]), int(cfg["seed"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    def gail_reward(ro):
        t = ro.tensors(device)
        return model.gail_reward(t["obs"], t["actions"])

    history, best_ll = [], -float("inf")
    for it in range(n_epoch):
        idx = rng.choice(len(tasks), int(gc["n_sample"]),
                         replace=len(tasks) < int(gc["n_sample"]))
        rollouts = sampler.collect(model.policy, [tasks[i] for i in idx])
        stats = model.step(rollouts, demo,
                           n_step=int(gc["disc_steps_per_iter"]))
        ppo_stats = ppo.step(rollouts, reward_override=gail_reward,
                             lr_mult=max(1.0 - it / n_epoch, 0.1))

        reached = float(np.mean([ro.reached for ro in rollouts]))
        row = {"iter": it, "reached": reached, **stats, **ppo_stats}
        for k, v in row.items():
            if k != "iter":
                tb.scalar(f"gail/{k}", float(v), it)
        if it % 5 == 0:
            log.info("iter %4d | reached %.2f | disc %.4f | pi %.4f",
                     it, reached, stats.get("disc_loss", float("nan")),
                     ppo_stats.get("pi_loss", float("nan")))

        if (it + 1) % args.eval_every == 0 and eps.get("val"):
            ll, n = policy_val_ll(model, eps["val"], fb, mdp)
            v = ll / max(n, 1)
            tb.scalar("gail/val_ll_per_decision", v, it)
            row["val_ll_per_decision"] = v
            log.info("iter %4d | val LL/decision %.4f", it, v)
            if v > best_ll:
                best_ll = v
                model.save(ckpt_dir / "gail.pt")
        history.append(row)

    if best_ll == -float("inf"):
        model.save(ckpt_dir / "gail.pt")
    tb.close()
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Done. Best val LL/decision: %.4f -> %s",
             best_ll, ckpt_dir / "gail.pt")


if __name__ == "__main__":
    main()
