"""05 — PPO baseline on the hand-crafted GCRL cost function.

    python scripts/05_train_ppo_baseline.py --config configs/ppo_baseline.yaml

Action-masked PPO in the goal-conditioned graph environment with the
hand-designed reward from the prior goal-conditioned navigation setup
cited in the paper:
    r = -(w_dist * step_km/50 + w_ice * (siconc + sithick)
          + w_wind * |wind|/20 + step_penalty) + goal_bonus.

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/ppo_baseline/policy.pt (+ tb logs)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import base_parser, load_cfg, load_pipeline, log

from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n_epoch", type=int, default=None)
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    pb = cfg["ppo_baseline"]
    n_epoch = args.n_epoch or int(pb["n_epoch"])

    mdp, fb, eps = load_pipeline(cfg)
    tasks = tasks_from_episodes(eps["train"])
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]),
                     hand_reward_cfg=dict(pb))
    sampler = Sampler(env, device=device)

    policy = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=0,
                          hidden=tuple(pb["hidden"]))
    ppo = PPO(policy, lr=float(pb["lr"]), clip=float(pb["ppo_clip"]),
              epochs=int(pb["ppo_epochs"]), gae_lambda=float(pb["gae_lambda"]),
              gamma=float(cfg["mdp"]["gamma"]),
              entropy_coef=float(pb["entropy_coef"]),
              value_coef=float(pb["value_coef"]),
              mini_batch_size=int(pb["mini_batch_size"]), device=device)

    ckpt_dir = Path(pb["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    rng = np.random.default_rng(int(cfg["seed"]))
    best_reached = -1.0
    for it in range(n_epoch):
        idx = rng.choice(len(tasks), int(pb["n_sample"]),
                         replace=len(tasks) < int(pb["n_sample"]))
        rollouts = sampler.collect(policy, [tasks[i] for i in idx])
        stats = ppo.step(rollouts)
        reached = float(np.mean([ro.reached for ro in rollouts]))
        ret = float(np.mean([sum(ro.rewards) for ro in rollouts]))
        tb.scalar("ppo/reached", reached, it)
        tb.scalar("ppo/return", ret, it)
        for k, v in stats.items():
            tb.scalar(f"ppo/{k}", float(v), it)
        if it % 5 == 0:
            log.info("iter %4d | reached %.2f | return %8.2f | pi %.4f",
                     it, reached, ret, stats["pi_loss"])
        if reached >= best_reached:
            best_reached = reached
            torch.save(policy.state_dict(), ckpt_dir / "policy.pt")

    tb.close()
    log.info("Done. Best goal-reach rate %.2f -> %s",
             best_reached, ckpt_dir / "policy.pt")


if __name__ == "__main__":
    main()
