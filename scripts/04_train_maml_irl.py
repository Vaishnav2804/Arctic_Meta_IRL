"""04 — Train MAML-IRL (Model-Agnostic Meta-Learning for Inverse RL).

    python scripts/04_train_maml_irl.py --config configs/maml_irl.yaml \
        --n_traj 1000 --seed 0

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/maml_irl/maml_irl.pt (+ tb logs, history.json)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.maml_irl import MAML_IRL
from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.eval.rollout import _episode_tensors
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n_traj", type=int, default=None,
                   help="expert trajectories used (cf. reference --n_traj 1000)")
    p.add_argument("--n_epoch", type=int, default=None, help="outer iterations")
    p.add_argument("--eval-every", type=int, default=25)
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    mc = cfg["maml_irl"]
    n_traj = args.n_traj or int(mc["n_traj"])
    n_epoch = args.n_epoch or int(mc["n_epoch"])

    mdp, fb, eps = load_pipeline(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    train_eps = list(eps["train"])
    if len(train_eps) > n_traj:
        train_eps = [train_eps[i] for i in
                     rng.choice(len(train_eps), n_traj, replace=False)]
    
    # Group demonstrations by vessel MMSI
    vessel_demos = defaultdict(list)
    for ep in train_eps:
        mmsi = int(getattr(ep, "mmsi", 0))
        o, a = _episode_tensors(ep, fb)
        vessel_demos[mmsi].append((o, a))
    
    vessel_demos_cat = {}
    for mmsi, pairs in vessel_demos.items():
        os = torch.cat([p[0] for p in pairs])
        as_ = torch.cat([p[1] for p in pairs])
        vessel_demos_cat[mmsi] = (os, as_)

    tasks = tasks_from_episodes(train_eps)
    log.info("MAML-IRL: %d expert demos, obs_dim=%d, n_actions=%d, device=%s",
             len(train_eps), fb.dim, mdp.n_actions, device)

    # ---- model / env / optimizers -------------------------------------------
    model = MAML_IRL(obs_dim=fb.dim, n_actions=mdp.n_actions, cfg=mc, device=device)
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    sampler = Sampler(env, device=device)
    ppo = PPO(model.policy,
              lr=float(mc["optimizer_lr_policy"]), clip=float(mc["ppo_clip"]),
              epochs=int(mc["ppo_epochs"]), gae_lambda=float(mc["gae_lambda"]),
              gamma=float(cfg["mdp"]["gamma"]),
              entropy_coef=float(mc["entropy_coef"]),
              value_coef=float(mc["value_coef"]),
              mini_batch_size=int(mc["mini_batch_size"]), device=device)

    ckpt_dir = seeded_dir(Path(mc["ckpt_dir"]), int(cfg["seed"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    def airl_reward(ro):
        t = ro.tensors(device)
        with torch.no_grad():
            return model.discriminator.get_unnormed_d(t["obs"], t["actions"], model._z(len(ro)))

    log.info("MAML-IRL training on %s with %d tasks, %d vessels across %d outer iterations",
             device, len(tasks), len(vessel_demos_cat), n_epoch)

    history = []
    n_sample = int(mc["n_sample"])

    for it in range(n_epoch):
        # Sample tasks
        t_idx = rng.choice(len(tasks), n_sample, replace=len(tasks) < n_sample)
        batch_tasks = [tasks[i] for i in t_idx]

        rollouts = sampler.collect(model.policy, batch_tasks)
        stats = model.step(rollouts, vessel_demos_cat, n_step=int(mc["disc_steps_per_iter"]))
        ppo_stats = ppo.step(rollouts, reward_override=airl_reward,
                             lr_mult=max(1.0 - it / n_epoch, 0.1))

        reached = float(np.mean([ro.reached for ro in rollouts]))
        row = {"iter": it, "reached": reached, **stats, **ppo_stats}
        for k, v in row.items():
            if k != "iter":
                tb.scalar(f"maml_irl/{k}", float(v), it)

        if it == 0 or (it + 1) % 5 == 0 or (it + 1) == n_epoch:
            log.info("iter %4d | reached %.2f | disc %.4f | pi %.4f",
                     it, reached, stats.get("disc_loss", float("nan")),
                     ppo_stats.get("pi_loss", float("nan")))

        history.append(row)

    model.save(ckpt_dir / "maml_irl.pt")
    tb.close()
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Done. Checkpoint saved to %s", ckpt_dir / "maml_irl.pt")


if __name__ == "__main__":
    main()
