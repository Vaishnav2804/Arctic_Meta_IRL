"""04 — Train PEMIRL (meta-IRL with probabilistic context embeddings).

    python scripts/04_train_pemirl.py --config configs/pemirl.yaml \
        --n_traj 1000 --seed 0

per outer iteration
  1. sample (n_sample / repeat_num) train tasks; draw z ~ clipped N(0, I)
     per task and repeat each (task, z) `context_repeat_num` times so the
     Lemma-2 info-max baseline can be formed over repeat groups,
  2. roll the context-conditioned masked policy in the GCRL graph env,
  3. PEMIRL.step(): posterior training (after cnt_starting_iter) ->
     info-max update -> AIRL discriminator BCE with gradient penalties,
  4. PPO generator update with the AIRL reward D = sigmoid(f(s, a, z)).

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/pemirl/pemirl.pt (+ tb logs, history.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.eval.rollout import _episode_tensors, pemirl_log_likelihood
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
    pc = cfg["pemirl"]
    n_traj = args.n_traj or int(pc["n_traj"])
    n_epoch = args.n_epoch or int(pc["n_epoch"])

    mdp, fb, eps = load_pipeline(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    # ---- expert demonstrations (unlabeled — contexts are inferred) ----------
    train_eps = list(eps["train"])
    if len(train_eps) > n_traj:
        train_eps = [train_eps[i] for i in
                     rng.choice(len(train_eps), n_traj, replace=False)]
    demo = [_episode_tensors(ep, fb) for ep in train_eps]
    tasks = tasks_from_episodes(train_eps)
    log.info("PEMIRL: %d expert demos, obs_dim=%d, n_actions=%d, device=%s",
             len(demo), fb.dim, mdp.n_actions, device)

    # ---- model / env / optimizers -------------------------------------------
    model = PEMIRL(obs_dim=fb.dim, n_actions=mdp.n_actions, cfg=pc,
                   device=device)
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    sampler = Sampler(env, device=device)
    ppo = PPO(model.policy,
              lr=float(pc["optimizer_lr_policy"]), clip=float(pc["ppo_clip"]),
              epochs=int(pc["ppo_epochs"]), gae_lambda=float(pc["gae_lambda"]),
              gamma=float(cfg["mdp"]["gamma"]),
              entropy_coef=float(pc["entropy_coef"]),
              value_coef=float(pc["value_coef"]),
              mini_batch_size=int(pc["mini_batch_size"]), device=device)

    ckpt_dir = seeded_dir(Path(pc["ckpt_dir"]), int(cfg["seed"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    def airl_reward(ro):
        t = ro.tensors(device)
        z = ro.z.to(device).unsqueeze(0).expand(len(ro), -1)
        return model.airl_reward(t["obs"], t["actions"], z)

    # ---- outer loop -----------------------------------------------------------
    R = int(pc["context_repeat_num"])
    n_groups = max(int(pc["n_sample"]) // R, 1)
    history, best_ll = [], -float("inf")
    for it in range(n_epoch):
        # tasks + contexts, each repeated R times (consecutively) for Lemma 2
        gidx = rng.choice(len(tasks), n_groups, replace=len(tasks) < n_groups)
        z_groups = model.sample_prior(n_groups)              # (G, Z)
        batch_tasks = [tasks[i] for i in gidx for _ in range(R)]
        z_per_task = z_groups.repeat_interleave(R, dim=0)    # (G*R, Z)

        rollouts = sampler.collect(model.policy, batch_tasks, z_per_task)
        stats = model.step(rollouts, demo, training_itr=it,
                           n_step=int(pc["disc_steps_per_iter"]))
        ppo_stats = ppo.step(rollouts, reward_override=airl_reward,
                             lr_mult=max(1.0 - it / n_epoch, 0.1))

        reached = float(np.mean([ro.reached for ro in rollouts]))
        row = {"iter": it, "reached": reached, **stats, **ppo_stats}
        for k, v in row.items():
            if k != "iter":
                tb.scalar(f"pemirl/{k}", float(v), it)
        if it % 5 == 0:
            log.info("iter %4d | reached %.2f | disc %.4f | pi %.4f%s",
                     it, reached, stats.get("disc_loss", float("nan")),
                     ppo_stats.get("pi_loss", float("nan")),
                     f" | info {stats['info_loss']:.4f}"
                     if "info_loss" in stats else "")

        # ---- periodic support-conditioned val LL + checkpoint -----------------
        if (it + 1) % args.eval_every == 0 and eps.get("val"):
            ll, n = pemirl_log_likelihood(
                model, eps["val"], fb, mdp,
                support_size=int(pc["support_set_size"]), seed=int(cfg["seed"]))
            v = ll / max(n, 1)
            tb.scalar("pemirl/val_ll_per_decision", v, it)
            row["val_ll_per_decision"] = v
            log.info("iter %4d | val LL/decision %.4f", it, v)
            if v > best_ll:
                best_ll = v
                model.save(ckpt_dir / "pemirl.pt")
        history.append(row)

    if best_ll == -float("inf"):  # never evaluated (short run) — save final
        model.save(ckpt_dir / "pemirl.pt")
    tb.close()
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Done. Best val LL/decision: %.4f -> %s",
             best_ll, ckpt_dir / "pemirl.pt")


if __name__ == "__main__":
    main()
