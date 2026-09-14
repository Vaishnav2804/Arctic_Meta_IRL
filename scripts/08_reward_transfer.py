"""08 — Validate the learned PEMIRL reward by deploying it (reward transfer).

    # full training run (fresh PPO agent on the frozen PEMIRL reward)
    python scripts/08_reward_transfer.py --config configs/pemirl.yaml \
        --pem-ckpt runs/pemirl/pemirl.pt --n_epoch 200

    # reproduce the paper figure/summary from the released transfer policy
    python scripts/08_reward_transfer.py --config configs/pemirl.yaml \
        --released --eval-only

    # CONTROL: same fresh-PPO harness on the frozen MCE-IRL linear reward
    # (no context, no support sets) -> runs/ppo_on_mce/
    python scripts/08_reward_transfer.py --config configs/pemirl.yaml \
        --reward mce --released

Question this answers: *is the reward PEMIRL recovered actually USEFUL?* High
test log-likelihood (06_evaluate) says the reward ranks the expert's next move
well, but the real test of an IRL reward is whether a FRESH agent, trained only
on that reward, learns to navigate.

Setup:
  - Load the trained PEMIRL discriminator f(s,a,z) + bi-LSTM posterior; FREEZE both.
  - Per training vessel, infer z from a small support set of that vessel's
    episodes (exactly as eval / paper §II does).
  - Train a brand-new context-conditioned MaskedPolicy with PPO whose per-step
    reward is r = f(s, a, z) from the frozen discriminator (no hand-crafted cost,
    no expert labels at train time beyond the support set used to infer z).
  - Track goal-reach rate over training. If it climbs, the learned reward
    teaches navigation; if it stays flat, the reward is not a useful training
    signal (only a good move-ranker).

Reads  : cached MDP/episodes/features (00-01) + runs/pemirl/pemirl.pt
         (--released: models/pemirl/pemirl.pt, models/ppo_on_pemirl/)
Writes : runs/ppo_on_pemirl/policy.pt (+ tb logs, history.json,
         goal_reach_summary.json) and runs/ppo_on_pemirl_trajectory.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common import (RELEASED_MCE_CKPT, RELEASED_PEMIRL_CKPT,
                    RELEASED_TRANSFER_DIR, base_parser,
                    load_cfg, load_pipeline, log, seeded_dir, workdir)

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.eval.rollout import _episode_tensors, pemirl_support_sets
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


# --------------------------------------------------------------------------- #
# goal-reach summary + learning-curve figure
# --------------------------------------------------------------------------- #

def _smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    if len(x) < 2:
        return np.asarray(x, dtype=float)
    w = max(min(window, len(x)), 1)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def summarize_history(history: list[dict]) -> dict:
    """First value, peak, and mean of the last 20% ('steady') of goal-reach."""
    reached = np.array([row["reached"] for row in history], dtype=float)
    tail = reached[-max(len(reached) // 5, 1):]
    return {"n_iters": int(len(reached)),
            "first": float(reached[0]),
            "peak": float(reached.max()),
            "peak_iter": int(reached.argmax()),
            "steady": float(tail.mean())}


def _pemirl_generator_rate(fresh="runs/eval/results.json",
                           shipped="results/reference/results.json"):
    """PEMIRL's own adversarial-generator goal-reach rate, for the figure."""
    for path in (fresh, shipped):
        try:
            with open(path) as f:
                return float(json.load(f)["test"]["pemirl_reached_goal"]), path
        except (OSError, KeyError, ValueError):
            continue
    return None, None


def plot_history(history: list[dict], out_png: Path,
                 n_sample: int = 64, reward_label: str = "PEMIRL") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize_history(history)
    it = np.array([row["iter"] for row in history])
    reached = np.array([row["reached"] for row in history], dtype=float)
    ep_len = np.array([row.get("ep_len", np.nan) for row in history],
                      dtype=float)
    entropy = np.array([row.get("entropy", np.nan) for row in history],
                       dtype=float)
    gen_rate, gen_src = _pemirl_generator_rate()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(it, reached, color="0.6", lw=0.8, alpha=0.6,
            label=f"per-iter ({n_sample} tasks)")
    sm = _smooth(reached)
    ax.plot(it[len(it) - len(sm):], sm, color="tab:blue", lw=2,
            label="smoothed")
    ax.axhline(s["first"], color="tab:gray", ls="--", lw=1,
               label=f"init ({s['first']:.3f})")
    if gen_rate is not None:
        ax.axhline(gen_rate, color="tab:red", ls="--", lw=1,
                   label=f"PEMIRL generator ({gen_rate:.3f})")
    ax.plot([s["peak_iter"]], [s["peak"]], "o", color="tab:red", ms=6,
            label=f"peak {s['peak']:.3f}")
    ax.axhline(s["steady"], color="tab:green", ls=":", lw=1,
               label=f"steady {s['steady']:.3f}")
    ax.set_title(f"Goal-reach: fresh PPO on the frozen {reward_label} reward",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("goal-reach rate")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ax.plot(it, ep_len, color="tab:orange", lw=1)
    ax.set_title("Episode length (lower = more direct routes)", fontsize=10)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean steps")

    ax = axes[2]
    ax.plot(it, entropy, color="tab:green", lw=1)
    ax.set_title("Policy entropy (lower = more decisive)", fontsize=10)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("entropy (nats)")

    gen_txt = f", vs PEMIRL's own generator at {gen_rate:.3f}" \
        if gen_rate is not None else ""
    fig.suptitle(f"Validating the learned {reward_label} reward: a fresh agent trained "
                 f"on it learns to navigate ({s['first']:.2f} → "
                 f"{s['peak']:.2f} peak{gen_txt})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    log.info("Learning-curve figure written to %s%s", out_png,
             f" (generator rate from {gen_src})" if gen_src else "")
    return s


def write_summary(history: list[dict], ckpt_dir: Path, source: str) -> dict:
    s = {**summarize_history(history), "source": source}
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "goal_reach_summary.json", "w") as f:
        json.dump(s, f, indent=2)
    log.info("Goal-reach: first %.3f | peak %.3f (iter %d) | "
             "steady (last 20%%) %.3f  [%d iters, %s]",
             s["first"], s["peak"], s["peak_iter"], s["steady"],
             s["n_iters"], source)
    return s


# --------------------------------------------------------------------------- #

def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--reward", choices=["pemirl", "mce"], default="pemirl",
                   help="which frozen recovered reward the fresh PPO trains on: "
                        "PEMIRL's f(s,a,z) (default) or MCE-IRL's linear "
                        "theta.phi(s) — the control that separates 'this "
                        "specific reward transfers' from 'any recovered "
                        "reward transfers'")
    p.add_argument("--pem-ckpt", default=None,
                   help="PEMIRL checkpoint (default: <reward_transfer.pem_ckpt> "
                        "or <pemirl.ckpt_dir>/pemirl.pt)")
    p.add_argument("--mce-ckpt", default=None,
                   help="MCE theta.npz for --reward mce "
                        "(default: released/config ckpt)")
    p.add_argument("--n_epoch", type=int, default=None,
                   help="PPO iterations (default: <reward_transfer.n_epoch> or 200)")
    p.add_argument("--n_sample", type=int, default=None,
                   help="rollouts per PPO iteration "
                        "(default: <reward_transfer.n_sample> or 64)")
    p.add_argument("--released", action="store_true",
                   help="use the released checkpoints under models/ "
                        "(PEMIRL reward; transfer policy for --eval-only)")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training: load the transfer policy + history, "
                        "print the goal-reach summary and regenerate the "
                        "learning-curve figure")
    p.add_argument("--label", default=None,
                   help="reward name for figure titles (default: --reward "
                        "upper-cased; e.g. NOCTX for the context-free AIRL "
                        "ablation checkpoint)")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    pc = cfg["pemirl"]
    # optional per-run overrides (used by smoke/config.yaml to keep 08 toy-scale)
    rt = cfg.get("reward_transfer") or {}
    n_epoch = args.n_epoch if args.n_epoch is not None else int(rt.get("n_epoch", 200))
    n_sample = args.n_sample if args.n_sample is not None else int(rt.get("n_sample", 64))

    seed = int(cfg["seed"])
    run_name = "ppo_on_pemirl" if args.reward == "pemirl" else "ppo_on_mce"
    pem_ckpt = args.pem_ckpt or (
        RELEASED_PEMIRL_CKPT if args.released
        else rt.get("pem_ckpt")
        or str(seeded_dir(Path(pc.get("ckpt_dir", "runs/pemirl/")), seed)
               / "pemirl.pt"))
    out_dir = (Path(rt["ckpt_dir"]) if rt.get("ckpt_dir")
               and args.reward == "pemirl"
               else seeded_dir(workdir(cfg) / run_name, seed))
    # figure named after the actual output dir so arms (pemirl / mce / an
    # ablation checkpoint routed via reward_transfer.ckpt_dir) never clobber
    # each other's figures
    fig_png = workdir(cfg) / (
        f"{out_dir.name}_trajectory_s{seed}.png" if seed
        else f"{out_dir.name}_trajectory.png")
    reward_label = args.label or args.reward.upper()
    policy_dir = (Path(RELEASED_TRANSFER_DIR)
                  if args.released and args.reward == "pemirl" else out_dir)

    mdp, fb, eps = load_pipeline(cfg)

    # ---- eval-only: reload the trained transfer policy + its history --------
    if args.eval_only:
        hist_path = policy_dir / "history.json"
        pol_path = policy_dir / "policy.pt"
        with open(hist_path) as f:
            history = json.load(f)
        policy = MaskedPolicy(fb.dim, mdp.n_actions,
                              context_dim=(int(pc["context_dim"])
                                           if args.reward == "pemirl" else 0),
                              hidden=tuple(pc["policy_hidden"]))
        policy.load_state_dict(torch.load(pol_path, map_location=device))
        policy.to(device).eval()
        log.info("Loaded transfer policy %s + history %s (%d iters)",
                 pol_path, hist_path, len(history))
        write_summary(history, out_dir, source=str(policy_dir))
        plot_history(history, fig_png, n_sample=n_sample,
                     reward_label=reward_label)
        return

    # ---- load + FREEZE the recovered reward ---------------------------------
    if args.reward == "pemirl":
        pem = PEMIRL(obs_dim=fb.dim, n_actions=mdp.n_actions, cfg=pc,
                     device=device)
        pem.load(pem_ckpt)
        pem.eval()
        for q in pem.parameters():
            q.requires_grad_(False)
        log.info("Loaded frozen PEMIRL reward from %s (context_dim=%d)",
                 pem_ckpt, pem.dim_cnt)

        # ---- per-vessel support sets -> inferred z (train vessels) ---------
        support = pemirl_support_sets(eps["train"],
                                      int(pc["support_set_size"]),
                                      seed=int(cfg["seed"]))
        z_by_vessel: dict[int, torch.Tensor] = {}
        for mmsi, (sup, _qry) in support.items():
            z_by_vessel[mmsi] = pem.infer_context_support(
                [_episode_tensors(e, fb) for e in sup], fixed=True).detach()
        log.info("Inferred z for %d train vessels (support_size=%d)",
                 len(z_by_vessel), int(pc["support_set_size"]))

        # only train on tasks whose vessel has an inferred z
        train_eps = [e for e in eps["train"] if e.mmsi in z_by_vessel]
        tasks = tasks_from_episodes(train_eps)
        task_z = torch.stack([z_by_vessel[t.mmsi].squeeze(0)
                              for t in tasks])                     # (N, Z)
        policy_ctx = pem.dim_cnt
        log.info("PPO-on-reward: %d tasks across %d vessels",
                 len(tasks), len(z_by_vessel))

        # reward override: frozen discriminator f(s,a,z) for this rollout's z
        def reward_fn(ro):
            t = ro.tensors(device)
            z = ro.z.to(device).unsqueeze(0).expand(len(ro), -1)
            return pem.airl_reward(t["obs"], t["actions"], z)
    else:
        # MCE-IRL control: same PPO harness, frozen linear reward
        # r(s) = theta . phi(s). State-only and pooled — no context, no
        # support sets. phi(s) is exactly the policy observation, so the
        # reward is a dot product with the (frozen) theta, used RAW: phi is
        # already standardized/clipped, and PPO's per-batch advantage
        # normalization handles scale (as in 05's hand-crafted baseline).
        # Per-rollout standardization is NOT applicable here — it NaNs on
        # length-1 rollouts once the policy gets good, and centering a
        # state-only reward per rollout would erase its absolute level.
        mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
        mce.load(Path(args.mce_ckpt or
                      (RELEASED_MCE_CKPT if args.released
                       else cfg["mce_irl"]["ckpt"])))
        theta = torch.as_tensor(mce.theta, dtype=torch.float32, device=device)
        log.info("Loaded frozen MCE-IRL reward (theta dim %d)", theta.numel())

        train_eps = list(eps["train"])
        tasks = tasks_from_episodes(train_eps)
        task_z = None
        policy_ctx = 0
        log.info("PPO-on-reward (MCE control): %d tasks", len(tasks))

        def reward_fn(ro):
            t = ro.tensors(device)
            return t["obs"] @ theta

    # ---- fresh policy + PPO --------------------------------------------------
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    sampler = Sampler(env, device=device)
    policy = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=policy_ctx,
                          hidden=tuple(pc["policy_hidden"]))
    ppo = PPO(policy, lr=float(pc["optimizer_lr_policy"]),
              clip=float(pc["ppo_clip"]), epochs=int(pc["ppo_epochs"]),
              gae_lambda=float(pc["gae_lambda"]), gamma=float(cfg["mdp"]["gamma"]),
              entropy_coef=float(pc["entropy_coef"]),
              value_coef=float(pc["value_coef"]),
              mini_batch_size=int(pc["mini_batch_size"]), device=device)

    ckpt_dir = out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    rng = np.random.default_rng(int(cfg["seed"]))
    history, best_reached = [], -1.0
    for it in range(n_epoch):
        idx = rng.choice(len(tasks), n_sample,
                         replace=len(tasks) < n_sample)
        batch_tasks = [tasks[i] for i in idx]
        z_batch = task_z[idx].to(device) if task_z is not None else None
        rollouts = sampler.collect(policy, batch_tasks, z_batch)
        stats = ppo.step(rollouts, reward_override=reward_fn,
                         lr_mult=max(1.0 - it / n_epoch, 0.1))

        reached = float(np.mean([ro.reached for ro in rollouts]))
        steplen = float(np.mean([len(ro) for ro in rollouts]))
        row = {"iter": it, "reached": reached, "ep_len": steplen, **stats}
        for k, v in row.items():
            if k != "iter":
                tb.scalar(f"{run_name}/{k}", float(v), it)
        history.append(row)
        if it % 5 == 0:
            log.info("iter %4d | reached %.3f | ep_len %5.1f | pi %.4f | ent %.3f",
                     it, reached, steplen, stats["pi_loss"], stats["entropy"])
        if reached >= best_reached:
            best_reached = reached
            torch.save(policy.state_dict(), ckpt_dir / "policy.pt")

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    tb.close()
    write_summary(history, ckpt_dir, source=str(ckpt_dir))
    plot_history(history, fig_png, n_sample=n_sample,
                     reward_label=reward_label)
    log.info("Done. Best goal-reach rate %.3f -> %s",
             best_reached, ckpt_dir / "policy.pt")
    log.info("INTERPRETATION: if reached climbed from ~0 toward a high value, "
             "the learned %s reward TEACHES navigation. If it stayed flat, "
             "the reward ranks expert moves (good LL) but is not a useful "
             "training signal on its own.", args.reward.upper())


if __name__ == "__main__":
    main()
