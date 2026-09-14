"""06 — Evaluate MCE-IRL vs PEMIRL vs PPO baseline (paper Sec. II/III metrics).

    python scripts/06_evaluate.py --config configs/pemirl.yaml \
        --splits test temporal_shift --max-routes 200

    # reproduce the paper numbers from the released checkpoints in models/
    python scripts/06_evaluate.py --config configs/pemirl.yaml --released

Metrics
  trajectory fit  : per-decision test log-likelihood (MCE-IRL pooled policy;
                    PEMIRL conditioned on a per-vessel support set) and
                    feature expectation error (FEE),
  route fidelity  : Hausdorff distance (km), cell overlap (Jaccard),
                    route length ratio for origin--destination paths,
  generalization  : unseen vessels (vessel-disjoint `test`) and temporal
                    shift (`test_temporal`).

Also reports PEMIRL's relative likelihood improvement over MCE-IRL.

Writes : runs/eval/results.json (+ per-route CSVs)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (RELEASED_MCE_CKPT, RELEASED_PEMIRL_CKPT, base_parser,
                    load_cfg, load_pipeline, log, workdir)

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.algos.ppo import PPO  # noqa: F401 (kept for parity)
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, Task
from arctic_meta_irl.eval.metrics import (aggregate, feature_expectation_error,
                                          route_metrics)
from arctic_meta_irl.eval.rollout import (generate_route_pemirl,
                                          generate_route_pemirl_softvi,
                                          generate_route_policy,
                                          pemirl_log_likelihood,
                                          pemirl_support_sets)
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def _expert_fe(fb, episodes) -> np.ndarray:
    phis = [fb.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month,
                              ep.mmsi).sum(axis=0) for ep in episodes]
    return np.mean(phis, axis=0)


def _route_fe(fb, path, ep) -> np.ndarray:
    return fb.episode_matrix(np.asarray(path), ep.goal, ep.year, ep.month,
                             ep.mmsi).sum(axis=0)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--splits", nargs="+", default=["test", "temporal_shift"])
    p.add_argument("--max-routes", type=int, default=200,
                   help="cap on O-D route generations per split/method")
    p.add_argument("--decoder", choices=["native", "soft_vi"], default="native",
                   help="PEMIRL/AIRL route decoder: 'native' policy rollout, or "
                        "'soft_vi' = planner-matched soft-VI decode of the frozen "
                        "reward f (equalizes the planner with MCE-IRL; §6.3.4)")
    p.add_argument("--mce-ckpt", default=None)
    p.add_argument("--pemirl-ckpt", default=None)
    p.add_argument("--ppo-ckpt", default=None)
    p.add_argument("--released", action="store_true",
                   help="evaluate the released checkpoints under models/ "
                        "(MCE-IRL + PEMIRL) instead of runs/")
    p.add_argument("--out", default=None,
                   help="output directory (default: <workdir>/eval)")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])

    # --released overrides checkpoint *inputs* only; caches/data via config.
    if args.released:
        args.mce_ckpt = args.mce_ckpt or RELEASED_MCE_CKPT
        args.pemirl_ckpt = args.pemirl_ckpt or RELEASED_PEMIRL_CKPT
        log.info("--released: mce=%s, pemirl=%s",
                 args.mce_ckpt, args.pemirl_ckpt)

    mdp, fb, eps = load_pipeline(cfg)
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    out = Path(args.out) if args.out else workdir(cfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sfx = "_softvi" if args.decoder == "soft_vi" else ""

    # ---- load models (skip any whose checkpoint is missing) -----------------
    models = {}
    mce_ckpt = Path(args.mce_ckpt or cfg.get("mce_irl", {}).get(
        "ckpt", "runs/mce_irl/theta.npz"))
    if mce_ckpt.exists():
        mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
        mce.load(mce_ckpt)
        models["mce_irl"] = mce
    pem_ckpt = Path(args.pemirl_ckpt or
                    Path(cfg.get("pemirl", {}).get("ckpt_dir",
                                                   "runs/pemirl/")) / "pemirl.pt")
    if pem_ckpt.exists():
        pem_cfg = cfg.get("pemirl", {"context_dim": 8})
        pem = PEMIRL(fb.dim, mdp.n_actions, pem_cfg, device=device)
        pem.load(pem_ckpt)
        pem.eval()
        models["pemirl"] = pem
    ppo_ckpt = Path(args.ppo_ckpt or
                    Path(cfg.get("ppo_baseline", {}).get(
                        "ckpt_dir", "runs/ppo_baseline/")) / "policy.pt")
    if ppo_ckpt.exists():
        pb = cfg.get("ppo_baseline", {})
        pol = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=0,
                           hidden=tuple(pb.get("hidden", (256, 256))))
        pol.load_state_dict(torch.load(ppo_ckpt, map_location=device))
        pol.to(device).eval()
        models["ppo_baseline"] = pol
    if not models:
        raise FileNotFoundError("No checkpoints found; run scripts/03-05 first.")
    log.info("Evaluating: %s on splits %s", sorted(models), args.splits)

    results: dict = defaultdict(dict)
    support_size = int(cfg.get("pemirl", {}).get("support_set_size", 3))

    for split in args.splits:
        episodes = eps.get(split, [])
        if not episodes:
            log.warning("split %s empty; skipping", split)
            continue
        f_expert = _expert_fe(fb, episodes)

        # ================= trajectory fit: log-likelihood ====================
        if "mce_irl" in models:
            ll, n = models["mce_irl"].log_likelihood(episodes)
            results[split]["mce_irl_ll_per_decision"] = ll / max(n, 1)
        if "pemirl" in models:
            ll, n = pemirl_log_likelihood(models["pemirl"], episodes, fb, mdp,
                                          support_size=support_size,
                                          seed=int(cfg["seed"]))
            results[split]["pemirl_ll_per_decision"] = ll / max(n, 1)
        a = results[split].get("mce_irl_ll_per_decision")
        b = results[split].get("pemirl_ll_per_decision")
        if a is not None and b is not None:
            # likelihood ratio per decision: exp(LL_pem - LL_mce) - 1
            results[split]["pemirl_vs_mce_likelihood_improvement_pct"] = \
                100.0 * (float(np.exp(b - a)) - 1.0)

        # ================= route fidelity + FEE ==============================
        support_map = pemirl_support_sets(episodes, support_size,
                                          seed=int(cfg["seed"]))
        sub = episodes[:args.max_routes]
        for name, model in models.items():
            per_route, f_gen = [], []
            for ep in sub:
                task = Task(start=int(ep.states[0]), goal=int(ep.goal),
                            year=ep.year, month=ep.month, mmsi=ep.mmsi,
                            category=ep.category)
                if name == "mce_irl":
                    path = model.greedy_route(task.start, task.goal, ep.year,
                                              ep.month, ep.mmsi,
                                              horizon=env.max_horizon)
                elif name == "pemirl":
                    sup = support_map.get(ep.mmsi, ([ep], []))[0]
                    if args.decoder == "soft_vi":
                        path = generate_route_pemirl_softvi(
                            model, mdp, task, sup, fb,
                            horizon=env.max_horizon)
                    else:
                        path = generate_route_pemirl(model, env, task, sup, fb)
                else:
                    path = generate_route_policy(model, env, task,
                                                 deterministic=True,
                                                 device=device)
                m = route_metrics(mdp, path, list(ep.states))
                per_route.append(m)
                f_gen.append(_route_fe(fb, path, ep))
            agg = aggregate(per_route)
            agg["fee"] = feature_expectation_error(
                f_expert, np.mean(f_gen, axis=0))
            for k, v in agg.items():
                results[split][f"{name}_{k}"] = v
            pd.DataFrame(per_route).to_csv(
                out / f"routes_{split}_{name}{sfx}.csv", index=False)

    # ---- report ----------------------------------------------------------------
    with open(out / f"results{sfx}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    for split, d in results.items():
        log.info("== %s ==", split)
        for k in sorted(d):
            log.info("  %-48s %10.4f", k, d[k])
    log.info("Results written to %s/results%s.json", out, sfx)


if __name__ == "__main__":
    main()
