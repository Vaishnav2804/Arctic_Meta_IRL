"""15 — Context sample-complexity: support-set size, selection strategy, and
latent-dimension proxies for the PEMIRL posterior.

Answers the standing reviewer objection to the paper's negative context
result — *"you under-fed the posterior"* — without retraining anything.

The orphaned ``runs/eval/support_sweep.json`` varied k by taking a longer
support prefix and letting the *query* set shrink (66,568 -> 52,392
decisions), so its k arms are not comparable. Here the query set is
**frozen** per vessel and only the support grows, which makes every arm score
the identical decisions and makes per-vessel differences paired.

Arms
----
1. **k-curve** (``--strategy random`` and ``temporal_earliest``):
   k in {0, 1, 2, 3, 5, 10, all}. k=0 is the no-adaptation arm — an empty
   support set makes ``PEMIRL.infer_context_support`` fall back to the prior
   mean z=0. If LL(k=0) == LL(k=all), context is inert by construction.
2. **selection strategy** at fixed k: random / temporal_earliest / longest /
   most_diverse, plus two controls:
   * ``mismatched`` — z inferred from a *different* vessel's support set. If
     LL does not move, z carries no vessel-specific signal.
   * ``oracle_leak`` — z inferred from the query episode being scored. Leaks,
     so it is a *ceiling*: if even this barely moves LL, no support-set
     design can help.
3. **z-convergence**: ||z_k - z_all|| per vessel, separating "posterior
   under-fed" from "reward ignores z".
4. **latent-dimension proxies** (no retraining): PCA effective rank of the
   inferred z across vessels, and per-dimension zeroing sensitivity.

    python scripts/15_support_sweep.py --config configs/pemirl.yaml --released
    python scripts/15_support_sweep.py --config configs/pemirl_noctx.yaml \
        --pemirl-ckpt runs/pemirl_noctx/pemirl.pt --tag noctx
    python scripts/15_support_sweep.py --config configs/pemirl.yaml \
        --released --legacy-split --splits test   # reproduces results.json

Writes <workdir>/eval/support_sweep_v2[_<tag>].json,
support_sweep_per_vessel[_<tag>].csv and support_sweep_curve[_<tag>].png.
"""
from __future__ import annotations

import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (RELEASED_MCE_CKPT, RELEASED_PEMIRL_CKPT, base_parser,
                    load_cfg, load_pipeline, log, workdir)
from eval_robustness import mce_per_episode_ll

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.eval.heterogeneity import (DESCRIPTOR_COLS,
                                                behavior_descriptors)
from arctic_meta_irl.eval.rollout import _episode_tensors, pemirl_support_sets
from arctic_meta_irl.utils.seeding import resolve_device, set_seed

# paired bootstrap + Wilcoxon, shared with the RQ2 significance table
paired_stats = importlib.import_module("09_significance").paired_stats

K_GRID = [0, 1, 2, 3, 5, 10, -1]        # -1 == "all available support"
STRATEGIES = ["random", "temporal_earliest", "longest", "most_diverse",
              "mismatched", "oracle_leak"]


def _k_label(k: int) -> str:
    return "all" if k < 0 else str(k)


# --------------------------------------------------------------------------- #
# Matched support/query construction
# --------------------------------------------------------------------------- #

def frozen_query_split(episodes, min_pool: int = 1):
    """{mmsi: (pool, query)} with the query set FROZEN across all k arms.

    Episodes are ordered by (year, month, voyage_index); the last
    Q = max(1, n // 3) are held out as the query set and never enter any
    support set. Support sets are drawn from the earlier `pool` only, so
    growing k never changes what is being scored — and it also keeps the
    support strictly in the query's past, which the paper's random scheme
    does not.
    """
    by_v = defaultdict(list)
    for ep in episodes:
        by_v[ep.mmsi].append(ep)
    out = {}
    for mmsi, eps in by_v.items():
        eps = sorted(eps, key=lambda e: (e.year or 0, e.month or 0,
                                         e.voyage_index))
        q = max(1, len(eps) // 3)
        pool, query = eps[:len(eps) - q], eps[len(eps) - q:]
        if len(pool) < min_pool or not query:
            continue
        out[mmsi] = (pool, query)
    return out


def legacy_split(episodes, k: int, seed: int):
    """The paper's scheme verbatim, for the reproduction assertion."""
    return pemirl_support_sets(episodes, k, seed)


def _descriptor_matrix(pool, mdp):
    df = behavior_descriptors(pool, mdp)
    X = df[DESCRIPTOR_COLS].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)


def select_support(pool, k: int, strategy: str, rng, mdp):
    """Choose k support episodes from a vessel's pool under `strategy`."""
    if k == 0:
        return []
    kk = len(pool) if k < 0 else min(k, len(pool))
    if strategy == "random":
        idx = rng.permutation(len(pool))[:kk]
        return [pool[i] for i in idx]
    if strategy == "temporal_earliest":
        return pool[:kk]            # pool is already chronologically sorted
    if strategy == "longest":
        return sorted(pool, key=len, reverse=True)[:kk]
    if strategy == "most_diverse":
        if kk >= len(pool):
            return list(pool)
        X = _descriptor_matrix(pool, mdp)
        chosen = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(chosen) < kk:      # greedy max-min in descriptor space
            d = np.min(np.linalg.norm(X[:, None, :] - X[None, chosen, :],
                                      axis=-1), axis=1)
            d[chosen] = -np.inf
            chosen.append(int(np.argmax(d)))
        return [pool[i] for i in chosen]
    raise ValueError(f"unknown strategy {strategy!r}")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

class Scorer:
    """Caches per-episode feature tensors; scores query episodes under any z."""

    def __init__(self, pem: PEMIRL, fb, mdp):
        self.pem, self.fb, self.mdp = pem, fb, mdp
        self._cache: dict[int, tuple] = {}

    def tensors(self, ep):
        key = ep.voyage_index
        if key not in self._cache:
            phi, act = _episode_tensors(ep, self.fb)
            mask = torch.from_numpy(self.mdp.action_mask[ep.states[:-1]])
            self._cache[key] = (phi.to(self.pem.device),
                                act.to(self.pem.device),
                                mask.to(self.pem.device))
        return self._cache[key]

    @torch.no_grad()
    def infer_z(self, support):
        return self.pem.infer_context_support(
            [self.tensors(ep)[:2] for ep in support], fixed=True)

    @torch.no_grad()
    def episode_ll(self, ep, z) -> float:
        phi, act, mask = self.tensors(ep)
        lp = self.pem.policy.log_prob_action(phi, mask, act,
                                             z.expand(phi.size(0), -1))
        return float(lp.sum().item())


def aggregate(per_ep: dict[int, float], query_by_vessel,
              mce_ll: dict[int, float]) -> dict:
    """micro (per-decision) + macro (mean per-vessel) LL, + per-vessel rows."""
    rows, tot, dec = [], 0.0, 0
    mce_tot = 0.0
    for mmsi, query in query_by_vessel.items():
        s = sum(per_ep[ep.voyage_index] for ep in query)
        m = sum(mce_ll[ep.voyage_index] for ep in query)
        d = sum(len(ep) for ep in query)
        tot += s; mce_tot += m; dec += d
        rows.append({"mmsi": int(mmsi), "n_decisions": int(d),
                     "ll_per_decision": s / d, "mce_ll_per_decision": m / d})
    return {
        "n_vessels": len(rows), "n_decisions": int(dec),
        "ll_per_decision": tot / max(dec, 1),
        "macro_ll_per_decision": float(np.mean([r["ll_per_decision"]
                                                for r in rows])),
        "mce_ll_per_decision": mce_tot / max(dec, 1),
        "mce_macro_ll_per_decision": float(np.mean(
            [r["mce_ll_per_decision"] for r in rows])),
        "wins_vs_mce": int(sum(r["ll_per_decision"] > r["mce_ll_per_decision"]
                               for r in rows)),
        "_per_vessel": rows,
    }


def run_arm(scorer: Scorer, split_map, k: int, strategy: str, mdp,
            seed: int) -> dict[int, float]:
    """One (k, strategy) arm -> ({voyage_index: LL}, {mmsi: z}, n_clamped)."""
    rng = np.random.default_rng(seed)
    mmsis = sorted(split_map)
    # mismatched: derange vessels so each gets a *different* vessel's context
    donor = {}
    if strategy == "mismatched":
        shifted = mmsis[1:] + mmsis[:1] if len(mmsis) > 1 else mmsis
        donor = dict(zip(mmsis, shifted))

    per_ep, zs, clamped = {}, {}, 0
    for mmsi in mmsis:
        pool, query = split_map[mmsi]
        if strategy == "oracle_leak":
            for ep in query:                      # z from the scored episode
                z = scorer.infer_z([ep])
                per_ep[ep.voyage_index] = scorer.episode_ll(ep, z)
                zs[mmsi] = z.squeeze(0).cpu().numpy()
            continue
        src_pool = split_map[donor[mmsi]][0] if strategy == "mismatched" \
            else pool
        sub = "random" if strategy == "mismatched" else strategy
        support = select_support(src_pool, k, sub, rng, mdp)
        if k > 0 and len(support) < k:
            clamped += 1
        z = scorer.infer_z(support)
        zs[mmsi] = z.squeeze(0).cpu().numpy()
        for ep in query:
            per_ep[ep.voyage_index] = scorer.episode_ll(ep, z)
    if clamped:
        log.info("  [k=%s/%s] %d/%d vessels clamped below the requested k",
                 _k_label(k), strategy, clamped, len(mmsis))
    return per_ep, zs, clamped


# --------------------------------------------------------------------------- #
# Latent-dimension proxies (no retraining)
# --------------------------------------------------------------------------- #

def latent_diagnostics(scorer: Scorer, split_map, zs_all: dict, mce_ll,
                       query_by_vessel) -> dict:
    """PCA effective rank of z + per-dimension zeroing sensitivity."""
    Z = np.stack([zs_all[m] for m in sorted(zs_all)])       # (V, dim)
    dim = Z.shape[1]
    out: dict = {"context_dim": int(dim),
                 "per_dim_std": [float(s) for s in Z.std(axis=0)]}
    if dim == 0:
        return out
    Zc = Z - Z.mean(axis=0)
    sv = np.linalg.svd(Zc, compute_uv=False)
    var = sv ** 2
    p = var / max(var.sum(), 1e-12)
    out["pca_explained_ratio"] = [float(x) for x in p]
    out["pca_cumulative"] = [float(x) for x in np.cumsum(p)]
    # participation-ratio effective rank + entropy-based effective rank
    out["effective_rank_participation"] = float(
        (var.sum() ** 2) / max((var ** 2).sum(), 1e-12))
    nz = p[p > 1e-12]
    out["effective_rank_entropy"] = float(np.exp(-(nz * np.log(nz)).sum()))

    # per-dimension zeroing: how much LL depends on each latent coordinate
    base = {}
    for mmsi, query in query_by_vessel.items():
        z = torch.as_tensor(zs_all[mmsi], dtype=torch.float32,
                            device=scorer.pem.device).unsqueeze(0)
        for ep in query:
            base[ep.voyage_index] = scorer.episode_ll(ep, z)
    base_agg = aggregate(base, query_by_vessel, mce_ll)
    out["baseline_ll_per_decision"] = base_agg["ll_per_decision"]
    abl = {}
    for d in range(dim):
        per_ep = {}
        for mmsi, query in query_by_vessel.items():
            z = torch.as_tensor(zs_all[mmsi], dtype=torch.float32,
                                device=scorer.pem.device).unsqueeze(0).clone()
            z[0, d] = 0.0
            for ep in query:
                per_ep[ep.voyage_index] = scorer.episode_ll(ep, z)
        a = aggregate(per_ep, query_by_vessel, mce_ll)
        abl[str(d)] = {"ll_per_decision": a["ll_per_decision"],
                       "delta_nats": a["ll_per_decision"]
                       - base_agg["ll_per_decision"]}
    out["dim_ablation"] = abl
    # all-dims-zero == the k=0 prior-mean arm; reported for cross-checking
    zero = {}
    zz = torch.zeros(1, dim, device=scorer.pem.device)
    for mmsi, query in query_by_vessel.items():
        for ep in query:
            zero[ep.voyage_index] = scorer.episode_ll(ep, zz)
    out["all_zero_ll_per_decision"] = aggregate(
        zero, query_by_vessel, mce_ll)["ll_per_decision"]
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def plot_curve(results: dict, dest: Path, ref: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in results if results[s].get("k_curve")]
    if not splits:
        return
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.9),
                             squeeze=False)
    for ax, split in zip(axes[0], splits):
        blk = results[split]
        for name, style in (("k_curve", "-o"), ("k_curve_temporal", "--s")):
            cur = blk.get(name)
            if not cur:
                continue
            ks = [c for c in cur]
            x = range(len(ks))
            ax.plot(x, [cur[c]["ll_per_decision"] for c in ks], style,
                    label=("PEMIRL (random support)" if name == "k_curve"
                           else "PEMIRL (earliest support)"))
            ax.set_xticks(list(x)); ax.set_xticklabels(ks)
        mce = blk["k_curve"][next(iter(blk["k_curve"]))]["mce_ll_per_decision"]
        ax.axhline(mce, color="gray", ls=":", label="MCE-IRL (linear, pooled)")
        if ref and split in ref and ref[split].get("k_curve"):
            r = ref[split]["k_curve"]
            ax.axhline(r[next(iter(r))]["ll_per_decision"], color="tab:red",
                       ls="-.", label="AIRL (nonlinear, pooled)")
        ax.axhline(-np.log(6), color="black", lw=0.7, alpha=0.4,
                   label="uniform log(1/6)")
        ax.set_xlabel("support-set size k (frozen query set)")
        ax.set_ylabel("test LL / decision")
        ax.set_title(split)
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    log.info("Figure written to %s", dest)


# --------------------------------------------------------------------------- #

def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--released", action="store_true")
    p.add_argument("--splits", nargs="+", default=["test", "temporal_shift"])
    p.add_argument("--mce-ckpt", default=None)
    p.add_argument("--pemirl-ckpt", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--strategy-k", type=int, default=3,
                   help="k at which the selection-strategy sweep is run")
    p.add_argument("--legacy-split", action="store_true",
                   help="use the paper's shrinking-query scheme (reproduction "
                        "check against runs/eval/results.json only)")
    p.add_argument("--no-latent", action="store_true",
                   help="skip the latent-dimension proxies")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    seed = int(cfg["seed"])

    mdp, fb, eps = load_pipeline(cfg)
    mce = MCEIRL(mdp, fb, seed=seed)
    mce.load(Path(args.mce_ckpt or (RELEASED_MCE_CKPT if args.released
                                    else cfg["mce_irl"]["ckpt"])))
    pem = PEMIRL(fb.dim, mdp.n_actions, cfg["pemirl"], device=device)
    pem.load(Path(args.pemirl_ckpt or
                  (RELEASED_PEMIRL_CKPT if args.released else
                   Path(cfg["pemirl"].get("ckpt_dir", "runs/pemirl/"))
                   / "pemirl.pt")))
    pem.eval()
    scorer = Scorer(pem, fb, mdp)

    results: dict = {}
    per_vessel_rows: list[dict] = []

    for split in args.splits:
        episodes = eps.get(split, [])
        if not episodes:
            log.warning("split %s empty; skipping", split)
            continue
        log.info("[%s] MCE per-episode LLs (support-independent)...", split)
        mce_ll = mce_per_episode_ll(mce, episodes)
        blk: dict = {}

        if args.legacy_split:
            # ---- reproduction mode: the paper's shrinking-query scheme -----
            cur = {}
            for k in [1, 3, 5, 10]:
                smap = legacy_split(episodes, k, seed)
                qbv = {m: q for m, (_, q) in smap.items()}
                per_ep, _, _ = run_arm(scorer, smap, k, "random", mdp, seed)
                a = aggregate(per_ep, qbv, mce_ll)
                a.pop("_per_vessel")
                cur[_k_label(k)] = a
                log.info("[%s/legacy k=%d] LL/decision %.6f over %d decisions",
                         split, k, a["ll_per_decision"], a["n_decisions"])
            blk["legacy_k_curve"] = cur
            results[split] = blk
            continue

        split_map = frozen_query_split(episodes)
        query_by_vessel = {m: q for m, (_, q) in split_map.items()}
        n_dec = sum(len(e) for q in query_by_vessel.values() for e in q)
        log.info("[%s] frozen query set: %d vessels, %d decisions "
                 "(pool sizes %d-%d)", split, len(split_map), n_dec,
                 min(len(p_) for p_, _ in split_map.values()),
                 max(len(p_) for p_, _ in split_map.values()))
        blk["frozen_query"] = {"n_vessels": len(split_map),
                               "n_decisions": int(n_dec)}

        # ---- 1. k-curve, two support-ordering schemes ---------------------
        zs_by_k = {}
        for name, strat in (("k_curve", "random"),
                            ("k_curve_temporal", "temporal_earliest")):
            cur = {}
            for k in K_GRID:
                per_ep, zs, clamped = run_arm(scorer, split_map, k, strat,
                                              mdp, seed)
                a = aggregate(per_ep, query_by_vessel, mce_ll)
                for r in a.pop("_per_vessel"):
                    per_vessel_rows.append({"split": split, "arm": name,
                                            "k": _k_label(k), **r})
                a["vessels_clamped"] = clamped
                cur[_k_label(k)] = a
                if strat == "random":
                    zs_by_k[_k_label(k)] = zs
                log.info("[%s/%s k=%-3s] LL/dec %.6f | macro %.6f | %d dec",
                         split, strat, _k_label(k), a["ll_per_decision"],
                         a["macro_ll_per_decision"], a["n_decisions"])
            blk[name] = cur

        # ---- 2. selection strategy at fixed k -----------------------------
        kk = int(args.strategy_k)
        strat_block, strat_pv = {}, {}
        for strategy in STRATEGIES:
            per_ep, _, _ = run_arm(scorer, split_map, kk, strategy, mdp, seed)
            a = aggregate(per_ep, query_by_vessel, mce_ll)
            pv = a.pop("_per_vessel")
            strat_pv[strategy] = {r["mmsi"]: r["ll_per_decision"] for r in pv}
            for r in pv:
                per_vessel_rows.append({"split": split, "arm": "strategy",
                                        "k": str(kk), "strategy": strategy,
                                        **r})
            strat_block[strategy] = a
            log.info("[%s/strategy=%-17s k=%d] LL/dec %.6f | macro %.6f",
                     split, strategy, kk, a["ll_per_decision"],
                     a["macro_ll_per_decision"])
        blk["strategy"] = strat_block

        # ---- paired per-vessel contrasts ----------------------------------
        rng = np.random.default_rng(0)
        base = strat_pv["random"]
        blk["paired_vs_random"] = {
            s: paired_stats(np.array([base[m] for m in sorted(base)]),
                            np.array([strat_pv[s][m] for m in sorted(base)]),
                            rng)
            for s in STRATEGIES if s != "random"}
        # k arms vs the no-adaptation k=0 arm (random ordering)
        pv_by_k = defaultdict(dict)
        for r in per_vessel_rows:
            if r["split"] == split and r["arm"] == "k_curve":
                pv_by_k[r["k"]][r["mmsi"]] = r["ll_per_decision"]
        z0 = pv_by_k["0"]
        blk["paired_vs_k0"] = {
            kl: paired_stats(np.array([z0[m] for m in sorted(z0)]),
                             np.array([pv_by_k[kl][m] for m in sorted(z0)]),
                             rng)
            for kl in pv_by_k if kl != "0"}

        # ---- 3. z-convergence --------------------------------------------
        z_all = zs_by_k["all"]
        conv = {}
        for kl, zs in zs_by_k.items():
            d = [float(np.linalg.norm(zs[m] - z_all[m])) for m in sorted(z_all)]
            conv[kl] = {"mean_dist_to_z_all": float(np.mean(d)) if d else 0.0,
                        "max_dist_to_z_all": float(np.max(d)) if d else 0.0,
                        "mean_norm": float(np.mean(
                            [np.linalg.norm(zs[m]) for m in sorted(zs)]))}
        blk["z_convergence"] = conv

        # ---- 4. latent-dimension proxies ----------------------------------
        if not args.no_latent:
            blk["latent"] = latent_diagnostics(scorer, split_map, z_all,
                                               mce_ll, query_by_vessel)
        results[split] = blk

    suffix = f"_{args.tag}" if args.tag else ""
    if args.legacy_split:
        suffix += "_legacy"
    out_dir = workdir(cfg) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"support_sweep_v2{suffix}.json"
    dest.write_text(json.dumps(results, indent=2, default=float))
    if per_vessel_rows:
        pd.DataFrame(per_vessel_rows).to_csv(
            out_dir / f"support_sweep_per_vessel{suffix}.csv", index=False)
    log.info("Results written to %s", dest)

    if not args.legacy_split:
        ref_path = out_dir / "support_sweep_v2_noctx.json"
        ref = (json.loads(ref_path.read_text())
               if ref_path.exists() and not args.tag else None)
        plot_curve(results, out_dir / f"support_sweep_curve{suffix}.png", ref)


if __name__ == "__main__":
    main()
