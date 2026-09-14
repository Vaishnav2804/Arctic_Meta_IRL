"""Paired per-vessel significance tests + bootstrap CIs across the three
reward classes, to back the mechanism-isolation claim (capacity vs context).

Reward classes (the 2x2, minus the empty linear+context cell):
    MCE   -- linear, pooled            (mce_ll in every per-episode file)
    NC    -- nonlinear, pooled         (PEMIRL_NOCTX; per_episode_ll_noctx*.json)
    CTX   -- nonlinear, per-vessel z   (full PEMIRL;  per_episode_ll{,_s1,_s2}.json)

Contrasts:
    capacity  =  NC  - MCE   (linear -> nonlinear, holding pooling fixed)
    context   =  CTX - NC    (pooled -> per-vessel, holding nonlinearity fixed)
    full      =  CTX - MCE   (the paper's original RQ2 gap)

For each (split, support-variant, contrast, seed) we aggregate to per-vessel
per-decision LL exactly as eval_robustness.aggregate() does (sum LL over a
vessel's query episodes / sum decisions), then report on the per-vessel
paired differences:
    * macro-LL improvement %  = 100 * (exp(mean_v[model] - mean_v[mce]) - 1),
      with a paired vessel-level bootstrap 95% CI,
    * Wilcoxon signed-rank p-value on the per-vessel LL differences,
    * per-vessel win count.

NC's s1/s2 evals may not exist yet (training in flight); missing seeds are
skipped and noted. Deterministic: fixed bootstrap RNG seed.

    python scripts/09_significance.py
    python scripts/09_significance.py --eval_dir runs/eval --out runs/eval/significance.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# per-episode file tags -> seed, per model. CTX seed0 is the untagged file.
CTX_FILES = {0: "per_episode_ll.json", 1: "per_episode_ll_s1.json",
             2: "per_episode_ll_s2.json"}
NC_FILES = {0: "per_episode_ll_noctx.json", 1: "per_episode_ll_noctx_s1.json",
            2: "per_episode_ll_noctx_s2.json"}
VARIANTS = {"random_support": "pemirl_ll_random_support",
            "temporal_support": "pemirl_ll_temporal_support"}
N_BOOT = 10000


def per_vessel_pd(entries: dict, model_field: str):
    """Per-vessel per-decision LL for MCE and the model, over query episodes
    (those carrying `model_field`). Returns aligned dict {mmsi: (mce, model)}."""
    num = defaultdict(float)   # sum model LL
    mce = defaultdict(float)   # sum mce LL
    dec = defaultdict(float)   # sum decisions
    for e in entries.values():
        if model_field not in e:
            continue
        v = int(e["mmsi"])
        num[v] += e[model_field]
        mce[v] += e["mce_ll"]
        dec[v] += e["n_decisions"]
    return {v: (mce[v] / dec[v], num[v] / dec[v]) for v in dec}


def paired_stats(mce_pd: np.ndarray, model_pd: np.ndarray, rng) -> dict:
    """Macro improvement % + paired vessel bootstrap CI + Wilcoxon."""
    diff = model_pd - mce_pd
    macro_gap = float(model_pd.mean() - mce_pd.mean())
    improve = 100.0 * (np.exp(macro_gap) - 1.0)
    n = len(diff)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_gap = model_pd[idx].mean(1) - mce_pd[idx].mean(1)
    boot_improve = 100.0 * (np.exp(boot_gap) - 1.0)
    lo, hi = np.percentile(boot_improve, [2.5, 97.5])
    # Wilcoxon needs a non-degenerate signed set; guard tiny/degenerate cases.
    try:
        w_p = float(stats.wilcoxon(diff, zero_method="wilcox",
                                   alternative="two-sided").pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "n_vessels": n,
        "macro_improve_pct": round(improve, 3),
        "ci95_pct": [round(float(lo), 3), round(float(hi), 3)],
        "wilcoxon_p": w_p,
        "model_wins": int((model_pd > mce_pd).sum()),
        "median_vessel_gap_nats": round(float(np.median(diff)), 4),
    }


def contrast_stats(base_pd: dict, model_pd: dict, rng) -> dict:
    """Generic paired contrast on the intersection of two per-vessel dicts,
    each mapping mmsi -> per-decision LL. base is the subtrahend."""
    shared = sorted(set(base_pd) & set(model_pd))
    b = np.array([base_pd[v] for v in shared])
    m = np.array([model_pd[v] for v in shared])
    return paired_stats(b, m, rng)


def load(eval_dir: Path, name: str):
    p = eval_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="runs/eval")
    ap.add_argument("--out", default="runs/eval/significance.json")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    rng = np.random.default_rng(0)

    ctx = {s: load(eval_dir, f) for s, f in CTX_FILES.items()}
    nc = {s: load(eval_dir, f) for s, f in NC_FILES.items()}
    ctx = {s: d for s, d in ctx.items() if d is not None}
    nc = {s: d for s, d in nc.items() if d is not None}
    print(f"CTX seeds present: {sorted(ctx)} | NC seeds present: {sorted(nc)}")
    if not nc:
        print("WARNING: no PEMIRL_NOCTX per-episode files found; the capacity/"
              "context decomposition needs runs/eval/per_episode_ll_noctx*.json "
              "(run eval_robustness.py --config configs/pemirl_noctx.yaml "
              "--pemirl-ckpt runs/pemirl_noctx_s<k>/pemirl.pt --tag noctx[_s<k>]).")

    result: dict = {}
    for split in ("test", "temporal_shift"):
        result[split] = {}
        for variant, field in VARIANTS.items():
            block: dict = {"per_seed": {}, "summary": {}}
            # per-vessel dicts per seed and model
            ctx_pv = {s: per_vessel_pd(d[split], field) for s, d in ctx.items()
                      if split in d}
            nc_pv = {s: per_vessel_pd(d[split], field) for s, d in nc.items()
                     if split in d}
            # per-vessel dict for MCE (identical across the two files); take
            # from whichever ctx seed is present, per seed.
            mce_pv = {}
            for s, d in ctx.items():
                if split in d:
                    mce_pv[s] = {v: mce for v, (mce, _) in ctx_pv[s].items()}

            def acc(contrast_by_seed):
                vals = [v["macro_improve_pct"] for v in contrast_by_seed.values()]
                return {
                    "seeds": sorted(contrast_by_seed),
                    "mean_improve_pct": round(float(np.mean(vals)), 3) if vals else None,
                    "min_improve_pct": round(float(np.min(vals)), 3) if vals else None,
                    "max_improve_pct": round(float(np.max(vals)), 3) if vals else None,
                    "all_same_sign": bool(len(vals) and
                                          (all(x > 0 for x in vals) or
                                           all(x < 0 for x in vals))),
                }

            cap, ctxc, full = {}, {}, {}
            for s in sorted(set(ctx_pv) | set(nc_pv)):
                block["per_seed"].setdefault(f"seed_{s}", {})
                # full: CTX - MCE
                if s in ctx_pv:
                    base = {v: mce for v, (mce, _) in ctx_pv[s].items()}
                    modl = {v: mo for v, (_, mo) in ctx_pv[s].items()}
                    full[s] = contrast_stats(base, modl, rng)
                    block["per_seed"][f"seed_{s}"]["full (CTX-MCE)"] = full[s]
                # capacity: NC - MCE
                if s in nc_pv:
                    base = {v: mce for v, (mce, _) in nc_pv[s].items()}
                    modl = {v: mo for v, (_, mo) in nc_pv[s].items()}
                    cap[s] = contrast_stats(base, modl, rng)
                    block["per_seed"][f"seed_{s}"]["capacity (NC-MCE)"] = cap[s]
                # context: CTX - NC  (same seed => same query split => aligned)
                if s in ctx_pv and s in nc_pv:
                    base = {v: mo for v, (_, mo) in nc_pv[s].items()}
                    modl = {v: mo for v, (_, mo) in ctx_pv[s].items()}
                    ctxc[s] = contrast_stats(base, modl, rng)
                    block["per_seed"][f"seed_{s}"]["context (CTX-NC)"] = ctxc[s]

            block["summary"] = {"capacity (NC-MCE)": acc(cap),
                                "context (CTX-NC)": acc(ctxc),
                                "full (CTX-MCE)": acc(full)}
            result[split][variant] = block

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}\n")

    # readable digest: test / random_support
    tr = result["test"]["random_support"]
    print("=== test / random_support ===")
    for s, blk in sorted(tr["per_seed"].items()):
        print(f"  {s}:")
        for name, st in blk.items():
            ci = st["ci95_pct"]
            print(f"    {name:20s}  {st['macro_improve_pct']:+7.2f}%  "
                  f"CI[{ci[0]:+.2f},{ci[1]:+.2f}]  "
                  f"wilcoxon p={st['wilcoxon_p']:.2e}  "
                  f"wins {st['model_wins']}/{st['n_vessels']}")
    print("  summary:")
    for name, sm in tr["summary"].items():
        print(f"    {name:20s}  mean {sm['mean_improve_pct']}%  "
              f"seeds {sm['seeds']}  same_sign={sm['all_same_sign']}")


if __name__ == "__main__":
    main()
