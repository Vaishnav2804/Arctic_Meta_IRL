"""Compare a freshly reproduced evaluation against the shipped paper numbers.

    python scripts/compare_results.py
    python scripts/compare_results.py --fresh runs/eval/results.json \
        --reference results/reference/results.json

Prints a side-by-side table (metric | reference | reproduced | abs diff |
rel %) with per-metric PASS/FAIL under these tolerances:
  - per-decision log-likelihoods : abs diff <= 0.02
  - rate metrics (reached_goal, cell_overlap): rel <= 5% OR abs diff <= 0.02
  - everything else              : rel diff <= 5%

Exits nonzero if any HEADLINE metric fails: the four LL values
(mce_irl / pemirl x test / temporal_shift) and both test FEE values.

If scripts/08_reward_transfer.py --eval-only wrote a goal-reach summary
(runs/ppo_on_pemirl/goal_reach_summary.json), it is printed as well.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LL_ABS_TOL = 0.02
REL_TOL_PCT = 5.0
RATE_ABS_TOL = 0.02

HEADLINE = [
    ("test", "mce_irl_ll_per_decision"),
    ("test", "pemirl_ll_per_decision"),
    ("temporal_shift", "mce_irl_ll_per_decision"),
    ("temporal_shift", "pemirl_ll_per_decision"),
    ("test", "mce_irl_fee"),
    ("test", "pemirl_fee"),
]


def _is_ll(name: str) -> bool:
    return name.endswith("_ll_per_decision")


def _is_rate(name: str) -> bool:
    return name.endswith("reached_goal") or "cell_overlap" in name


def check(name: str, ref: float, rep: float) -> tuple[bool, float, float]:
    """(passed, abs_diff, rel_pct) under the metric-specific tolerance."""
    abs_diff = abs(rep - ref)
    rel_pct = 100.0 * abs_diff / max(abs(ref), 1e-12)
    if _is_ll(name):
        return abs_diff <= LL_ABS_TOL, abs_diff, rel_pct
    if _is_rate(name):
        return rel_pct <= REL_TOL_PCT or abs_diff <= RATE_ABS_TOL, \
            abs_diff, rel_pct
    return rel_pct <= REL_TOL_PCT, abs_diff, rel_pct


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh", default="runs/eval/results.json",
                   help="freshly reproduced results (scripts/06)")
    p.add_argument("--reference", default="results/reference/results.json",
                   help="shipped paper numbers")
    p.add_argument("--summary", default="runs/ppo_on_pemirl/goal_reach_summary.json",
                   help="optional goal-reach summary from scripts/08 --eval-only")
    args = p.parse_args()

    with open(args.reference) as f:
        ref = json.load(f)
    if not Path(args.fresh).exists():
        print(f"FAIL: fresh results {args.fresh} not found; run "
              "scripts/06_evaluate.py first (e.g. `make reproduce`).")
        return 1
    with open(args.fresh) as f:
        rep = json.load(f)

    headline_failed = []
    w = max((len(f"{s}/{m}") for s in ref for m in ref[s]), default=40)
    header = (f"{'metric'.ljust(w)}  {'reference':>12}  {'reproduced':>12}  "
              f"{'abs diff':>10}  {'rel %':>8}  status")
    print(header)
    print("-" * len(header))

    for split, metrics in ref.items():
        for name, ref_v in metrics.items():
            label = f"{split}/{name}"
            is_headline = (split, name) in HEADLINE
            tag = " [headline]" if is_headline else ""
            rep_v = rep.get(split, {}).get(name)
            if rep_v is None:
                print(f"{label.ljust(w)}  {ref_v:>12.4f}  {'—':>12}  "
                      f"{'—':>10}  {'—':>8}  MISSING{tag}")
                if is_headline:
                    headline_failed.append(label)
                continue
            ok, abs_diff, rel_pct = check(name, float(ref_v), float(rep_v))
            status = "PASS" if ok else "FAIL"
            print(f"{label.ljust(w)}  {ref_v:>12.4f}  {rep_v:>12.4f}  "
                  f"{abs_diff:>10.4f}  {rel_pct:>8.2f}  {status}{tag}")
            if is_headline and not ok:
                headline_failed.append(label)

    # ---- optional reward-transfer goal-reach summary -------------------------
    if Path(args.summary).exists():
        with open(args.summary) as f:
            s = json.load(f)
        print("\nReward transfer (scripts/08 --eval-only, "
              f"source={s.get('source', '?')}):")
        print(f"  goal-reach first {s['first']:.3f} | peak {s['peak']:.3f} "
              f"(iter {s.get('peak_iter', '?')}) | steady (last 20%) "
              f"{s['steady']:.3f} over {s['n_iters']} iters")

    if headline_failed:
        print(f"\nFAIL: {len(headline_failed)} headline metric(s) out of "
              f"tolerance: {', '.join(headline_failed)}")
        return 1
    print("\nPASS: all headline metrics within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
