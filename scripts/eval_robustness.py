"""Robustness pass over the LL comparison (meta-learning data-design audit).

Computes per-episode log-likelihoods for the released MCE-IRL and PEMIRL
checkpoints and re-aggregates them three ways:

1. micro vs MACRO: per-decision LL (dominated by high-traffic vessels; the
   test top-3 vessels hold ~38% of episodes) vs the mean of per-VESSEL LLs
   (each task counts once — the honest metric for a meta-learning claim).
2. random vs TEMPORAL support/query: the paper's support sets are a random
   permutation of each vessel's episodes; deployment-realistic conditioning
   infers z from the vessel's EARLIEST episodes only (sorted by year, month,
   voyage_index) and predicts the later ones.
3. truncation sensitivity: excluding the 37 episodes truncated at the
   512-step horizon (whose goals are artifacts of the cap).

    python scripts/eval_robustness.py --config configs/pemirl.yaml --released

Writes <workdir>/eval/robustness.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import (RELEASED_MCE_CKPT, RELEASED_PEMIRL_CKPT, base_parser,
                    load_cfg, load_pipeline, log, workdir)

from arctic_meta_irl.algos.mce_irl import MCEIRL, soft_value_iteration
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.eval.rollout import _episode_tensors
from arctic_meta_irl.utils.seeding import resolve_device, set_seed

TRUNC_LEN = 511  # episodes with >= this many decisions hit the horizon cap


def mce_per_episode_ll(mce: MCEIRL, episodes) -> dict[int, float]:
    """{voyage_index: total LL} — mirrors MCEIRL.log_likelihood exactly."""
    out = {}
    for key, eps in mce._group(episodes).items():
        ep0 = eps[0]
        phi_all = mce.features.phi_all_states(ep0.goal, ep0.year,
                                              ep0.month, ep0.mmsi)
        r = phi_all @ mce.theta
        H = max(len(ep) for ep in eps)
        vi = soft_value_iteration(mce.mdp, r, ep0.goal, H, mce.temperature)
        for ep in eps:
            out[ep.voyage_index] = float(
                vi.log_pi[np.arange(len(ep)), ep.states[:-1],
                          ep.actions].sum())
    return out


@torch.no_grad()
def pemirl_per_episode_ll(pem: PEMIRL, support_map, fb, mdp
                          ) -> dict[int, float]:
    """{voyage_index: total LL} on query episodes, z from each support set."""
    out = {}
    for mmsi, (support, query) in support_map.items():
        z = pem.infer_context_support(
            [_episode_tensors(ep, fb) for ep in support], fixed=True)
        for ep in query:
            phi, act = _episode_tensors(ep, fb)
            phi, act = phi.to(pem.device), act.to(pem.device)
            mask = torch.from_numpy(
                mdp.action_mask[ep.states[:-1]]).to(pem.device)
            lp = pem.policy.log_prob_action(phi, mask, act,
                                            z.expand(phi.size(0), -1))
            out[ep.voyage_index] = float(lp.sum().item())
    return out


def support_split_random(episodes, k, seed):
    """The paper's scheme (mirrors eval.rollout.pemirl_support_sets)."""
    rng = np.random.default_rng(seed)
    by_v = defaultdict(list)
    for ep in episodes:
        by_v[ep.mmsi].append(ep)
    out = {}
    for mmsi, eps in by_v.items():
        if len(eps) < 2:
            continue
        idx = rng.permutation(len(eps))
        kk = min(k, len(eps) - 1)
        out[mmsi] = ([eps[i] for i in idx[:kk]], [eps[i] for i in idx[kk:]])
    return out


def support_split_temporal(episodes, k):
    """Deployment-realistic: support = the vessel's EARLIEST k episodes."""
    by_v = defaultdict(list)
    for ep in episodes:
        by_v[ep.mmsi].append(ep)
    out = {}
    for mmsi, eps in by_v.items():
        if len(eps) < 2:
            continue
        eps = sorted(eps, key=lambda e: (e.year, e.month, e.voyage_index))
        kk = min(k, len(eps) - 1)
        out[mmsi] = (eps[:kk], eps[kk:])
    return out


def aggregate(query_eps, mce_ll, pem_ll, drop_truncated=False):
    """micro (per-decision) + macro (mean per-vessel) LLs on the query set."""
    if drop_truncated:
        query_eps = [ep for ep in query_eps if len(ep) < TRUNC_LEN]
    by_v = defaultdict(list)
    for ep in query_eps:
        by_v[ep.mmsi].append(ep)
    tm = tp = n = 0.0
    vm, vp = [], []
    for mmsi, eps in by_v.items():
        m = sum(mce_ll[e.voyage_index] for e in eps)
        p = sum(pem_ll[e.voyage_index] for e in eps)
        d = sum(len(e) for e in eps)
        tm += m; tp += p; n += d
        vm.append(m / d); vp.append(p / d)
    return {
        "n_vessels": len(by_v),
        "n_decisions": int(n),
        "mce_micro": tm / n, "pemirl_micro": tp / n,
        "mce_macro": float(np.mean(vm)), "pemirl_macro": float(np.mean(vp)),
        "pemirl_wins_vessels": int(sum(p > m for m, p in zip(vm, vp))),
    }


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--released", action="store_true")
    p.add_argument("--splits", nargs="+", default=["test", "temporal_shift"])
    p.add_argument("--mce-ckpt", default=None,
                   help="explicit MCE theta.npz (overrides --released/config)")
    p.add_argument("--pemirl-ckpt", default=None,
                   help="explicit PEMIRL .pt (overrides --released/config); "
                        "pair with --config pemirl_noctx.yaml for the ablation")
    p.add_argument("--tag", default=None,
                   help="suffix for the output json (robustness_<tag>.json)")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    k = int(cfg.get("pemirl", {}).get("support_set_size", 3))

    mdp, fb, eps = load_pipeline(cfg)
    mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
    mce.load(Path(args.mce_ckpt or (RELEASED_MCE_CKPT if args.released
                                    else cfg["mce_irl"]["ckpt"])))
    pem = PEMIRL(fb.dim, mdp.n_actions, cfg["pemirl"], device=device)
    pem.load(Path(args.pemirl_ckpt or
                  (RELEASED_PEMIRL_CKPT if args.released else
                   Path(cfg["pemirl"].get("ckpt_dir", "runs/pemirl/"))
                   / "pemirl.pt")))
    pem.eval()

    out = {}
    per_ep: dict[str, dict] = {}
    for split in args.splits:
        episodes = eps.get(split, [])
        if not episodes:
            continue
        log.info("[%s] MCE per-episode LLs (support-independent)...", split)
        mce_ll = mce_per_episode_ll(mce, episodes)

        splits_defs = {
            "random_support": support_split_random(episodes, k,
                                                   int(cfg["seed"])),
            "temporal_support": support_split_temporal(episodes, k),
        }
        out[split] = {}
        # per-episode dump for downstream analysis (per-vessel scatter etc.)
        per_ep[split] = {
            str(ep.voyage_index): {
                "mmsi": int(ep.mmsi), "n_decisions": len(ep),
                "category": ep.category, "year": int(ep.year),
                "month": int(ep.month), "truncated": len(ep) >= TRUNC_LEN,
                "mce_ll": mce_ll[ep.voyage_index],
            } for ep in episodes}
        for name, smap in splits_defs.items():
            pem_ll = pemirl_per_episode_ll(pem, smap, fb, mdp)
            query = [e for _, (_, q) in smap.items() for e in q]
            for ep in query:
                per_ep[split][str(ep.voyage_index)][
                    "pemirl_ll_" + name] = pem_ll[ep.voyage_index]
            out[split][name] = aggregate(query, mce_ll, pem_ll)
            out[split][name + "_no_trunc"] = aggregate(
                query, mce_ll, pem_ll, drop_truncated=True)
            a = out[split][name]
            log.info("[%s/%s] micro: MCE %.4f vs PEM %.4f | "
                     "MACRO: MCE %.4f vs PEM %.4f | PEM wins %d/%d vessels",
                     split, name, a["mce_micro"], a["pemirl_micro"],
                     a["mce_macro"], a["pemirl_macro"],
                     a["pemirl_wins_vessels"], a["n_vessels"])

    suffix = f"_{args.tag}" if args.tag else ""
    dest = workdir(cfg) / "eval" / f"robustness{suffix}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    (dest.parent / f"per_episode_ll{suffix}.json").write_text(json.dumps(per_ep))
    log.info("Robustness results written to %s (+ per_episode_ll%s.json)",
             dest, suffix)


if __name__ == "__main__":
    main()
