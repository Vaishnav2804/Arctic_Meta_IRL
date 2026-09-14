"""14 — Evaluate imitation/IRL baselines with the paper's LL protocol.

    python scripts/14_eval_baselines.py --algo bc
    python scripts/14_eval_baselines.py --algo gail --seed 0
    python scripts/14_eval_baselines.py --algo deep_mce --tag deep_mce_s1 \
        --ckpt runs/deep_mce_s1/reward.pt

Computes per-episode log-likelihoods on the test + temporal_shift splits for
one baseline (bc | seq_lstm | seq_transformer | gail | gcl | deep_mce),
paired with the released MCE-IRL per-episode LLs, and aggregates them the
same three ways as eval_robustness.py:

* over ALL episodes of the split (baselines need no support set), and
* restricted to the PEMIRL random_support / temporal_support QUERY sets
  (same seed => same membership), so per-vessel paired contrasts against the
  AIRL / PEMIRL per-episode files are apples-to-apples.

Writes runs/eval/robustness_<tag>.json + runs/eval/per_episode_ll_<tag>.json
(fields: mce_ll, model_ll, model_ll_random_support, model_ll_temporal_support).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (RELEASED_MCE_CKPT, base_parser, load_cfg, load_pipeline,
                    log, seeded_dir, workdir)
from eval_robustness import (TRUNC_LEN, mce_per_episode_ll,
                             support_split_random, support_split_temporal)

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.utils.seeding import resolve_device, set_seed

ALGOS = ("bc", "seq_lstm", "seq_transformer", "gail", "gcl", "deep_mce")
DEFAULT_CFG = {a: f"configs/{a}.yaml" for a in ALGOS}
CKPT_NAME = {"bc": "policy.pt", "seq_lstm": "policy.pt",
             "seq_transformer": "policy.pt", "gail": "gail.pt",
             "gcl": "gcl.pt", "deep_mce": "reward.pt"}


def load_model(algo: str, cfg, mdp, fb, device: str, ckpt: Path):
    """Returns (model_or_policy, per_episode_ll_fn(episodes) -> dict)."""
    import torch
    if algo == "deep_mce":
        from arctic_meta_irl.algos.deep_mce import DeepMCEIRL
        m = cfg["deep_mce"]
        model = DeepMCEIRL(mdp, fb, hidden=tuple(m["hidden"]),
                           temperature=float(m["vi_temperature"]),
                           device=device, seed=int(cfg["seed"]))
        model.load(ckpt)
        return model, model.per_episode_ll
    from arctic_meta_irl.algos.bc import build_policy, per_episode_ll
    if algo in ("bc", "seq_lstm", "seq_transformer"):
        policy = build_policy(cfg["bc"].get("arch", "mlp"), fb.dim,
                              mdp.n_actions, dict(cfg["bc"]))
        state = torch.load(ckpt, map_location=device)["policy"]
    elif algo == "gail":
        from arctic_meta_irl.algos.gail import GAIL
        g = GAIL(fb.dim, mdp.n_actions, cfg["gail"], device=device)
        g.load(ckpt)
        policy = g.policy
        state = None
    elif algo == "gcl":
        from arctic_meta_irl.algos.gcl import GCL
        g = GCL(fb.dim, mdp.n_actions, cfg["gcl"], device=device)
        g.load(ckpt)
        policy = g.policy
        state = None
    else:
        raise ValueError(algo)
    if state is not None:
        policy.load_state_dict(state)
    policy.to(device).eval()
    return policy, lambda eps: per_episode_ll(policy, eps, fb, mdp, device)


def aggregate(query_eps, mce_ll, model_ll, drop_truncated=False):
    """micro (per-decision) + macro (mean per-vessel) — eval_robustness
    schema with 'model_*' in place of 'pemirl_*'."""
    if drop_truncated:
        query_eps = [ep for ep in query_eps if len(ep) < TRUNC_LEN]
    by_v = defaultdict(list)
    for ep in query_eps:
        by_v[ep.mmsi].append(ep)
    tm = tp = n = 0.0
    vm, vp = [], []
    for _, eps_v in by_v.items():
        m = sum(mce_ll[e.voyage_index] for e in eps_v)
        p = sum(model_ll[e.voyage_index] for e in eps_v)
        d = sum(len(e) for e in eps_v)
        tm += m; tp += p; n += d
        vm.append(m / d); vp.append(p / d)
    return {
        "n_vessels": len(by_v),
        "n_decisions": int(n),
        "mce_micro": tm / n, "model_micro": tp / n,
        "mce_macro": float(np.mean(vm)), "model_macro": float(np.mean(vp)),
        "model_wins_vessels": int(sum(p > m for m, p in zip(vm, vp))),
    }


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--algo", required=True, choices=ALGOS)
    p.add_argument("--ckpt", default=None, help="checkpoint path override")
    p.add_argument("--mce-ckpt", default=RELEASED_MCE_CKPT)
    p.add_argument("--tag", default=None,
                   help="output suffix (default: <algo>[_s<seed>])")
    p.add_argument("--splits", nargs="+", default=["test", "temporal_shift"])
    args = p.parse_args()
    if args.config == "configs/default.yaml":       # not explicitly given
        args.config = DEFAULT_CFG[args.algo]
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    seed = int(cfg["seed"])
    tag = args.tag or (args.algo if seed == 0 else f"{args.algo}_s{seed}")
    k = 3  # support_set_size of the PEMIRL protocol (query-set membership)

    mdp, fb, eps = load_pipeline(cfg)
    mce = MCEIRL(mdp, fb, seed=seed)
    mce.load(Path(args.mce_ckpt))

    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        blk = cfg.get(args.algo if args.algo in ("gail", "gcl", "deep_mce")
                      else "bc", {})
        base = (Path(blk["ckpt"]).parent if args.algo == "deep_mce"
                else Path(blk["ckpt_dir"]))
        ckpt = seeded_dir(base, seed) / CKPT_NAME[args.algo]
    log.info("Evaluating %s from %s (tag=%s)", args.algo, ckpt, tag)
    _, ll_fn = load_model(args.algo, cfg, mdp, fb, device, ckpt)

    out, per_ep = {}, {}
    for split in args.splits:
        episodes = eps.get(split, [])
        if not episodes:
            log.warning("split %s empty; skipping", split)
            continue
        log.info("[%s] MCE per-episode LLs...", split)
        mce_ll = mce_per_episode_ll(mce, episodes)
        log.info("[%s] %s per-episode LLs...", split, args.algo)
        model_ll = ll_fn(episodes)

        per_ep[split] = {
            str(ep.voyage_index): {
                "mmsi": int(ep.mmsi), "n_decisions": len(ep),
                "category": ep.category, "year": int(ep.year),
                "month": int(ep.month), "truncated": len(ep) >= TRUNC_LEN,
                "mce_ll": mce_ll[ep.voyage_index],
                "model_ll": model_ll[ep.voyage_index],
            } for ep in episodes}

        out[split] = {"all_episodes": aggregate(episodes, mce_ll, model_ll),
                      "all_episodes_no_trunc": aggregate(
                          episodes, mce_ll, model_ll, drop_truncated=True)}
        variants = {
            "random_support": support_split_random(episodes, k, seed),
            "temporal_support": support_split_temporal(episodes, k),
        }
        for name, smap in variants.items():
            query = [e for _, (_, q) in smap.items() for e in q]
            for ep in query:
                per_ep[split][str(ep.voyage_index)][
                    "model_ll_" + name] = model_ll[ep.voyage_index]
            out[split][name] = aggregate(query, mce_ll, model_ll)
            out[split][name + "_no_trunc"] = aggregate(
                query, mce_ll, model_ll, drop_truncated=True)
            a = out[split][name]
            log.info("[%s/%s] micro: MCE %.4f vs %s %.4f | "
                     "MACRO: MCE %.4f vs %.4f | wins %d/%d vessels",
                     split, name, a["mce_micro"], args.algo,
                     a["model_micro"], a["mce_macro"], a["model_macro"],
                     a["model_wins_vessels"], a["n_vessels"])

    result = {"model": args.algo, "ckpt": str(ckpt), "seed": seed, **out}
    dest = workdir(cfg) / "eval" / f"robustness_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    (dest.parent / f"per_episode_ll_{tag}.json").write_text(json.dumps(per_ep))
    log.info("Written %s (+ per_episode_ll_%s.json)", dest, tag)


if __name__ == "__main__":
    main()
