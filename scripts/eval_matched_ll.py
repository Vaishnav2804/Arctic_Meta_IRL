"""Matched-set log-likelihood comparison (audit fix).

In 06_evaluate the two LLs are computed on different episode sets: MCE-IRL on
ALL episodes of the split, PEMIRL only on the QUERY episodes (vessels with >= 2
episodes, minus the support set used to infer z). This script recomputes the
MCE-IRL LL restricted to exactly PEMIRL's query set, so the comparison is
apples-to-apples.

    python scripts/eval_matched_ll.py --config configs/pemirl.yaml --released

Writes <workdir>/eval/matched_ll.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import (RELEASED_MCE_CKPT, RELEASED_PEMIRL_CKPT, base_parser,
                    load_cfg, load_pipeline, log, workdir)

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.eval.rollout import (pemirl_log_likelihood,
                                          pemirl_support_sets)
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--released", action="store_true",
                   help="use the released checkpoints under models/")
    p.add_argument("--splits", nargs="+", default=["test", "temporal_shift"])
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    support_size = int(cfg.get("pemirl", {}).get("support_set_size", 3))

    mdp, fb, eps = load_pipeline(cfg)

    mce_ckpt = Path(RELEASED_MCE_CKPT if args.released
                    else cfg["mce_irl"]["ckpt"])
    mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
    mce.load(mce_ckpt)

    pem_ckpt = Path(RELEASED_PEMIRL_CKPT if args.released else
                    Path(cfg["pemirl"].get("ckpt_dir", "runs/pemirl/"))
                    / "pemirl.pt")
    pem = PEMIRL(fb.dim, mdp.n_actions, cfg["pemirl"], device=device)
    pem.load(pem_ckpt)
    pem.eval()

    out: dict[str, dict] = {}
    for split in args.splits:
        episodes = eps.get(split, [])
        if not episodes:
            log.warning("split %s empty; skipping", split)
            continue
        # exactly the same support/query partition pemirl_log_likelihood uses
        support_map = pemirl_support_sets(episodes, support_size,
                                          seed=int(cfg["seed"]))
        query = [ep for _, (_, qry) in sorted(support_map.items())
                 for ep in qry]
        n_all = sum(len(ep) for ep in episodes)

        mce_all_ll, mce_all_n = mce.log_likelihood(episodes)
        mce_q_ll, mce_q_n = mce.log_likelihood(query)
        pem_q_ll, pem_q_n = pemirl_log_likelihood(
            pem, episodes, fb, mdp, support_size=support_size,
            seed=int(cfg["seed"]))
        assert pem_q_n == mce_q_n, (pem_q_n, mce_q_n)

        out[split] = {
            "n_episodes_all": len(episodes),
            "n_episodes_query": len(query),
            "n_decisions_all": n_all,
            "n_decisions_query": mce_q_n,
            "mce_irl_ll_all_episodes": mce_all_ll / max(mce_all_n, 1),
            "mce_irl_ll_query_only": mce_q_ll / max(mce_q_n, 1),
            "pemirl_ll_query_only": pem_q_ll / max(pem_q_n, 1),
        }
        m, q = out[split]["mce_irl_ll_query_only"], \
            out[split]["pemirl_ll_query_only"]
        out[split]["pemirl_vs_mce_matched_improvement_pct"] = \
            100.0 * (float(__import__("numpy").exp(q - m)) - 1.0)
        log.info("%s: MCE all=%.4f | MCE query=%.4f | PEMIRL query=%.4f "
                 "(%d query decisions)", split,
                 out[split]["mce_irl_ll_all_episodes"], m, q, mce_q_n)

    dest = workdir(cfg) / "eval" / "matched_ll.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    log.info("Matched-set LL written to %s", dest)


if __name__ == "__main__":
    main()
