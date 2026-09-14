"""03 — Train pooled MCE-IRL (single shared linear reward).

    python scripts/03_train_mce_irl.py --config configs/mce_irl.yaml

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : mce_irl.ckpt (runs/mce_irl/theta.npz) + runs/mce_irl/val_ll.json
"""
from __future__ import annotations

import json
from pathlib import Path

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n-iters", type=int, default=None, help="override config")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    m = cfg["mce_irl"]

    mdp, fb, eps = load_pipeline(cfg)
    model = MCEIRL(mdp, fb,
                   lr=float(m["lr"]), l2=float(m["l2"]),
                   temperature=float(m["vi_temperature"]),
                   group_by_goal=bool(m["group_by_goal"]),
                   seed=int(cfg["seed"]))

    _ckpt = Path(m["ckpt"])
    ckpt = seeded_dir(_ckpt.parent, int(cfg["seed"])) / _ckpt.name
    tb = TBWriter(ckpt.parent / "tb", enabled=bool(cfg["logging"]["tensorboard"]))
    best = {"val_ll_per_decision": -float("inf"), "iter": -1}

    eval_every = 10

    def callback(it: int, train_ll: float, _theta) -> None:
        tb.scalar("train/ll_per_decision", train_ll, it)
        if it % eval_every != 0 or not eps.get("val"):
            return
        ll, n = model.log_likelihood(eps["val"])
        v = ll / max(n, 1)
        tb.scalar("val/ll_per_decision", v, it)
        log.info("iter %4d | train LL/dec %.4f | val LL/dec %.4f", it, train_ll, v)
        if v > best["val_ll_per_decision"]:
            best.update(val_ll_per_decision=v, iter=it)
            model.save(ckpt)

    n_iters = args.n_iters or int(m["n_iters"])
    model.fit(eps["train"], n_iters=n_iters, log_every=10, callback=callback)
    if best["iter"] < 0:  # no callback fired (e.g. tiny run) — save final
        model.save(ckpt)
    tb.close()

    with open(ckpt.parent / "val_ll.json", "w") as f:
        json.dump(best, f, indent=2)
    log.info("Best val LL/decision %.4f at iter %d -> %s",
             best["val_ll_per_decision"], best["iter"], ckpt)


if __name__ == "__main__":
    main()
