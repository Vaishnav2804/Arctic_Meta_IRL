"""13 — Train Deep-MCE (nonlinear reward under exact soft-VI MaxEnt IRL).

    python scripts/13_train_deep_mce.py --config configs/deep_mce.yaml

Mirrors scripts/03_train_mce_irl.py with r(s) = MLP(phi(s)) instead of
theta . phi(s): same grouping, same exact backward/forward passes, same
val-LL checkpoint gate — the "capacity without adversarial training" rung.

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/deep_mce/reward.pt (+ tb logs, val_ll.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.deep_mce import DeepMCEIRL
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n-iters", type=int, default=None, help="override config")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    m = cfg["deep_mce"]

    mdp, fb, eps = load_pipeline(cfg)
    model = DeepMCEIRL(mdp, fb, hidden=tuple(m["hidden"]),
                       lr=float(m["lr"]),
                       weight_decay=float(m["weight_decay"]),
                       grad_clip=float(m["grad_clip"]),
                       temperature=float(m["vi_temperature"]),
                       device=device, seed=int(cfg["seed"]))

    _ckpt = Path(m["ckpt"])
    ckpt = seeded_dir(_ckpt.parent, int(cfg["seed"])) / _ckpt.name
    tb = TBWriter(ckpt.parent / "tb", enabled=bool(cfg["logging"]["tensorboard"]))
    best = {"val_ll_per_decision": -float("inf"), "iter": -1}
    eval_every = int(m.get("eval_every", 5))

    def callback(it: int, train_ll: float) -> None:
        tb.scalar("train/ll_per_decision", train_ll, it)
        if it % eval_every != 0 or not eps.get("val"):
            return
        ll, n = model.log_likelihood(eps["val"])
        v = ll / max(n, 1)
        tb.scalar("val/ll_per_decision", v, it)
        log.info("iter %4d | train LL/dec %.4f | val LL/dec %.4f",
                 it, train_ll, v)
        if v > best["val_ll_per_decision"]:
            best.update(val_ll_per_decision=v, iter=it)
            model.save(ckpt)

    n_iters = args.n_iters or int(m["n_iters"])
    model.fit(eps["train"], n_iters=n_iters, log_every=10, callback=callback)
    if best["iter"] < 0:
        model.save(ckpt)
    tb.close()

    with open(ckpt.parent / "val_ll.json", "w") as f:
        json.dump(best, f, indent=2)
    log.info("Best val LL/decision %.4f at iter %d -> %s",
             best["val_ll_per_decision"], best["iter"], ckpt)


if __name__ == "__main__":
    main()
