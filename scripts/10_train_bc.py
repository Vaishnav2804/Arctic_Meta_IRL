"""10 — Train supervised imitation baselines: BC (MLP) and sequence models.

    python scripts/10_train_bc.py --config configs/bc.yaml
    python scripts/10_train_bc.py --config configs/seq_lstm.yaml
    python scripts/10_train_bc.py --config configs/seq_transformer.yaml

Maximum-likelihood next-action prediction on the full train split with early
stopping on val per-decision LL — the pure-imitation reference points for the
IRL ladder (no reward is learned).

Reads  : cached MDP / episodes / splits / features (scripts/00-01)
Writes : runs/<arch>/policy.pt (+ tb logs, val_ll.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from common import base_parser, load_cfg, load_pipeline, log, seeded_dir

from arctic_meta_irl.algos.bc import build_policy, train_bc
from arctic_meta_irl.utils.logging import TBWriter
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--epochs", type=int, default=None, help="override config")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    bc = dict(cfg["bc"])
    if args.epochs is not None:
        bc["epochs"] = args.epochs

    mdp, fb, eps = load_pipeline(cfg)
    policy = build_policy(bc.get("arch", "mlp"), fb.dim, mdp.n_actions, bc)
    n_params = sum(pp.numel() for pp in policy.parameters())
    log.info("BC[%s]: %d train / %d val episodes, obs_dim=%d, params=%d, "
             "device=%s", bc.get("arch"), len(eps["train"]), len(eps["val"]),
             fb.dim, n_params, device)

    ckpt_dir = seeded_dir(Path(bc["ckpt_dir"]), int(cfg["seed"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = TBWriter(ckpt_dir / "tb", enabled=bool(cfg["logging"]["tensorboard"]))

    best = train_bc(policy, eps["train"], eps["val"], fb, mdp, bc, device,
                    ckpt=ckpt_dir / "policy.pt", tb=tb)
    tb.close()
    with open(ckpt_dir / "val_ll.json", "w") as f:
        json.dump(best, f, indent=2)
    log.info("Done. Best val LL/decision %.4f at epoch %d -> %s",
             best["val_ll_per_decision"], best["epoch"],
             ckpt_dir / "policy.pt")


if __name__ == "__main__":
    main()
