"""07 — Generate a route for a vessel and origin/destination pair.

    python scripts/07_generate_routes.py --config configs/pemirl.yaml \
        --method pemirl --mmsi 316001234 \
        --start 860c6d2... --goal 860c6e5... --year 2022 --month 8

`--start/--goal` accept either H3 cell ids or integer state indices.
For PEMIRL the context is inferred from the vessel's training episodes
(a support set of size pemirl.support_set_size). Output: ordered H3 cells
(+ GeoJSON LineString of the cell centroids).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from common import base_parser, load_cfg, load_pipeline, log

from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, Task
from arctic_meta_irl.eval.metrics import route_length_km
from arctic_meta_irl.eval.rollout import (generate_route_pemirl,
                                          generate_route_policy)
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.utils.seeding import resolve_device, set_seed


def _state(mdp, token: str) -> int:
    if token in mdp.cell_to_idx:
        return int(mdp.cell_to_idx[token])
    s = int(token)
    assert 0 <= s < mdp.n_states, f"state index {s} out of range"
    return s


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--method", choices=["pemirl", "mce_irl", "ppo_baseline"],
                   default="pemirl")
    p.add_argument("--mmsi", type=int, required=True)
    p.add_argument("--start", required=True, help="H3 cell id or state index")
    p.add_argument("--goal", required=True, help="H3 cell id or state index")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--month", type=int, default=None)
    p.add_argument("--out", default=None, help="GeoJSON output path")
    args = p.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])

    mdp, fb, eps = load_pipeline(cfg)
    env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))
    task = Task(start=_state(mdp, args.start), goal=_state(mdp, args.goal),
                year=args.year, month=args.month, mmsi=args.mmsi)

    if args.method == "mce_irl":
        model = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
        model.load(cfg.get("mce_irl", {}).get("ckpt", "runs/mce_irl/theta.npz"))
        path = model.greedy_route(task.start, task.goal, task.year, task.month,
                                  task.mmsi, horizon=env.max_horizon)
    elif args.method == "pemirl":
        pc = cfg.get("pemirl", {"context_dim": 8})
        model = PEMIRL(fb.dim, mdp.n_actions, pc, device=device)
        model.load(Path(pc.get("ckpt_dir", "runs/pemirl/")) / "pemirl.pt")
        model.eval()
        support = [ep for ep in eps["train"] if ep.mmsi == args.mmsi]
        if not support:  # unseen vessel: fall back to any episodes it has
            support = [ep for s in eps.values() for ep in s
                       if ep.mmsi == args.mmsi]
        assert support, f"no episodes found for MMSI {args.mmsi}"
        support = support[:int(pc.get("support_set_size", 3))]
        log.info("Inferring context from %d support episodes of MMSI %d",
                 len(support), args.mmsi)
        path = generate_route_pemirl(model, env, task, support, fb)
    else:
        pb = cfg.get("ppo_baseline", {})
        policy = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=0,
                              hidden=tuple(pb.get("hidden", (256, 256))))
        policy.load_state_dict(torch.load(
            Path(pb.get("ckpt_dir", "runs/ppo_baseline/")) / "policy.pt",
            map_location=device))
        policy.to(device).eval()
        path = generate_route_policy(policy, env, task, deterministic=True,
                                     device=device)

    cells = [mdp.cells[s] for s in path]
    reached = path[-1] == task.goal
    log.info("%s route: %d cells, %.1f km, reached_goal=%s",
             args.method, len(path), route_length_km(mdp, path), reached)
    print("\n".join(cells))

    if args.out:
        coords = [[float(mdp.latlng[s][1]), float(mdp.latlng[s][0])]
                  for s in path]
        gj = {"type": "Feature",
              "properties": {"method": args.method, "mmsi": args.mmsi,
                             "reached_goal": bool(reached), "cells": cells},
              "geometry": {"type": "LineString", "coordinates": coords}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(gj, f, indent=2)
        log.info("GeoJSON written to %s", args.out)


if __name__ == "__main__":
    main()
