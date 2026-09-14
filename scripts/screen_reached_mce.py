"""Second screening pass: does the MCE-IRL soft-VI decode reach the goal?

Runs mce.greedy_route on the subset of query episodes that already passed the
transfer-agent screen (runs/eval/reached_screen.json), so the gallery can, if
wanted, require all three decodes to terminate at the goal.

Output: runs/eval/reached_screen_mce.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import arctic_meta_irl.data  # noqa: F401
import common

import torch

torch.set_num_threads(8)

from arctic_meta_irl.utils.config import load_config
from arctic_meta_irl.utils.seeding import set_seed
from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.eval.rollout import pemirl_support_sets
from arctic_meta_irl.eval.metrics import route_metrics, route_length_km

warnings.filterwarnings("ignore", category=UserWarning)

MIN_LR, MAX_LR = 0.7, 1.6

cfg = load_config("configs/pemirl.yaml")
set_seed(int(cfg["seed"]))
mdp, fb, eps = common.load_pipeline(cfg)
mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
mce.load("models/mce_irl/theta.npz")
HORIZON = int(cfg["mdp"]["max_horizon"])

with open("runs/eval/reached_screen.json") as f:
    screen = json.load(f)
keep = {(r["mmsi"], r["start"], r["goal"], r["year"], r["month"])
        for r in screen
        if r["transfer_reached"] and r["airl_reached"]
        and min(r["transfer_length_ratio"], r["airl_length_ratio"]) >= MIN_LR
        and max(r["transfer_length_ratio"], r["airl_length_ratio"]) <= MAX_LR}
print(f"{len(keep)} pairs passed the transfer screen", flush=True)

support_map = pemirl_support_sets(eps["test"],
                                  int(cfg["pemirl"]["support_set_size"]),
                                  seed=int(cfg["seed"]))
out, t0 = [], time.time()
for mmsi, (sup, qry) in support_map.items():
    for ep in qry:
        key = (int(ep.mmsi), int(ep.states[0]), int(ep.goal),
               int(ep.year), int(ep.month))
        if key not in keep:
            continue
        real = list(ep.states)
        r = mce.greedy_route(int(real[0]), int(ep.goal), ep.year, ep.month,
                             ep.mmsi, horizon=HORIZON)
        met = route_metrics(mdp, r, real)
        out.append({"mmsi": key[0], "start": key[1], "goal": key[2],
                    "year": key[3], "month": key[4],
                    "real_km": float(route_length_km(mdp, real)),
                    "mce_reached": bool(met["reached_goal"]),
                    "mce_steps": len(r),
                    "mce_hausdorff_km": float(met["hausdorff_km"]),
                    "mce_length_ratio": float(met["length_ratio"])})
        print(f"{len(out):3d}/{len(keep)} | {out[-1]['real_km']:6.0f} km | "
              f"reached={out[-1]['mce_reached']} steps={out[-1]['mce_steps']} "
              f"H={out[-1]['mce_hausdorff_km']:6.1f}", flush=True)

with open("runs/eval/reached_screen_mce.json", "w") as f:
    json.dump(out, f, indent=1)
nr = sum(r["mce_reached"] for r in out)
print(f"\ndone in {time.time()-t0:.0f}s | MCE reached {nr}/{len(out)}")
