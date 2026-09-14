"""Screen every held-out query episode for decodes that reach the goal.

Rolls out both transfer agents on all test query episodes and records whether
each reached the goal (plus Hausdorff / length ratio). Cheap pass: no MCE-IRL
soft value iteration, only the two masked-PPO policies.

Output: runs/eval/reached_screen.json  (one record per query episode)
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

import numpy as np
import torch

torch.set_num_threads(1)

from arctic_meta_irl.utils.config import load_config
from arctic_meta_irl.utils.seeding import set_seed
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, Task
from arctic_meta_irl.eval.rollout import (generate_route_policy,
                                          pemirl_support_sets,
                                          _episode_tensors)
from arctic_meta_irl.eval.metrics import route_metrics, route_length_km

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cpu"
cfg = load_config("configs/pemirl.yaml")
set_seed(int(cfg["seed"]))

t0 = time.time()
mdp, fb, eps = common.load_pipeline(cfg)
env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))

pem = PEMIRL(fb.dim, mdp.n_actions, cfg["pemirl"], device=DEVICE)
pem.load("models/pemirl/pemirl.pt")
pem.eval()

CTX = int(cfg["pemirl"]["context_dim"])
HID = tuple(cfg["pemirl"]["policy_hidden"])

transfer_pem = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=CTX, hidden=HID)
transfer_pem.load_state_dict(
    torch.load("models/ppo_on_pemirl/policy.pt", map_location=DEVICE))
transfer_pem.to(DEVICE).eval()

transfer_airl = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=0, hidden=HID)
transfer_airl.load_state_dict(
    torch.load("runs/ppo_on_noctx/policy.pt", map_location=DEVICE))
transfer_airl.to(DEVICE).eval()
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

SUPPORT_SIZE = int(cfg["pemirl"]["support_set_size"])
support_map = pemirl_support_sets(eps["test"], SUPPORT_SIZE,
                                  seed=int(cfg["seed"]))

# z is per-vessel: infer once, reuse across that vessel's query episodes
z_by_vessel = {}
for mmsi, (sup, _q) in support_map.items():
    z_by_vessel[mmsi] = pem.infer_context_support(
        [_episode_tensors(e, fb) for e in sup], fixed=True)
print(f"inferred z for {len(z_by_vessel)} test vessels", flush=True)

cands = []
for mmsi, (sup, qry) in support_map.items():
    for ep in qry:
        if len(ep) < 8:
            continue
        cands.append((ep, route_length_km(mdp, list(ep.states))))
cands.sort(key=lambda t: t[1])
print(f"{len(cands)} candidate query episodes", flush=True)

out = []
t0 = time.time()
for j, (ep, km) in enumerate(cands):
    task = Task(start=int(ep.states[0]), goal=int(ep.goal), year=ep.year,
                month=ep.month, mmsi=ep.mmsi, category=ep.category)
    real = list(ep.states)
    rec = {"idx": j, "mmsi": int(ep.mmsi), "category": ep.category,
           "year": int(ep.year), "month": int(ep.month), "real_km": float(km),
           "n_steps": len(real), "start": int(real[0]), "goal": int(ep.goal)}
    for name, pol, z in (("transfer", transfer_pem, z_by_vessel[ep.mmsi]),
                         ("airl", transfer_airl, None)):
        r = generate_route_policy(pol, env, task, z=z, deterministic=True,
                                  device=DEVICE)
        met = route_metrics(mdp, r, real)
        rec[f"{name}_reached"] = bool(met["reached_goal"])
        rec[f"{name}_hausdorff_km"] = float(met["hausdorff_km"])
        rec[f"{name}_steps"] = len(r)
        for k, v in met.items():
            if k not in ("reached_goal", "hausdorff_km"):
                rec[f"{name}_{k}"] = float(v)
    out.append(rec)
    if (j + 1) % 25 == 0:
        both = sum(r["transfer_reached"] and r["airl_reached"] for r in out)
        print(f"{j+1}/{len(cands)} | {time.time()-t0:5.0f}s | both-reached "
              f"{both}", flush=True)

os.makedirs("runs/eval", exist_ok=True)
with open("runs/eval/reached_screen.json", "w") as f:
    json.dump(out, f, indent=1)

nt = sum(r["transfer_reached"] for r in out)
na = sum(r["airl_reached"] for r in out)
nb = sum(r["transfer_reached"] and r["airl_reached"] for r in out)
print(f"\ndone in {time.time()-t0:.0f}s | n={len(out)} | "
      f"transfer reached {nt} ({nt/len(out):.1%}) | "
      f"airl reached {na} ({na/len(out):.1%}) | both {nb} ({nb/len(out):.1%})")
