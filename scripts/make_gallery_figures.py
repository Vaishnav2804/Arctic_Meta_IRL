"""Route-gallery figures without the PEMIRL generator decode.

Two showcase-pair selection modes (--select):

  tercile  (default) -- reproduces notebooks/02_route_gallery.ipynb exactly
                        (same support/query protocol, seed and length
                        terciles -> identical "Pair i" labels)
  reached            -- only pairs where BOTH transfer agents actually reached
                        the goal, read from runs/eval/reached_screen.json
                        (produced by scripts/screen_reached_pairs.py), still
                        spread over length terciles with distinct vessels

Each pair is decoded with:

  * MCE-IRL              -- pooled linear reward + finite-horizon soft VI
  * Transfer (PEMIRL)    -- fresh masked PPO on the frozen PEMIRL reward
                            f(s,a,z), conditioned on the vessel's support z
  * Transfer (AIRL)      -- same, on the frozen *context-free* reward f(s,a)
                            recovered by the PEMIRL_NOCTX ablation

The PEMIRL generator rollout ("pemirl decode") is deliberately absent.

Outputs (docs/aaai-paper-consolidated/figures/), with --suffix appended:
  fig_gallery_transfer.{pdf,png}   real + MCE + transfer(PEMIRL reward)
  fig_gallery_airl.{pdf,png}       real + MCE + transfer(AIRL reward)
  fig_gallery_combined.{pdf,png}   all four
  fig_gallery_metrics.json         per-pair Hausdorff / length / reached
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import arctic_meta_irl.data  # noqa: F401  (must precede env/features/algos)
import common

import numpy as np
import torch

from arctic_meta_irl.utils.config import load_config
from arctic_meta_irl.utils.seeding import set_seed
from arctic_meta_irl.algos.mce_irl import MCEIRL
from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.models.policy import MaskedPolicy
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, Task
from arctic_meta_irl.eval.rollout import (generate_route_policy,
                                          pemirl_support_sets,
                                          _episode_tensors)
from arctic_meta_irl.eval.metrics import route_metrics, route_length_km

warnings.filterwarnings("ignore", category=UserWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--select", choices=("tercile", "reached"), default="tercile",
                help="showcase pair selection (see module docstring)")
ap.add_argument("--suffix", default="", help="appended to output filenames")
ap.add_argument("--n_pairs", type=int, default=6)
ap.add_argument("--screen", default="runs/eval/reached_screen.json")
# reject degenerate "successes": a decode that jumps to the goal in 2 hops
# technically reaches it, but says nothing about route quality
ap.add_argument("--min_lr", type=float, default=0.7,
                help="reached mode: min decoded/real length ratio")
ap.add_argument("--max_lr", type=float, default=1.6,
                help="reached mode: max decoded/real length ratio")
ap.add_argument("--exclude", default="", help="comma-separated "
                "mmsi:year:month episodes to drop from the candidate pool")
ap.add_argument("--pin", default="", help="comma-separated mmsi:start:goal "
                "episodes to use as the showcase verbatim, bypassing the "
                "tercile picker (ordered by real route length)")
ap.add_argument("--compact", action="store_true", help="figure height matched "
                "to the panel aspect (removes the dead band between rows) "
                "plus smaller type and minimal title/legend gaps")
ap.add_argument("--aspect", type=float, default=1.6,
                help="panel width:height ratio (km-true)")
ap.add_argument("--width", type=float, default=17.0,
                help="figure width in inches. The default 17in is ~4x the "
                     "LaTeX \\textwidth, so every label is scaled down to "
                     "~0.4x on the page; pass 7.0 to render at print size")
ap.add_argument("--no_suptitle", action="store_true",
                help="drop the figure suptitle (it duplicates the caption)")
ap.add_argument("--short_labels", action="store_true",
                help="abbreviated panel title and a one-line Hausdorff strip "
                     "instead of a four-line box. Needed with --width 7: at "
                     "print size the long title and the stacked box are wider "
                     "than the panel they sit in")
args = ap.parse_args()
SUF = args.suffix
EXCLUDE = {tuple(int(x) for x in tok.split(":"))
           for tok in args.exclude.split(",") if tok.strip()}
PIN = [tuple(int(x) for x in tok.split(":"))
       for tok in args.pin.split(",") if tok.strip()]

DEVICE = "cpu"
OUT = "docs/aaai-paper-consolidated/figures"
cfg = load_config("configs/pemirl.yaml")
set_seed(int(cfg["seed"]))

# --------------------------------------------------------------------- load
t0 = time.time()
mdp, fb, eps = common.load_pipeline(cfg)
env = GCRLNavEnv(mdp, fb, max_horizon=int(cfg["mdp"]["max_horizon"]))

mce = MCEIRL(mdp, fb, seed=int(cfg["seed"]))
mce.load("models/mce_irl/theta.npz")

# PEMIRL is loaded only to infer z for the context-conditioned transfer agent
pem = PEMIRL(fb.dim, mdp.n_actions, cfg["pemirl"], device=DEVICE)
pem.load("models/pemirl/pemirl.pt")
pem.eval()

CTX = int(cfg["pemirl"]["context_dim"])
HID = tuple(cfg["pemirl"]["policy_hidden"])

transfer_pem = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=CTX, hidden=HID)
transfer_pem.load_state_dict(
    torch.load("models/ppo_on_pemirl/policy.pt", map_location=DEVICE))
transfer_pem.to(DEVICE).eval()

# AIRL ablation: context_dim = 0, so the policy takes phi(s) alone and z=None
transfer_airl = MaskedPolicy(fb.dim, mdp.n_actions, context_dim=0, hidden=HID)
transfer_airl.load_state_dict(
    torch.load("runs/ppo_on_noctx/policy.pt", map_location=DEVICE))
transfer_airl.to(DEVICE).eval()

print(f"pipeline + 3 checkpoints loaded in {time.time()-t0:.0f}s | "
      f"S={mdp.n_states}, A={mdp.n_actions}, D={fb.dim}", flush=True)

# ------------------------------------------------- showcase pair selection
# identical to notebooks/02_route_gallery.ipynb
SUPPORT_SIZE = int(cfg["pemirl"]["support_set_size"])
support_map = pemirl_support_sets(eps["test"], SUPPORT_SIZE,
                                  seed=int(cfg["seed"]))
cands = []
for mmsi, (sup, qry) in support_map.items():
    for ep in qry:
        if len(ep) < 8:
            continue
        cands.append((ep, route_length_km(mdp, list(ep.states))))
cands.sort(key=lambda t: t[1])
n_all = len(cands)

if EXCLUDE:
    before = len(cands)
    cands = [(ep, km) for ep, km in cands
             if (int(ep.mmsi), int(ep.year), int(ep.month)) not in EXCLUDE]
    print(f"excluded {before - len(cands)} episode(s) by --exclude", flush=True)

if args.select == "reached":
    # keep only pairs whose BOTH transfer decodes reached the goal
    with open(args.screen) as f:
        screen = json.load(f)
    ok = {(r["mmsi"], r["start"], r["goal"], r["year"], r["month"])
          for r in screen
          if r["transfer_reached"] and r["airl_reached"]
          and min(r["transfer_length_ratio"],
                  r["airl_length_ratio"]) >= args.min_lr
          and max(r["transfer_length_ratio"],
                  r["airl_length_ratio"]) <= args.max_lr}
    cands = [(ep, km) for ep, km in cands
             if (int(ep.mmsi), int(ep.states[0]), int(ep.goal),
                 int(ep.year), int(ep.month)) in ok]
    print(f"{len(cands)}/{n_all} candidates reached the goal under BOTH "
          "transfer agents", flush=True)
    if len(cands) < args.n_pairs:
        raise SystemExit(f"only {len(cands)} goal-reaching pairs available")

if PIN:
    by_key = {(int(ep.mmsi), int(ep.states[0]), int(ep.goal)): (ep, km)
              for ep, km in cands}
    missing = [p for p in PIN if p not in by_key]
    if missing:
        raise SystemExit(f"--pin episodes not in the candidate pool: {missing}")
    showcase = sorted((by_key[p] for p in PIN), key=lambda t: t[1])
    print(f"pinned {len(showcase)} showcase pairs", flush=True)

else:
    # two per length tercile: the median-length episode, then the episode whose
    # start is farthest from it (distinct vessels, geographic spread)
    per_bucket = max(1, args.n_pairs // 3)
    n = len(cands)
    buckets = [cands[:n // 3], cands[n // 3: 2 * n // 3], cands[2 * n // 3:]]
    showcase, used_mmsi = [], set()
    for bucket in buckets:
        pool = [(ep, km) for ep, km in bucket if ep.mmsi not in used_mmsi]
        if not pool:
            continue
        first = pool[len(pool) // 2]
        used_mmsi.add(first[0].mmsi)
        picked = [first]
        la0, lo0 = mdp.latlng[int(first[0].states[0])]
        while len(picked) < per_bucket:
            second, best_d = None, -1.0
            for ep, km in pool:
                if ep.mmsi in used_mmsi:
                    continue
                la, lo = mdp.latlng[int(ep.states[0])]
                d = (la - la0) ** 2 + (lo - lo0) ** 2
                if d > best_d:
                    second, best_d = (ep, km), d
            if second is None:
                break
            used_mmsi.add(second[0].mmsi)
            picked.append(second)
        showcase += picked
    showcase.sort(key=lambda t: t[1])
    print(f"{n} candidates -> {len(showcase)} showcase pairs", flush=True)

# ------------------------------------------------------------------ decode
METHODS = ["mce_irl", "transfer", "airl"]
routes, rows = [], []
for i, (ep, km) in enumerate(showcase):
    task = Task(start=int(ep.states[0]), goal=int(ep.goal), year=ep.year,
                month=ep.month, mmsi=ep.mmsi, category=ep.category)
    sup = support_map[ep.mmsi][0]
    z = pem.infer_context_support(
        [_episode_tensors(e, fb) for e in sup], fixed=True)

    t1 = time.time()
    decoded = {
        "real": list(ep.states),
        "mce_irl": mce.greedy_route(task.start, task.goal, ep.year, ep.month,
                                    ep.mmsi, horizon=env.max_horizon),
        "transfer": generate_route_policy(transfer_pem, env, task, z=z,
                                          deterministic=True, device=DEVICE),
        "airl": generate_route_policy(transfer_airl, env, task, z=None,
                                      deterministic=True, device=DEVICE),
    }
    routes.append({"ep": ep, "real_km": km, **decoded})
    for m in METHODS:
        met = route_metrics(mdp, decoded[m], decoded["real"])
        rows.append({"pair": i, "method": m, "real_km": round(km),
                     "mmsi": int(ep.mmsi), "category": ep.category,
                     "year": int(ep.year), "month": int(ep.month),
                     "decoded_steps": len(decoded[m]), **met})
    print(f"pair {i}: {time.time()-t1:5.1f}s | " + " | ".join(
        f"{m} H={[r for r in rows if r['pair'] == i and r['method'] == m][0]['hausdorff_km']:7.1f}"
        for m in METHODS), flush=True)

with open(f"{OUT}/fig_gallery_metrics{SUF}.json", "w") as f:
    json.dump(rows, f, indent=1, default=float)

# ------------------------------------------------------------------ plotting
import matplotlib as mpl

if not hasattr(mpl.RcParams, "_get"):          # cartopy 0.24 / matplotlib 3.6
    mpl.RcParams._get = dict.__getitem__

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from cartopy.io import shapereader
    shapereader.natural_earth(resolution="50m", category="physical",
                              name="land")
    shapereader.natural_earth(resolution="50m", category="physical",
                              name="coastline")
    NE_OK = True
except Exception as e:
    NE_OK = False
    print(f"Natural Earth unavailable ({e!r}); water-cell scatter only")

PC = ccrs.PlateCarree()
cell_lat, cell_lon = mdp.latlng[:, 0], mdp.latlng[:, 1]
STYLE = {
    "real":     dict(color="black",      lw=2.4, zorder=6, label="Real (AIS)"),
    "mce_irl":  dict(color="tab:blue",   lw=1.5, zorder=5, label="MCE-IRL"),
    "transfer": dict(color="tab:green",  lw=1.6, zorder=5, alpha=0.95,
                     label="Transfer (PEMIRL reward)"),
    # dashed: the two transfer agents often decode the identical route, and a
    # solid orange line would hide the green one underneath it
    "airl":     dict(color="tab:orange", lw=1.6, zorder=5, alpha=0.95, ls="--",
                     label="AIRL transfer (no context)"),
}
ABBR = {"mce_irl": "MCE", "transfer": "TRF", "airl": "AIRL"}


def panel_extent(r, methods, pad=0.30, cap_deg=14.0,
                 aspect=None):
    aspect = args.aspect if aspect is None else aspect
    """Extent anchored on the real route; decodes included up to a cap."""
    R = mdp.latlng[np.asarray(r["real"], int)]
    c_la, c_lo = R[:, 0].mean(), R[:, 1].mean()
    A = mdp.latlng[np.asarray(sum((r[m] for m in methods), r["real"]), int)]
    lo0 = max(A[:, 1].min(), c_lo - cap_deg); lo1 = min(A[:, 1].max(), c_lo + cap_deg)
    la0 = max(A[:, 0].min(), c_la - cap_deg / 2); la1 = min(A[:, 0].max(), c_la + cap_deg / 2)
    lo0 = min(lo0, R[:, 1].min()); lo1 = max(lo1, R[:, 1].max())
    la0 = min(la0, R[:, 0].min()); la1 = max(la1, R[:, 0].max())
    dlo = max(lo1 - lo0, 1.5) * (1 + 2 * pad)
    dla = max(la1 - la0, 0.8) * (1 + 2 * pad)
    coslat = np.cos(np.radians((la0 + la1) / 2))
    if dlo * coslat / dla < aspect:
        dlo = aspect * dla / coslat
    else:
        dla = dlo * coslat / aspect
    c_lo, c_la = (lo0 + lo1) / 2, (la0 + la1) / 2
    return [c_lo - dlo / 2, c_lo + dlo / 2, c_la - dla / 2, c_la + dla / 2]


NCOL = 3 if len(routes) > 2 else len(routes)
NROW = int(np.ceil(len(routes) / NCOL))


# A cartopy GeoAxes keeps its projected aspect and centres the map inside
# whatever box it is given, so a figure taller than NROW * (W/NCOL/aspect)
# just pads every row with dead space. Size the figure to the maps instead.
W = args.width
if args.compact:
    TITLE_H, SUP_H, LEG_H = 0.34, 0.24, 0.34
    FS_TITLE, FS_BOX, FS_LEG, FS_SUP = 9.0, 7.0, 10.5, 10.5
    MS_START, MS_GOAL = 5.5, 12.0
    PAD, WPAD, HPAD, TPAD = 0.10, 0.15, 0.90, 3.0
else:
    TITLE_H, SUP_H, LEG_H = 1.30, 0.40, 0.37
    FS_TITLE, FS_BOX, FS_LEG, FS_SUP = 10.5, 8.5, 9.5, 13.0
    MS_START, MS_GOAL = 8.0, 17.0
    PAD, WPAD, HPAD, TPAD = 1.08, 1.08, 1.08, 6.0
if args.no_suptitle:
    SUP_H = 0.0
if args.short_labels:
    FS_TITLE, FS_BOX, FS_LEG = 8.0, 6.0, 7.5
    STYLE["transfer"]["label"] = "PEMIRL transfer"
    STYLE["airl"]["label"] = "AIRL transfer"
PANEL_H = W / NCOL / args.aspect
FIG_H = NROW * (PANEL_H + TITLE_H) + SUP_H + LEG_H


def make_gallery(methods, stem, suptitle):
    fig = plt.figure(figsize=(W, FIG_H))
    for i, r in enumerate(routes):
        ext = panel_extent(r, methods)
        proj = ccrs.Stereographic(central_longitude=(ext[0] + ext[1]) / 2,
                                  central_latitude=(ext[2] + ext[3]) / 2)
        ax = fig.add_subplot(NROW, NCOL, i + 1, projection=proj)
        ax.set_extent(ext, crs=PC)
        if NE_OK:
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ece7da",
                           edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5,
                           color="0.45", zorder=1)
        ax.scatter(cell_lon, cell_lat, s=0.5, color="#c8dff0", transform=PC,
                   zorder=1.5, rasterized=True)
        ax.gridlines(lw=0.3, alpha=0.35, draw_labels=False)

        for m in [*methods, "real"]:              # real drawn on top
            P = mdp.latlng[np.asarray(r[m], int)]
            ax.plot(P[:, 1], P[:, 0], transform=PC, **STYLE[m])
        s_la, s_lo = mdp.latlng[int(r["real"][0])]
        g_la, g_lo = mdp.latlng[int(r["real"][-1])]
        ax.plot(s_lo, s_la, "o", color="black", ms=MS_START, mec="white",
                mew=1.0, transform=PC, zorder=8)
        ax.plot(g_lo, g_la, "*", color="gold", ms=MS_GOAL, mec="black",
                mew=0.9, transform=PC, zorder=8)

        dm = {row["method"]: row for row in rows if row["pair"] == i}
        ep = r["ep"]
        if args.short_labels:
            txt = "H km  " + "  ".join(
                f"{ABBR[m]} {dm[m]['hausdorff_km']:.0f}"
                + ("*" if dm[m]["reached_goal"] else "") for m in methods)
            title = (f"P{i} \u00b7 {ep.category} {ep.year}-{ep.month:02d} "
                     f"\u00b7 {r['real_km']:.0f} km")
        else:
            txt = "Hausdorff vs real:\n" + "\n".join(
                f"{ABBR[m]:<4s} {dm[m]['hausdorff_km']:6.1f} km"
                + ("*" if dm[m]["reached_goal"] else "") for m in methods)
            title = (f"Pair {i} \u2014 {ep.category}, {ep.year}-{ep.month:02d}, "
                     f"real route {r['real_km']:.0f} km")
        ax.text(0.015, 0.015, txt, transform=ax.transAxes, fontsize=FS_BOX,
                family="monospace", va="bottom", ha="left", linespacing=1.05,
                bbox=dict(fc="white", alpha=0.85, ec="0.6", lw=0.5, pad=1.5),
                zorder=9)
        ax.set_title(title, fontsize=FS_TITLE, pad=TPAD)

    handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[m].items()
                                     if k in ("color", "lw", "alpha", "ls",
                                              "label")})
               for m in ["real", *methods]]
    handles += [plt.Line2D([], [], marker="o", color="black", ls="",
                           ms=MS_START, mec="white", label="Start"),
                plt.Line2D([], [], marker="*", color="gold", ls="",
                           ms=MS_GOAL * 0.8, mec="black", label="Goal"),
                plt.Line2D([], [], ls="", label="* = reached goal")]
    fig.legend(handles=handles, loc="upper center", ncol=min(7, len(handles)),
               fontsize=FS_LEG, frameon=False, handlelength=1.6,
               columnspacing=1.3, handletextpad=0.5,
               bbox_to_anchor=(0.5, LEG_H / FIG_H))
    if not args.no_suptitle:
        fig.suptitle(suptitle, fontsize=FS_SUP, va="bottom",
                     y=1.0 - SUP_H / FIG_H + 0.004)
    fig.tight_layout(rect=(0, LEG_H / FIG_H, 1, 1 - SUP_H / FIG_H),
                     pad=PAD, w_pad=WPAD, h_pad=HPAD)
    for ext_ in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}{SUF}.{ext_}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{stem}{SUF}.pdf", flush=True)


BASE = "Decoded routes vs. real AIS — six held-out test O–D pairs"
make_gallery(["mce_irl", "transfer"], "fig_gallery_transfer",
             BASE + ": transfer on frozen PEMIRL reward")
make_gallery(["mce_irl", "airl"], "fig_gallery_airl",
             BASE + ": transfer on frozen AIRL reward")
make_gallery(["mce_irl", "transfer", "airl"], "fig_gallery_combined",
             BASE + ": PEMIRL vs. AIRL reward transfer")
print("done")
