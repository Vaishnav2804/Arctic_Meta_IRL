"""Results figures for the AAAI paper (Sec. Results).

    python scripts/make_results_figures.py [--outdir docs/aaai-paper-consolidated/figures]

Every number is read from a repository artifact; the only hard-coded values are
the F-ratio decomposition and the context probes, which come from the descriptor
/ probe scripts and are recorded in the supplement (Tab. 3, App. D).

Writes fig_ladder, fig_criteria, fig_transfer, fig_audit, fig_synthetic as .pdf
(+ .png preview).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

# white halo so labels stay readable where series lines cross under them
HALO = [patheffects.withStroke(linewidth=2.4, foreground="white")]

ROOT = Path(__file__).resolve().parents[1]

# Colorblind-safe categorical order (validated: lightness band, chroma floor,
# CVD separation, normal-vision floor, contrast >= 3:1 on a light surface).
C = {
    "mce": "#0072B2",       # linear, shared
    "airl": "#D55E00",      # nonlinear, shared
    "pemirl": "#009E73",    # nonlinear, per-vessel
    "canon": "#8C5AA8",     # canonical AIRL
    "orig": "#5A5A5A",      # original authors' implementation
}
LABEL = {
    "mce": "MCE-IRL",
    "airl": "AIRL",
    "pemirl": "PEMIRL",
    "canon": "Canonical AIRL",
    "orig": "Original meta-IRL",
}
CHANCE = -1.896            # empirical masked-action chance floor (test split)
RANDOM_REACH = 0.089       # uniform-random valid policy, steady-state goal-reach

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.color": "#D8D8D8",
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------
def jload(rel: str):
    return json.loads((ROOT / rel).read_text())


def per_vessel_ll(seed: int, split: str = "test"):
    """Per-vessel LL/decision for the three ladder models on matched decisions.

    runs/eval/per_episode_ll*.json store per-episode totals; the *_noctx file
    holds AIRL under the same `pemirl_ll_*` key (context pathway disabled).
    Episodes missing any model's entry are support episodes and are dropped for
    all models alike, which is the matched-query-decision set.
    """
    sfx = "" if seed == 0 else f"_s{seed}"
    ctx = jload(f"runs/eval/per_episode_ll{sfx}.json")[split]
    noctx = jload(f"runs/eval/per_episode_ll_noctx{sfx}.json")[split]
    key = "pemirl_ll_random_support"

    tot = defaultdict(lambda: {"n": 0.0, "mce": 0.0, "airl": 0.0, "pemirl": 0.0})
    for eid, rec in ctx.items():
        other = noctx.get(eid)
        if other is None or key not in rec or key not in other:
            continue
        v = tot[rec["mmsi"]]
        v["n"] += rec["n_decisions"]
        v["mce"] += rec["mce_ll"]
        v["pemirl"] += rec[key]
        v["airl"] += other[key]
    return {m: {k: v[k] / v["n"] for k in ("mce", "airl", "pemirl")}
            for m, v in tot.items() if v["n"] > 0}


def transfer_curve(paths):
    """Stack goal-reach histories across seeds -> (iters, per-seed array)."""
    curves = []
    for p in paths:
        f = ROOT / p / "history.json"
        if f.exists():
            curves.append([h["reached"] for h in json.loads(f.read_text())])
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    return np.arange(n), np.array([c[:n] for c in curves])


def smooth(y, w=9):
    k = np.ones(w) / w
    pad = np.r_[np.repeat(y[:1], w // 2), y, np.repeat(y[-1:], w // 2)]
    return np.convolve(pad, k, mode="valid")


# --------------------------------------------------------------------------
# Figure 1 — the controlled ladder
# --------------------------------------------------------------------------
def fig_ladder(outdir: Path):
    ms = jload("runs/multiseed/summary.json")
    sig = jload("runs/eval/significance.json")["test"]["random_support"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                 gridspec_kw={"width_ratios": [1.0, 1.35]})

    # (a) macro likelihood, per-seed points
    order = ["mce", "airl", "pemirl"]
    keys = {"mce": "LL.test.mce_macro", "airl": "LL.test.airl_macro",
            "pemirl": "LL.test.pemirl_macro"}
    xs = np.arange(3)
    for i, m in enumerate(order):
        e = ms[keys[m]]
        seeds = list(e["by_seed"].values())
        ax.bar(i, e["mean"] - CHANCE, bottom=CHANCE, width=0.52,
               color=C[m], alpha=.85, edgecolor="white", linewidth=1.0, zorder=2)
        ax.errorbar(i, e["mean"], yerr=e["std"], color="#2B2B2B",
                    capsize=2.5, elinewidth=0.9, zorder=4, fmt="none")
        ax.scatter(np.full(len(seeds), i) + np.linspace(-.09, .09, len(seeds)),
                   seeds, s=9, color="#2B2B2B", zorder=5, linewidths=0)
        ax.text(i, max(e["mean"] + e["std"], max(seeds)) + 0.022,
                f"{e['mean']:.3f}", ha="center",
                va="bottom", fontsize=7, color="#2B2B2B")

    ax.axhline(CHANCE, color="#B00020", ls=(0, (3, 2)), lw=0.9, zorder=3)
    ax.text(2.45, CHANCE, "chance", color="#B00020", fontsize=6.5,
            ha="right", va="bottom")

    # capacity / context brackets
    def bracket(x0, x1, y, text, color):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, y + 0.018, text, ha="center", va="bottom",
                fontsize=7, color=color)

    bracket(0.3, 0.7, -1.34, r"$\Delta$capacity" "\n" r"$+50.9\%$", C["airl"])
    bracket(1.3, 1.7, -1.34, r"$\Delta$context" "\n" r"$-16.5\%$", C["pemirl"])

    ax.set_xticks(xs)
    ax.set_xticklabels(["MCE-IRL\nlinear, shared", "AIRL\nnonlin., shared",
                        "PEMIRL\nnonlin., per-vessel"], fontsize=6.8)
    ax.set_ylim(-1.95, -1.18)
    ax.set_ylabel("held-out LL / decision (macro)")
    ax.set_title("(a) Capacity helps, context hurts", loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # (b) paired per-vessel deltas, averaged over seeds
    pv = [per_vessel_ll(s) for s in (0, 1, 2)]
    vessels = sorted(set.intersection(*[set(p) for p in pv]))
    cap = np.array([np.mean([p[v]["airl"] - p[v]["mce"] for p in pv]) for v in vessels])
    ctxd = np.array([np.mean([p[v]["pemirl"] - p[v]["airl"] for p in pv]) for v in vessels])
    o = np.argsort(cap)
    idx = np.arange(len(vessels))

    bx.axhline(0, color="#8A8A8A", lw=0.8, zorder=1)
    bx.bar(idx - 0.21, cap[o], width=0.4, color=C["airl"], alpha=.9,
           linewidth=0, zorder=2, label=r"$\Delta$capacity (AIRL $-$ MCE-IRL)")
    bx.bar(idx + 0.21, ctxd[o], width=0.4, color=C["pemirl"], alpha=.9,
           linewidth=0, zorder=2, label=r"$\Delta$context (PEMIRL $-$ AIRL)")

    s0 = sig["per_seed"]["seed_0"]
    bx.text(0.015, 0.94,
            f"capacity ahead on {int(s0['capacity (NC-MCE)']['model_wins'])}/41 vessels "
            f"($p=1.1\\times10^{{-6}}$)",
            transform=bx.transAxes, fontsize=6.8, color=C["airl"], va="top")
    bx.text(0.015, 0.855,
            f"context ahead on {int(s0['context (CTX-NC)']['model_wins'])}/41 vessels "
            f"($p=6.9\\times10^{{-9}}$)",
            transform=bx.transAxes, fontsize=6.8, color=C["pemirl"], va="top")

    # one vessel's capacity gain is far off-scale; clip and label it so the
    # remaining 40 stay legible
    top = 1.45
    for j, val in enumerate(cap[o]):
        if val > top:
            bx.annotate(f"+{val:.1f}", xy=(j - 0.21, top), xytext=(j - 2.0, top - 0.28),
                        fontsize=6.5, color=C["airl"], ha="right",
                        arrowprops=dict(arrowstyle="-|>", color=C["airl"], lw=0.8))
    bx.set_ylim(-0.72, top)
    bx.set_xlim(-1, len(vessels))
    bx.set_xlabel("held-out vessels, sorted by capacity gain")
    bx.set_ylabel("$\\Delta$ LL / decision (nats)")
    bx.set_title("(b) Same two steps, vessel by vessel", loc="left")
    bx.set_xticks([])
    bx.yaxis.grid(True, zorder=0)
    bx.set_axisbelow(True)
    bx.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False,
              handlelength=1.1)

    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_ladder")


# --------------------------------------------------------------------------
# Figure 2 — the three criteria disagree
# --------------------------------------------------------------------------
def fig_criteria(outdir: Path):
    ms = jload("runs/multiseed/summary.json")
    canon = [jload(f"runs/eval/robustness_canonical_airl{s}.json")
             ["test"]["random_support"]["pemirl_macro"] for s in ("", "_s1", "_s2")]
    orig = jload("experiments/original_metairl/runs/eval/results_original_metairl.json")

    def xfer(dirs):
        vals = []
        for d in dirs:
            f = ROOT / d / "goal_reach_summary.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["steady"])
        return float(np.mean(vals)) if vals else np.nan

    models = {
        "mce": dict(
            ll=ms["LL.test.mce_macro"]["mean"],
            hd=ms["ROUTE.mce_irl.hausdorff_km"]["mean"],
            xf=ms["XFER.mce.steady"]["mean"]),
        "airl": dict(
            ll=ms["LL.test.airl_macro"]["mean"],
            hd=ms["ROUTE.airl.hausdorff_km"]["mean"],
            xf=ms["XFER.airl.steady"]["mean"]),
        "pemirl": dict(
            ll=ms["LL.test.pemirl_macro"]["mean"],
            hd=ms["ROUTE.pemirl.hausdorff_km"]["mean"],
            xf=ms["XFER.pemirl.steady"]["mean"]),
        "canon": dict(
            ll=float(np.mean(canon)),
            hd=float(np.mean([jload(f"runs/canonical_airl{sfx}/eval/results.json")
                              ["test"]["canonical_airl_hausdorff_km"]
                              for sfx in ("", "_s1", "_s2")])),
            xf=xfer([f"runs/ppo_on_canonical_airl_g{s}" for s in ("", "_s1", "_s2")])),
        "orig": dict(
            ll=float(np.mean([orig["test"][k] for k in (
                "orig_metairl_ll_macro", "orig_metairl_s1_ll_macro",
                "orig_metairl_s2_ll_macro")])),
            hd=float(np.mean([orig["test"][k] for k in (
                "orig_metairl_hausdorff_km", "orig_metairl_s1_hausdorff_km",
                "orig_metairl_s2_hausdorff_km")])),
            xf=xfer([f"experiments/original_metairl/runs/ppo_on_orig_metairl{s}"
                     for s in ("", "_s1", "_s2")])),
    }

    # rank-flip (bump) chart: rank 1 = best under that criterion
    panels = [("ll", "held-out likelihood\n(LL / decision, macro)", False, "{:.2f}"),
              ("hd", "decoded-route fidelity\n(Hausdorff, km)", True, "{:.0f}"),
              ("xf", "reward transfer\n(goal-reach of a fresh agent)", False, "{:.2f}")]

    ranks, values = {}, {}
    for j, (k, _, invert, _f) in enumerate(panels):
        items = [(m, v[k]) for m, v in models.items() if np.isfinite(v[k])]
        items.sort(key=lambda t: t[1], reverse=not invert)
        for r, (m, val) in enumerate(items, start=1):
            ranks.setdefault(m, {})[j] = r
            values.setdefault(m, {})[j] = val

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for m, rr in ranks.items():
        js = sorted(rr)
        # break the line where a criterion was not computed for this model
        segs = [[js[0]]]
        for prev, cur in zip(js, js[1:]):
            if cur == prev + 1:
                segs[-1].append(cur)
            else:
                segs.append([cur])
        for a_, b_ in zip(segs, segs[1:]):        # criterion not computed
            ax.plot([a_[-1], b_[0]], [rr[a_[-1]], rr[b_[0]]], color=C[m],
                    lw=1.0, ls=(0, (2, 2)), alpha=.45, zorder=2)
        for seg in segs:
            ax.plot(seg, [rr[j] for j in seg], color=C[m], lw=2.2,
                    marker="o", ms=8, mec="white", mew=1.1, zorder=3,
                    solid_capstyle="round")
        for j in js:
            # rank-1 points put their value above, leaving the row below free
            # for the label of a model entering the chart at that criterion
            ax.annotate(panels[j][3].format(values[m][j]), (j, rr[j]),
                        xytext=(0, 8 if rr[j] == 1 else -11),
                        textcoords="offset points", va="bottom" if rr[j] == 1 else "top",
                        ha="center", fontsize=7.4, color="#2B2B2B", zorder=4,
                        path_effects=HALO)
        j0, j1 = js[0], js[-1]
        if j0 == 0:
            ax.text(-0.09, rr[0], LABEL[m], color=C[m], fontsize=8.2,
                    ha="right", va="center", fontweight="bold")
        else:                                     # label above the first point
            ax.annotate(LABEL[m], (j0, rr[j0]), xytext=(0, 12),
                        textcoords="offset points", color=C[m], fontsize=8.2,
                        ha="center", va="bottom", fontweight="bold", zorder=5,
                        path_effects=HALO)
        if j1 == 2:                               # mirror label at the exit rank
            ax.text(2.09, rr[j1], LABEL[m], color=C[m], fontsize=8.2,
                    ha="left", va="center", fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels([p[1] for p in panels], fontsize=8.2)
    ax.set_xlim(-1.15, 3.30)
    ax.set_ylim(5.8, 0.4)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"{r}" for r in range(1, 6)], fontsize=8.2)
    ax.set_ylabel("rank (1 = best)", fontsize=8.5)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("The three criteria rank the recovered rewards differently",
                 loc="left", fontsize=9.5)
    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_criteria")


# --------------------------------------------------------------------------
# Figure — model nesting: AIRL is PEMIRL minus the context pathway;
# canonical AIRL structures the discriminator instead
# --------------------------------------------------------------------------
def fig_models(outdir: Path):
    from matplotlib.patches import FancyBboxPatch

    SHARED_FC, SHARED_EC = "#F2F2F2", "#8A8A8A"      # machinery both arms share
    CTX_FC, CTX_EC = "#E4F3EE", C["pemirl"]          # the context pathway
    CAN_FC, CAN_EC = "#F1EAF7", C["canon"]           # canonical structure
    INK, MUTED = "#2B2B2B", "#9A9A9A"

    def box(ax, x, y, w, h, text, fc, ec, ls="-", tc=INK, fs=6.6, lw=1.1):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.16",
            fc=fc, ec=ec, ls=ls, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)

    def arrow(ax, p0, p1, color="#6B6B6B", ls="-", lw=1.1, label=None,
              loff=(0, 0), lc=None, fs=5.8):
        ax.annotate("", p1, p0, zorder=1, arrowprops=dict(
            arrowstyle="-|>", color=color, ls=ls, lw=lw,
            shrinkA=2.5, shrinkB=2.5, mutation_scale=9))
        if label:
            mx, my = (p0[0] + p1[0]) / 2 + loff[0], (p0[1] + p1[1]) / 2 + loff[1]
            ax.text(mx, my, label, fontsize=fs, color=lc or color,
                    ha="center", va="center", zorder=4, path_effects=HALO)

    fig, ax = plt.subplots(figsize=(7.2, 2.85))
    ax.set_xlim(0, 21.9)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    def adversarial_pair(X, f_text, p_text):
        """The shared discriminator/generator loop, identical in (a) and (b)."""
        box(ax, X + 4.0, 4.6, 2.6, 1.3, f_text, SHARED_FC, SHARED_EC)
        box(ax, X + 4.0, 1.5, 2.6, 1.3, p_text, SHARED_FC, SHARED_EC)
        arrow(ax, (X + 4.7, 4.5), (X + 4.7, 2.95), label="reward",
              loff=(-0.62, 0))
        arrow(ax, (X + 5.9, 2.95), (X + 5.9, 4.5), label="rollouts",
              loff=(0.72, 0))

    # ---- (a) PEMIRL: full model with the latent-context pathway ----------
    X = 0.0
    ax.text(X + 3.5, 8.15, "(a) PEMIRL: latent context", color=C["pemirl"],
            fontsize=8.0, fontweight="bold", ha="center")
    box(ax, X + 0.3, 6.0, 3.0, 1.15, "support set\n$\\tau_1\\ldots\\tau_k$ per vessel",
        CTX_FC, CTX_EC)
    box(ax, X + 0.3, 3.9, 3.0, 1.15, "posterior $q(z\\,|\\,\\tau)$\n(bi-LSTM)",
        CTX_FC, CTX_EC)
    box(ax, X + 1.15, 1.9, 1.3, 0.95, "$z\\in\\mathbb{R}^{8}$", CTX_FC, CTX_EC)
    arrow(ax, (X + 1.8, 5.9), (X + 1.8, 5.15), color=C["pemirl"])
    arrow(ax, (X + 1.8, 3.8), (X + 1.8, 2.95), color=C["pemirl"])
    arrow(ax, (X + 2.55, 2.6), (X + 3.9, 4.9), color=C["pemirl"],
          label="info-max", loff=(0.5, -0.4), fs=5.6)
    arrow(ax, (X + 2.55, 2.25), (X + 3.9, 2.1), color=C["pemirl"])
    adversarial_pair(X, "reward $f(s,a,z)$\n(discriminator)",
                     "policy $\\pi(a\\,|\\,s,z)$\n(masked PPO)")

    # ---- (b) AIRL: the single-factor ablation of (a) ----------------------
    X = 7.35
    ax.text(X + 3.5, 8.15, "(b) AIRL: context pathway removed",
            color=C["airl"], fontsize=8.0, fontweight="bold", ha="center")
    for (bx, by, bw, bh) in [(0.3, 6.0, 3.0, 1.15), (0.3, 3.9, 3.0, 1.15),
                             (1.15, 1.9, 1.3, 0.95)]:
        box(ax, X + bx, by, bw, bh, "", "white", "#C4C4C4", ls=(0, (3, 3)),
            lw=0.9)
    ax.text(X + 1.8, 4.5, "ablated", color=C["airl"], fontsize=6.6,
            fontstyle="italic", ha="center", va="center", rotation=18,
            path_effects=HALO, zorder=4)
    adversarial_pair(X, "reward $f(s,a)$\n(discriminator)",
                     "policy $\\pi(a\\,|\\,s)$\n(masked PPO)")
    ax.text(X + 3.5, 0.55, "identical nets, optimizer, budget as (a)",
            color=MUTED, fontsize=6.0, ha="center", fontstyle="italic")

    # ---- (c) canonical AIRL: structured discriminator ---------------------
    X = 14.7
    ax.text(X + 3.5, 8.15, "(c) Canonical AIRL: structured $f$",
            color=C["canon"], fontsize=8.0, fontweight="bold", ha="center")
    box(ax, X + 0.5, 5.6, 6.0, 1.3,
        "$f(s,s') = g(s) + \\gamma\\,h(s') - h(s)$\n"
        "$D = e^{f}/(e^{f} + \\pi(a\\,|\\,s))$", CAN_FC, CAN_EC, fs=6.8)
    box(ax, X + 0.5, 3.4, 2.55, 1.25, "$g(s)$: state-only\nreward (transfers)",
        CAN_FC, CAN_EC)
    box(ax, X + 3.95, 3.4, 2.55, 1.25, "$h(s)$: shaping\n($h(\\mathrm{goal})=0$)",
        CAN_FC, CAN_EC)
    box(ax, X + 2.1, 1.35, 2.8, 1.2, "policy $\\pi(a\\,|\\,s)$\n(masked PPO)",
        SHARED_FC, SHARED_EC)
    arrow(ax, (X + 1.8, 4.75), (X + 1.8, 5.5), color=C["canon"])
    arrow(ax, (X + 5.2, 4.75), (X + 5.2, 5.5), color=C["canon"])
    arrow(ax, (X + 3.3, 2.65), (X + 3.3, 5.5), label="rollouts",
          loff=(-0.85, -1.15))
    arrow(ax, (X + 3.7, 5.5), (X + 3.7, 2.65), label="$r=f-\\log\\pi$",
          loff=(1.35, -1.15))

    fig.tight_layout(pad=0.3)
    save(fig, outdir, "fig_models")


# --------------------------------------------------------------------------
# Figure 3 — reward transfer
# --------------------------------------------------------------------------
def fig_transfer(outdir: Path):
    groups = {
        "mce": [f"runs/ppo_on_mce{s}" for s in ("", "_s1", "_s2")],
        "airl": [f"runs/ppo_on_noctx{s}" for s in ("", "_s1", "_s2")],
        "pemirl": [f"runs/ppo_on_pemirl{s}" for s in ("", "_s1", "_s2")],
        "canon": [f"runs/ppo_on_canonical_airl_g{s}" for s in ("", "_s1", "_s2")],
        "orig": [f"experiments/original_metairl/runs/ppo_on_orig_metairl{s}"
                 for s in ("", "_s1", "_s2")],
    }
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax.axhline(RANDOM_REACH, color="#B00020", ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax.text(2, RANDOM_REACH + 0.012, "uniform-random valid policy",
            color="#B00020", fontsize=6.5, va="bottom")

    ends = {}
    for m, dirs in groups.items():
        it, cur = transfer_curve(dirs)
        if it is None:
            continue
        mu = smooth(cur.mean(0))
        ax.fill_between(it, smooth(cur.min(0)), smooth(cur.max(0)),
                        color=C[m], alpha=.16, linewidth=0, zorder=3)
        ax.plot(it, mu, color=C[m], zorder=4)
        ends[m] = (it[-1], mu[-1], cur)

    for m, (x, y, _) in ends.items():
        dy = {"airl": 0.03, "pemirl": -0.035}.get(m, 0.0)
        ax.text(x + 3, y + dy, LABEL[m], color=C[m], fontsize=6.8, va="center")

    mce_mu = smooth(ends["mce"][2].mean(0))
    pk = int(np.argmax(mce_mu))
    ax.annotate("peak, then decay:\nloitering beats arriving",
                xy=(pk + 4, mce_mu[pk]), xytext=(46, 0.47),
                fontsize=6.8, color=C["mce"], ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["mce"], lw=0.8,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_xlim(0, 235)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("PPO iteration (fresh agent, frozen reward)")
    ax.set_ylabel("goal-reach rate")
    ax.set_title("(a) What each recovered reward teaches a new agent", loc="left")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # (b) seed spread of the steady-state value
    order = [m for m in ["mce", "pemirl", "airl", "canon", "orig"] if m in ends]
    for i, m in enumerate(order):
        # released steady-state summary where it exists (one seed has a summary
        # but no history), else the mean of the last 32 iterations
        steady = []
        for p in groups[m]:
            f = ROOT / p / "goal_reach_summary.json"
            h = ROOT / p / "history.json"
            if f.exists():
                steady.append(json.loads(f.read_text())["steady"])
            elif h.exists():
                y = [r["reached"] for r in json.loads(h.read_text())]
                steady.append(float(np.mean(y[-32:])))
        steady = np.array(steady)
        bx.scatter(steady, np.full(len(steady), i), s=22, color=C[m],
                   zorder=3, linewidths=0)
        bx.plot([steady.min(), steady.max()], [i, i], color=C[m], lw=1.2,
                alpha=.5, zorder=2)
    bx.axvline(RANDOM_REACH, color="#B00020", ls=(0, (3, 2)), lw=0.9)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([LABEL[m] for m in order], fontsize=7)
    bx.set_xlim(0, 1.0)
    bx.set_xlabel("steady-state goal-reach, one dot per seed")
    bx.set_title("(b) Bounded rewards transfer tightly", loc="left")
    bx.xaxis.grid(True)
    bx.set_axisbelow(True)
    bx.text(0.97, 0.90, "unclamped reward:\nseed lottery", transform=bx.transAxes,
            ha="right", va="top", fontsize=6.8, color=C["orig"])

    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_transfer")


# --------------------------------------------------------------------------
# Figure 4 — the observability audit
# --------------------------------------------------------------------------
def fig_audit(outdir: Path):
    # Descriptor F-ratios and probe values: descriptor / probe scripts,
    # recorded in the supplement (Tab. 3 and App. D).
    desc = [("straightness", 0.14), ("loiter fraction", 0.17),
            ("turn rate", 0.38), ("speed std", 0.97), ("mean speed", 2.47)]
    steps = [("by vessel", 0.83), ("by O–D route", 2.45),
             ("by vessel,\nroute-controlled", 0.23)]
    probe = [("metadata\nin $\\phi(s)$", 0.63), ("metadata\nhidden", 0.14)]

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.0, 2.2),
                                     gridspec_kw={"width_ratios": [1.15, 1.0, 0.75]})

    # (a) where the heterogeneity lives
    names = [d[0] for d in desc]
    vals = np.array([d[1] for d in desc])
    ys = np.arange(len(desc))
    cols = [C["pemirl"] if v > 1 else "#B6B6B6" for v in vals]
    ax.barh(ys, vals, height=0.62, color=cols, linewidth=0)
    for y, v in zip(ys, vals):
        ax.text(v + 0.06, y, f"{v:.2f}", va="center", fontsize=6.5, color="#2B2B2B")
    ax.axvline(1.0, color="#B00020", ls=(0, (3, 2)), lw=0.9)
    ax.text(1.07, len(desc) - 1.55, "$F=1$", color="#B00020", fontsize=6.5,
            ha="left", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("$F$ by vessel")
    ax.set_title("(a) Only speed varies by vessel;\nroute shape does not", loc="left")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    # (b) the heterogeneity is circumstantial
    xs = np.arange(3)
    v = np.array([s[1] for s in steps])
    bx.bar(xs, v, width=0.58, color=[C["mce"], C["airl"], "#B6B6B6"], linewidth=0)
    for x, y in zip(xs, v):
        bx.text(x, y + 0.06, f"{y:.2f}", ha="center", fontsize=6.8, color="#2B2B2B")
    bx.axhline(1.0, color="#B00020", ls=(0, (3, 2)), lw=0.9)
    bx.annotate("control for route\n$\\rightarrow$ vessel effect gone",
                xy=(2, 0.34), xytext=(2, 1.55), fontsize=6.6, ha="center",
                va="bottom", color="#2B2B2B",
                arrowprops=dict(arrowstyle="-|>", color="#2B2B2B", lw=0.9))
    bx.set_xticks(xs)
    bx.set_xticklabels([s[0] for s in steps], fontsize=6.2)
    bx.set_ylim(0, 3.0)
    bx.set_ylabel("$F$ (all descriptors)")
    bx.set_xlabel("grouping", fontsize=7)
    bx.set_title("(b) Route, not vessel, is the\ngrouping that matters", loc="left")
    bx.yaxis.grid(True)
    bx.set_axisbelow(True)

    # (c) what the trained context contains
    xs = np.arange(2)
    v = np.array([p[1] for p in probe])
    cx.bar(xs, v, width=0.52, color=[C["pemirl"], "#B6B6B6"], linewidth=0)
    for x, y in zip(xs, v):
        cx.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=6.8, color="#2B2B2B")
    cx.set_xticks(xs)
    cx.set_xticklabels([p[0] for p in probe], fontsize=6.6)
    cx.set_ylim(0, 0.95)
    cx.set_ylabel("$R^2$: observables from $z$")
    cx.set_title("(c) The context re-encodes\nwhat is already observed", loc="left")
    cx.yaxis.grid(True)
    cx.set_axisbelow(True)
    cx.text(0.98, 0.99, "posterior active\n($\\sigma_z/\\sigma_{sa}=0.60$)",
            transform=cx.transAxes, ha="right", va="top", fontsize=6.4,
            color="#2B2B2B")

    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_audit")


# --------------------------------------------------------------------------
# Figure 5 — semi-synthetic controlled test: available vs collected headroom
# --------------------------------------------------------------------------
def fig_synthetic(outdir: Path):
    root = ROOT / "experiments/synthetic_context/runs"

    def headroom(prefix, dose):
        """+nats/decision an oracle style-conditioned BC gains over pooled BC."""
        for d in (f"{dose:g}", f"{dose:.1f}", f"{dose:.2f}"):
            f = root / f"probe_{prefix}_{d}.log"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if "ORACLE CONTEXT HEADROOM" in line:
                    return float(line.split(":")[1].split()[0])
        return None

    axes_spec = [
        ("dose_response.json", "oracle",
         "(a) corridor-preference styles", "route choice at junctions"),
        ("actbias_response.json", "actbias",
         "(b) directional-habit styles", "a bias at every decision"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (fname, prefix, title, sub) in zip(axs, axes_spec):
        rows = json.loads((root / "eval" / fname).read_text())
        rows = [r for r in rows if "pemirl_ll" in r]
        eta = [r["eta2"] for r in rows]
        gap = [r["pemirl_ll"] - r["airl_ll"] for r in rows]
        hr = [headroom(prefix, r["dose"]) for r in rows]

        ax.axhline(0, color="#8A8A8A", lw=0.8, zorder=1)
        keep = [(e, h) for e, h in zip(eta, hr) if h is not None]
        if keep:
            ax.plot(*zip(*keep), color=C["canon"], marker="o", ms=5, zorder=3,
                    label="available: oracle-label headroom")
        ax.plot(eta, gap, color=C["pemirl"], marker="s", ms=5, zorder=3,
                label="collected: PEMIRL $-$ AIRL")
        ax.set_xlabel("injected heterogeneity $\\eta^2$")
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=7.8)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)

    axs[0].set_ylabel("$\\Delta$ LL / decision (nats)")
    axs[0].set_ylim(-0.20, 0.20)
    axs[1].annotate("headroom exists\nand is verified", xy=(0.936, 0.133),
                    xytext=(0.55, 0.165), fontsize=6.8, color=C["canon"],
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=C["canon"], lw=0.8))
    axs[1].annotate("the mechanism\ncollects none of it", xy=(0.936, -0.12),
                    xytext=(0.5, -0.17), fontsize=6.8, color=C["pemirl"],
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=C["pemirl"], lw=0.8))
    axs[0].legend(loc="upper left", frameon=False, handlelength=1.4, fontsize=6.8)
    fig.tight_layout(pad=0.4)
    save(fig, outdir, "fig_synthetic")


def save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir/name}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="docs/aaai-paper-consolidated/figures")
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    out = (ROOT / a.outdir) if not Path(a.outdir).is_absolute() else Path(a.outdir)
    figs = {"ladder": fig_ladder, "criteria": fig_criteria,
            "models": fig_models, "transfer": fig_transfer,
            "audit": fig_audit, "synthetic": fig_synthetic}
    for k, f in figs.items():
        if a.only and k not in a.only:
            continue
        f(out)


if __name__ == "__main__":
    main()
