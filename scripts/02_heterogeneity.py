"""02 — Behavioral heterogeneity analysis (paper Sec. III, first paragraph).

    python scripts/02_heterogeneity.py --config configs/default.yaml

Computes behavior descriptors per trajectory, multivariate eta-squared effect
sizes for {mmsi, size_class, subtype, category}, KMeans clustering with
silhouette scores, and the short-haul-excluded robustness check.

Writes : <workdir>/heterogeneity/descriptors.csv, eta_squared.json, clusters.csv
"""
from __future__ import annotations

import json
from pathlib import Path

from common import base_parser, load_cfg, load_pipeline, log, workdir

from arctic_meta_irl.data.loaders import load_vessel_registry
from arctic_meta_irl.eval.heterogeneity import (behavior_descriptors,
                                                cluster_trajectories,
                                                eta_squared_table,
                                                exclude_short_haul)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n-clusters", type=int, default=5)
    p.add_argument("--min-od-km", type=float, default=100.0,
                   help="short-haul threshold for the robustness check")
    p.add_argument("--out", default=None,
                   help="output directory (default: <workdir>/heterogeneity)")
    args = p.parse_args()
    cfg = load_cfg(args)

    mdp, fb, eps = load_pipeline(cfg)
    episodes = eps["train"] + eps.get("val", []) + eps.get("test", [])
    registry = load_vessel_registry(cfg["paths"]["vessel_registry"])

    df = behavior_descriptors(episodes, mdp, registry)
    log.info("%d trajectories from %d vessels",
             len(df), df["mmsi"].nunique())

    out = Path(args.out) if args.out else workdir(cfg) / "heterogeneity"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "descriptors.csv", index=False)

    # ---- effect sizes ---------------------------------------------------------
    eta_all = eta_squared_table(df)
    df_long = exclude_short_haul(df, min_od_km=args.min_od_km)
    eta_long = eta_squared_table(df_long)
    log.info("eta^2 (all %d trajectories):\n%s", len(df), eta_all.round(3))
    log.info("eta^2 (short-haul excluded, %d trajectories):\n%s",
             len(df_long), eta_long.round(3))

    with open(out / "eta_squared.json", "w") as f:
        json.dump({"all": eta_all.round(4).to_dict(),
                   "short_haul_excluded": eta_long.round(4).to_dict(),
                   "n_trajectories": int(len(df)),
                   "n_vessels": int(df["mmsi"].nunique())}, f, indent=2)

    # ---- clustering -------------------------------------------------------------
    labels, sil = cluster_trajectories(df, n_clusters=args.n_clusters,
                                       seed=int(cfg["seed"]))
    df["cluster"] = labels
    df.to_csv(out / "clusters.csv", index=False)
    log.info("KMeans k=%d silhouette=%.3f; cluster sizes=%s",
             args.n_clusters, sil,
             df["cluster"].value_counts().sort_index().tolist())
    log.info("Results written to %s/", out)


if __name__ == "__main__":
    main()
