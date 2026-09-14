"""Build a Gulf-of-St-Lawrence (GSL) IRL dataset in the exact format the
meta-IRL pipeline consumes, for the second-region replication.

Inputs  (already staged, raw AIS CSVs):
    <gsl_csv>/static/gsl_static_YYYYMM.csv   -- mmsi, ship_type (AIS code), dims
    <gsl_csv>/dynamic/gsl_dynamic_YYYYMM.csv -- mmsi, time (epoch), lat, lon, sog

Outputs (drop-in for configs/gsl.yaml paths):
    data/gsl/voyages_gsl.pkl          -- list[dict]: {mmsi, category, cells,
                                          speeds, times, year, month, start, goal}
    data/gsl/vessel_registry_gsl.json -- {mmsi: {length, width, ship_type, category}}
    data/gsl/gsl_r6.gexf              -- H3 res-6 navigation graph (AIS-derived)

Method (mirrors the Arctic corpus conventions, loaders.py / dataset.py):
  * category from AIS ship_type: 70-79 -> cargo, 80-89 -> tanker (others dropped)
  * per-MMSI voyage segmentation by a time gap (default 6 h)
  * lat/lon -> H3 res-6 cell; consecutive duplicates collapsed
  * graph = visited cells + edges between H3-adjacent visited cells, plus
    single intermediate cells that connect grid-distance-2 consecutive track
    cells (keeps the graph on water, i.e. the support of observed traffic,
    while maximizing episode yield through voyage_to_episode's 1-gap bridge)

    python scripts/build_gsl_dataset.py --years 2024 --gsl_csv /abs/path/to/gsl_csv
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h3
import networkx as nx
import pandas as pd

H3_RES = 6
GAP_SECONDS = 6 * 3600          # new voyage when the time gap exceeds this
STOP_KN = 0.5                   # speed below this counts as "stopped"
STOP_SECONDS = 30 * 60          # a stop longer than this ends a voyage (port call)
MIN_CELLS = 5                   # minimum distinct-cell voyage length to keep


def split_at_stops(t, sog):
    """Indices (start, end) of moving legs, splitting at sustained port stops."""
    subs = []
    n = len(t)
    cur = 0
    i = 0
    while i < n:
        if sog[i] < STOP_KN:
            j = i
            while j < n and sog[j] < STOP_KN:
                j += 1
            stop_dur = (t[j - 1] - t[i]) if j > i else 0
            if stop_dur > STOP_SECONDS:      # berthed: close leg, restart after
                if i > cur:
                    subs.append((cur, i))
                cur = j
            i = j
        else:
            i += 1
    if n > cur:
        subs.append((cur, n))
    return subs


def category_from_ship_type(code) -> str | None:
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return None
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    return None


def load_registry(static_files: list[Path]) -> dict[int, dict]:
    """mmsi -> {length, width, ship_type, category} from the static CSVs.

    Dimensions: length = dim_bow + dim_stern, width = dim_port + dim_star.
    Keeps the row with the largest reported length per MMSI (most complete)."""
    reg: dict[int, dict] = {}
    cols = ["mmsi", "ship_type", "dim_bow", "dim_stern", "dim_port", "dim_star"]
    for f in static_files:
        try:
            df = pd.read_csv(f, usecols=cols)
        except (ValueError, FileNotFoundError):
            continue
        for r in df.itertuples(index=False):
            cat = category_from_ship_type(r.ship_type)
            if cat is None:
                continue
            length = float(r.dim_bow or 0) + float(r.dim_stern or 0)
            width = float(r.dim_port or 0) + float(r.dim_star or 0)
            m = int(r.mmsi)
            prev = reg.get(m)
            if prev is None or length > prev["length"]:
                reg[m] = {"length": length, "width": width,
                          "ship_type": int(float(r.ship_type)), "category": cat}
    return reg


def segment_voyages(df_mmsi: pd.DataFrame, mmsi: int, cat: str) -> list[dict]:
    """One MMSI's time-sorted pings -> list of voyage dicts."""
    df_mmsi = df_mmsi.sort_values("time")
    t = df_mmsi["time"].to_numpy()
    lat = df_mmsi["latitude"].to_numpy()
    lon = df_mmsi["longitude"].to_numpy()
    sog = df_mmsi["sog"].to_numpy()
    # split first by temporal gaps, then by sustained port stops within each gap
    voyages = []
    start = 0
    n = len(t)
    gap_bounds = []
    for i in range(1, n + 1):
        if i == n or (t[i] - t[i - 1]) > GAP_SECONDS:
            gap_bounds.append((start, i))
            start = i
    for gs, ge in gap_bounds:
        la, lo, sg, tt = lat[gs:ge], lon[gs:ge], sog[gs:ge], t[gs:ge]
        for ls, le in split_at_stops(tt, sg):
            if le - ls < MIN_CELLS:
                continue
            cells, speeds, times = [], [], []
            for k in range(ls, le):
                c = h3.latlng_to_cell(float(la[k]), float(lo[k]), H3_RES)
                if cells and c == cells[-1]:
                    continue  # collapse consecutive duplicates
                cells.append(c)
                speeds.append(float(sg[k]))
                times.append(int(tt[k]))
            if len(cells) < MIN_CELLS or cells[0] == cells[-1]:
                continue
            dt = datetime.fromtimestamp(int(tt[ls]), tz=timezone.utc)
            voyages.append({
                "mmsi": mmsi, "category": cat, "cells": cells,
                "speeds": speeds, "times": times,
                "year": dt.year, "month": dt.month,
                "start": cells[0], "goal": cells[-1]})
    return voyages


def build_graph(voyages: list[dict]) -> nx.Graph:
    """AIS-derived res-6 water graph: visited cells + H3 adjacency, plus
    along-track single-cell bridges for grid-distance-2 consecutive cells."""
    nodes: set[str] = set()
    for v in voyages:
        nodes.update(v["cells"])
    # add along-track bridge cells (shared H3 neighbor of grid-distance-2 pairs)
    bridges: set[str] = set()
    for v in voyages:
        cs = v["cells"]
        for a, b in zip(cs, cs[1:]):
            if a == b or h3.are_neighbor_cells(a, b):
                continue
            if h3.grid_distance(a, b) == 2:
                common = set(h3.grid_ring(a, 1)) & set(h3.grid_ring(b, 1))
                bridges.update(common)
    nodes |= bridges
    g = nx.Graph()
    for c in nodes:
        lat, lng = h3.cell_to_latlng(c)
        g.add_node(c, lat=float(lat), lng=float(lng))
    for c in nodes:
        for nb in h3.grid_ring(c, 1):
            if nb in nodes:
                g.add_edge(c, nb)
    # keep the largest connected component (a single navigable water body)
    if g.number_of_nodes():
        comps = sorted(nx.connected_components(g), key=len, reverse=True)
        g = g.subgraph(comps[0]).copy()
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsl_csv", default="data/gsl_csv")
    ap.add_argument("--years", nargs="+", type=int, default=[2024])
    ap.add_argument("--months", nargs="+", type=int,
                    default=list(range(1, 13)))
    ap.add_argument("--out_dir", default="data/gsl")
    ap.add_argument("--max_vessels", type=int, default=0,
                    help="cap number of vessels (0 = keep all); busiest kept")
    ap.add_argument("--max_per_vessel", type=int, default=0,
                    help="cap voyages per vessel (0 = keep all); bounds MCE "
                         "group count. Random sample, seeded.")
    ap.add_argument("--min_per_vessel", type=int, default=0,
                    help="drop vessels with fewer voyages than this (so each "
                         "task has enough episodes for support+query).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    gsl = Path(args.gsl_csv)
    tags = [f"{y}{m:02d}" for y in args.years for m in args.months]
    static_files = [gsl / "static" / f"gsl_static_{t}.csv" for t in tags]
    dyn_files = [gsl / "dynamic" / f"gsl_dynamic_{t}.csv" for t in tags]

    print(f"[gsl] window: {tags[0]}..{tags[-1]} ({len(tags)} months)")
    registry = load_registry([f for f in static_files if f.exists()])
    keep_mmsi = set(registry)
    print(f"[gsl] cargo/tanker vessels in registry: {len(keep_mmsi)}")

    # stream dynamic pings for the kept vessels only
    frames = []
    use = ["mmsi", "time", "longitude", "latitude", "sog"]
    for f in dyn_files:
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=use)
        df = df[df["mmsi"].isin(keep_mmsi)]
        frames.append(df)
        print(f"[gsl]   {f.name}: {len(df):,} cargo/tanker pings")
    pings = pd.concat(frames, ignore_index=True)
    pings = pings.dropna(subset=["latitude", "longitude", "time"])
    print(f"[gsl] total cargo/tanker pings: {len(pings):,}")

    voyages: list[dict] = []
    for mmsi, grp in pings.groupby("mmsi"):
        vv = segment_voyages(grp, int(mmsi), registry[int(mmsi)]["category"])
        if args.min_per_vessel > 0 and len(vv) < args.min_per_vessel:
            continue                                    # too few for support+query
        if args.max_per_vessel > 0 and len(vv) > args.max_per_vessel:
            vv = rng.sample(vv, args.max_per_vessel)   # bound MCE group count
        voyages.extend(vv)

    # optional: keep the busiest vessels for an Arctic-comparable scale
    if args.max_vessels > 0:
        by_v = defaultdict(int)
        for v in voyages:
            by_v[v["mmsi"]] += 1
        top = {m for m, _ in sorted(by_v.items(), key=lambda kv: -kv[1])[:args.max_vessels]}
        voyages = [v for v in voyages if v["mmsi"] in top]

    graph = build_graph(voyages)
    node_set = set(graph.nodes)
    # drop voyages with <MIN_CELLS cells surviving on the graph (parity w/ pipeline)
    kept = []
    for v in voyages:
        on = [c for c in v["cells"] if c in node_set]
        if len(set(on)) >= MIN_CELLS:
            kept.append(v)
    voyages = kept

    vessels = {v["mmsi"] for v in voyages}
    reg_out = {str(m): registry[m] for m in vessels if m in registry}
    lens = [len(v["cells"]) for v in voyages]
    cats = defaultdict(int)
    for v in voyages:
        cats[v["category"]] += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "voyages_gsl.pkl", "wb") as f:
        pickle.dump(voyages, f)
    (out / "vessel_registry_gsl.json").write_text(json.dumps(reg_out))
    nx.write_gexf(graph, out / "gsl_r6.gexf")

    print("\n=== GSL dataset built ===")
    print(f"  voyages     : {len(voyages)}  ({dict(cats)})")
    print(f"  vessels     : {len(vessels)}")
    print(f"  graph       : {graph.number_of_nodes()} cells, "
          f"{graph.number_of_edges()} edges")
    print(f"  voyage len  : mean {sum(lens)/max(1,len(lens)):.1f}, "
          f"min {min(lens)}, max {max(lens)} cells")
    print(f"  written to  : {out}/")


if __name__ == "__main__":
    main()
