"""Synthetic fixtures shaped like the real Arctic inputs:

- a small hex-like navigation graph saved as GEXF (cf. all_water_arctic_r6.gexf),
- a voyages pickle of dict records (cf. timestamped_voyages_all.pkl),
- a vessel registry JSON keyed by MMSI (cf. vessel_registry_unified.json),
- ERA5/ORAS5 weather cache dirs of columnar monthly dict pickles whose month is
  encoded in the FILENAME as YYYYMM (cf. era5_h3/era5_h3_201607.pkl with
  {"cells": [...], "var1": [...], ...} parallel lists — the real layout).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

GRID_W, GRID_H = 8, 5  # 40-cell grid "ocean"


def _cell(i: int, j: int) -> str:
    return f"86cell{i:02d}{j:02d}fffff"  # H3-ish 15-char token


@pytest.fixture(scope="session")
def nav_graph() -> nx.Graph:
    g = nx.Graph()
    for i in range(GRID_W):
        for j in range(GRID_H):
            g.add_node(_cell(i, j), lat=70.0 + 0.1 * j, lng=-95.0 + 0.2 * i)
    for i in range(GRID_W):
        for j in range(GRID_H):
            if i + 1 < GRID_W:
                g.add_edge(_cell(i, j), _cell(i + 1, j))
            if j + 1 < GRID_H:
                g.add_edge(_cell(i, j), _cell(i, j + 1))
            if i + 1 < GRID_W and j + 1 < GRID_H:  # hex-ish diagonal
                g.add_edge(_cell(i, j), _cell(i + 1, j + 1))
    return g


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, nav_graph) -> Path:
    root = tmp_path_factory.mktemp("arctic_data")

    # ---- GEXF graph ---------------------------------------------------------
    nx.write_gexf(nav_graph, root / "graph.gexf")

    # ---- voyages pickle ------------------------------------------------------
    rng = np.random.default_rng(0)
    voyages = []
    mmsis = [316000000 + k for k in range(6)]
    for v in range(30):
        mmsi = mmsis[v % len(mmsis)]
        j = rng.integers(0, GRID_H)
        i0 = int(rng.integers(0, 2))
        length = int(rng.integers(5, GRID_W))
        cells = [_cell(min(i0 + t, GRID_W - 1), j) for t in range(length + 1)]
        t0 = np.datetime64("2022-07-01") + np.timedelta64(int(v), "D")
        voyages.append({
            "mmsi": mmsi,
            "category": "cargo" if v % 2 == 0 else "tanker",
            "cells": cells,
            "speeds": rng.uniform(5, 14, size=len(cells)).tolist(),
            "times": [str(t0 + np.timedelta64(2 * t, "h"))
                      for t in range(len(cells))],
            "year": 2022, "month": 7 + (v % 3),
            "start": cells[0], "goal": cells[-1],
            "metadata": {"voyage_id": v},
        })
    with open(root / "voyages.pkl", "wb") as f:
        pickle.dump(voyages, f)

    # ---- vessel registry -----------------------------------------------------
    registry = {str(m): {"length": float(120 + 30 * k), "width": float(18 + 3 * k),
                         "type": "cargo" if k % 2 == 0 else "tanker",
                         "subtype": f"sub{k % 3}"}
                for k, m in enumerate(mmsis)}
    with open(root / "registry.json", "w") as f:
        json.dump(registry, f)

    # ---- weather caches (REAL layout: columnar monthly dict pickles, YYYYMM
    #      month encoded in the filename; one value per cell = monthly mean) ---
    cells = [str(c) for c in nav_graph.nodes]
    for name, vars_ in (("era5_h3", ["u10", "v10", "siconc", "sithick"]),
                        ("oras5_h3", ["votemper"])):
        d = root / name
        d.mkdir()
        for month in (7, 8, 9):
            obj: dict = {"cells": cells}
            for var in vars_:
                if var in ("siconc", "sithick"):  # ice: non-negative, skewed
                    obj[var] = rng.uniform(0.0, 1.5, size=len(cells)).tolist()
                else:
                    obj[var] = rng.normal(size=len(cells)).tolist()
            with open(d / f"{name}_2022{month:02d}.pkl", "wb") as f:
                pickle.dump(obj, f)
    return root


@pytest.fixture(scope="session")
def cfg(data_dir, tmp_path_factory):
    """Config dict pointing at the synthetic data + a temp workdir."""
    from arctic_meta_irl.utils.config import Config
    run = tmp_path_factory.mktemp("runs")
    return Config({
        "paths": {
            "voyages_pkl": str(data_dir / "voyages.pkl"),
            "vessel_registry": str(data_dir / "registry.json"),
            "graph_gexf": str(data_dir / "graph.gexf"),
            "era5_dir": str(data_dir / "era5_h3"),
            "oras5_dir": str(data_dir / "oras5_h3"),
            "workdir": str(run),
            "mdp_cache": str(run / "mdp.npz"),
            "feature_cache": str(run / "features.npz"),
            "episodes_cache": str(run / "episodes.pkl"),
            "splits_cache": str(run / "splits.json"),
        },
        "mdp": {"h3_resolution": 6, "allow_stay": True, "max_actions": 9,
                "absorbing_goal": True, "gamma": 0.99, "max_horizon": 64},
        "features": {"era5_vars": ["u10", "v10", "siconc", "sithick"],
                     "oras5_vars": ["votemper"],
                     "use_latlon": True, "use_goal_distance": True,
                     "use_goal_bearing": True, "use_vessel_static": True,
                     "standardize": True, "monthly_aggregate": True},
        "data": {"categories": ["cargo", "tanker"], "min_episode_len": 3,
                 "drop_offgraph_cells": True, "interpolate_gaps": True,
                 "batch_size": 8, "num_workers": 0},
        "splits": {"by": "mmsi", "train": 0.7, "val": 0.1, "test": 0.2,
                   "temporal_shift_test": True, "shift_months": 1, "seed": 0},
        "logging": {"level": "INFO", "tensorboard": False},
        "seed": 0, "device": "cpu",
    })


@pytest.fixture(scope="session")
def mdp(nav_graph, cfg):
    from arctic_meta_irl.env.graph_mdp import build_mdp
    return build_mdp(nav_graph, allow_stay=True,
                     max_actions=int(cfg["mdp"]["max_actions"]))


@pytest.fixture(scope="session")
def pipeline(cfg, mdp):
    """(features, episodes) fit on the synthetic data."""
    from arctic_meta_irl.data.dataset import build_episodes
    from arctic_meta_irl.data.loaders import (load_vessel_registry, load_voyages)
    from arctic_meta_irl.data.vessel_features import VesselFeaturizer
    from arctic_meta_irl.data.weather import WeatherStore
    from arctic_meta_irl.features import FeatureBuilder, FeatureSpec

    voyages = load_voyages(cfg["paths"]["voyages_pkl"])
    episodes = build_episodes(voyages, mdp, cfg)
    spec = FeatureSpec.from_config(cfg)
    fb = FeatureBuilder(
        mdp,
        WeatherStore(cfg["paths"]["era5_dir"], cfg["features"]["era5_vars"], "era5"),
        WeatherStore(cfg["paths"]["oras5_dir"], cfg["features"]["oras5_vars"], "oras5"),
        VesselFeaturizer(load_vessel_registry(cfg["paths"]["vessel_registry"]))
        .fit([ep.mmsi for ep in episodes]),
        spec)
    fb.fit_standardizer(
        [fb.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month, ep.mmsi)
         for ep in episodes])
    return fb, episodes


@pytest.fixture(scope="session")
def feature_builder(pipeline):
    return pipeline[0]


@pytest.fixture(scope="session")
def episodes(pipeline):
    return pipeline[1]
