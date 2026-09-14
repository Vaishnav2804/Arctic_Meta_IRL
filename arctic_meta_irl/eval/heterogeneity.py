"""Behavioral heterogeneity analysis (paper §III).

Computes per-trajectory behavior descriptors, one-way-ANOVA effect sizes
(eta^2) for grouping factors (vessel identity / size class / subtype /
cargo-vs-tanker), and KMeans clustering of trajectories. Reproduces the
analysis behind: eta^2(mmsi)=0.66, eta^2(size class)=0.50,
eta^2(subtype)=0.35, eta^2(category)=0.11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import Episode
from ..data.loaders import haversine_km
from ..data.vessel_features import _first, _LENGTH_KEYS, _TYPE_KEYS, _to_float
from ..env.graph_mdp import GraphMDP


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #

def behavior_descriptors(episodes: list[Episode], mdp: GraphMDP,
                         registry: dict[int, dict] | None = None) -> pd.DataFrame:
    """One row per trajectory: route geometry + kinematics + grouping labels."""
    rows = []
    for ep in episodes:
        coords = mdp.latlng[ep.states]
        seg = np.array([haversine_km(*coords[i], *coords[i + 1])
                        for i in range(len(coords) - 1)])
        path_km = float(seg.sum())
        od_km = haversine_km(*coords[0], *coords[-1])
        sinuosity = path_km / max(od_km, 1e-6)
        # heading changes (proxy for maneuvering)
        dlat = np.diff(coords[:, 0]); dlng = np.diff(coords[:, 1])
        headings = np.arctan2(dlng, dlat)
        turn = np.abs(np.diff(np.unwrap(headings))) if len(headings) > 1 else np.zeros(1)
        speeds = ep.speeds[ep.speeds > 0] if ep.speeds.size else np.zeros(1)

        rec = registry.get(int(ep.mmsi), {}) if registry else {}
        length = _to_float(_first(rec, _LENGTH_KEYS), np.nan)
        subtype = str(_first(rec, _TYPE_KEYS, "unknown")).lower()
        size_class = pd.cut([length], bins=[0, 100, 150, 200, 250, 1e9],
                            labels=["<100m", "100-150m", "150-200m",
                                    "200-250m", ">250m"])[0] \
            if np.isfinite(length) else "unknown"

        rows.append({
            "mmsi": ep.mmsi, "category": ep.category, "subtype": subtype,
            "size_class": str(size_class),
            "n_steps": len(ep), "path_km": path_km, "od_km": od_km,
            "sinuosity": sinuosity,
            "mean_turn": float(turn.mean()),
            "mean_speed": float(speeds.mean()),
            "std_speed": float(speeds.std()),
            "lat_mean": float(coords[:, 0].mean()),
            "lng_mean": float(coords[:, 1].mean()),
            "month": ep.month or 0,
        })
    return pd.DataFrame(rows)


DESCRIPTOR_COLS = ["n_steps", "path_km", "od_km", "sinuosity", "mean_turn",
                   "mean_speed", "std_speed", "lat_mean", "lng_mean"]


# --------------------------------------------------------------------------- #
# eta^2 effect size
# --------------------------------------------------------------------------- #

def eta_squared(df: pd.DataFrame, factor: str,
                cols: list[str] = DESCRIPTOR_COLS) -> float:
    """Multivariate eta^2: ratio of between-group to total sum of squares,
    averaged over standardized descriptor dimensions (one-way ANOVA effect
    size; larger = the factor explains more behavioral variation)."""
    X = df[cols].to_numpy(dtype=float)
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9)
    X = np.nan_to_num(X)
    g = df[factor].astype(str).to_numpy()
    grand = X.mean(axis=0)
    ss_total = ((X - grand) ** 2).sum()
    ss_between = 0.0
    for lvl in np.unique(g):
        sub = X[g == lvl]
        ss_between += len(sub) * ((sub.mean(axis=0) - grand) ** 2).sum()
    return float(ss_between / max(ss_total, 1e-12))


def eta_squared_table(df: pd.DataFrame,
                      factors=("mmsi", "size_class", "subtype", "category")
                      ) -> pd.Series:
    return pd.Series({f: eta_squared(df, f) for f in factors}).sort_values(
        ascending=False)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

def cluster_trajectories(df: pd.DataFrame, n_clusters: int = 5,
                         cols: list[str] = DESCRIPTOR_COLS, seed: int = 0
                         ) -> tuple[np.ndarray, float]:
    """KMeans over standardized descriptors -> (labels, silhouette score)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.nan_to_num(df[cols].to_numpy(float)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    sil = silhouette_score(X, km.labels_) if n_clusters > 1 else float("nan")
    return km.labels_, float(sil)


def exclude_short_haul(df: pd.DataFrame, min_od_km: float = 100.0) -> pd.DataFrame:
    """Robustness check from the paper: differences persist after excluding
    short-haul movements."""
    return df[df["od_km"] >= min_od_km].reset_index(drop=True)
