"""Loaders for the raw project inputs.

These functions are written against the *exact* files used in the paper:

* ``timestamped_voyages_all.pkl`` — 3,254 AIS voyages. Each voyage is a mapping
  with keys ``mmsi``, ``category`` (cargo/tanker/other), ``cells`` (H3 sequence),
  ``speeds``, ``times``, ``year``, ``month``, ``start``, ``goal`` plus metadata.
* ``vessel_registry_unified.json`` — vessel metadata keyed by MMSI.
* ``all_water_arctic_r6.gexf`` — H3 res-6 navigation graph
  (14,206 water-cell nodes, 39,053 neighbor edges).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Voyages
# --------------------------------------------------------------------------- #

REQUIRED_VOYAGE_KEYS = ("mmsi", "category", "cells")
OPTIONAL_VOYAGE_KEYS = ("speeds", "times", "year", "month", "start", "goal")


@dataclass
class Voyage:
    """One AIS voyage mapped onto the H3 grid (raw, pre-MDP)."""

    mmsi: int
    category: str                       # cargo / tanker / other
    cells: list[str]                    # H3 cell index strings
    speeds: list[float] = field(default_factory=list)
    times: list[Any] = field(default_factory=list)   # timestamps (any parseable)
    year: int | None = None
    month: int | None = None
    start: str | None = None
    goal: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start is None and self.cells:
            self.start = self.cells[0]
        if self.goal is None and self.cells:
            self.goal = self.cells[-1]

    def __len__(self) -> int:
        return len(self.cells)


def _as_voyage(rec: Any) -> Voyage:
    """Coerce one record from the pickle into a Voyage (dict or object-like)."""
    if isinstance(rec, Voyage):
        return rec
    if not isinstance(rec, dict):
        # tolerate namedtuple / simple objects
        rec = {k: getattr(rec, k) for k in dir(rec) if not k.startswith("_")}
    missing = [k for k in REQUIRED_VOYAGE_KEYS if k not in rec]
    if missing:
        raise KeyError(f"voyage record missing required keys {missing}; got {sorted(rec)}")
    known = set(REQUIRED_VOYAGE_KEYS) | set(OPTIONAL_VOYAGE_KEYS)
    meta = {k: v for k, v in rec.items() if k not in known}
    return Voyage(
        mmsi=int(rec["mmsi"]),
        category=str(rec["category"]).lower(),
        cells=[str(c) for c in rec["cells"]],
        speeds=list(rec.get("speeds") or []),
        times=list(rec.get("times") or []),
        year=rec.get("year"),
        month=rec.get("month"),
        start=rec.get("start"),
        goal=rec.get("goal"),
        meta=meta,
    )


def load_voyages(path: str | Path,
                 categories: Iterable[str] | None = ("cargo", "tanker"),
                 min_len: int = 2) -> list[Voyage]:
    """Load ``timestamped_voyages_all.pkl``.

    The pickle may be a list of dicts, or a dict mapping an id -> record;
    both are handled. Voyages are filtered to the given ``categories``
    (case-insensitive; ``None`` keeps all) and to ``len(cells) >= min_len``.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = pickle.load(f)
    records = list(raw.values()) if isinstance(raw, dict) else list(raw)
    voyages = [_as_voyage(r) for r in records]
    n_total = len(voyages)
    if categories is not None:
        cats = {c.lower() for c in categories}
        voyages = [v for v in voyages if v.category in cats]
    voyages = [v for v in voyages if len(v) >= min_len]
    log.info("Loaded %d/%d voyages from %s (categories=%s, min_len=%d)",
             len(voyages), n_total, path.name, list(categories or []), min_len)
    return voyages


# --------------------------------------------------------------------------- #
# Vessel registry
# --------------------------------------------------------------------------- #

def load_vessel_registry(path: str | Path) -> dict[int, dict]:
    """Load ``vessel_registry_unified.json`` -> {mmsi: metadata}.

    MMSI keys are normalized to int. Values are kept as-is; downstream feature
    extraction (``vessel_features.py``) is defensive about which fields exist.
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):  # tolerate a list of records with an mmsi field
        raw = {r["mmsi"]: r for r in raw}
    reg: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            reg[int(k)] = v
        except (TypeError, ValueError):
            log.warning("Skipping registry entry with non-numeric MMSI key: %r", k)
    log.info("Loaded vessel registry: %d vessels from %s", len(reg), path.name)
    return reg


# --------------------------------------------------------------------------- #
# Navigation graph
# --------------------------------------------------------------------------- #

def load_navigation_graph(path: str | Path) -> nx.Graph:
    """Load the H3 res-6 water graph (``all_water_arctic_r6.gexf``).

    Node ids are H3 index strings. If the GEXF stores lat/lng node attributes
    they are preserved; otherwise centroids are derived later via the ``h3``
    package (see :mod:`arctic_meta_irl.env.graph_mdp`).
    """
    path = Path(path)
    g = nx.read_gexf(path)
    g = nx.Graph(g)  # drop any directedness / multi-edges; neighbors are symmetric
    log.info("Loaded navigation graph: %d nodes, %d edges from %s",
             g.number_of_nodes(), g.number_of_edges(), path.name)
    return g


def cell_centroid(cell: str, node_attrs: dict | None = None) -> tuple[float, float]:
    """(lat, lng) of an H3 cell. Prefers GEXF node attributes, falls back to h3."""
    if node_attrs:
        lat = node_attrs.get("lat", node_attrs.get("latitude"))
        lng = node_attrs.get("lng", node_attrs.get("lon", node_attrs.get("longitude")))
        if lat is not None and lng is not None:
            return float(lat), float(lng)
    try:
        import h3
        if hasattr(h3, "cell_to_latlng"):           # h3 v4
            return tuple(h3.cell_to_latlng(cell))    # type: ignore[return-value]
        return tuple(h3.h3_to_geo(cell))             # h3 v3
    except ImportError as e:
        raise RuntimeError(
            "Cell centroid unavailable: GEXF has no lat/lng attributes and the "
            "`h3` package is not installed (pip install h3)."
        ) from e


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometers."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def initial_bearing_rad(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial great-circle bearing (radians, from north) from point 1 to 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lng2 - lng1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.arctan2(x, y))
