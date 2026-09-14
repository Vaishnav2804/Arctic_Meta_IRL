"""Route- and trajectory-level evaluation metrics (paper §II):

* trajectory fit: test log-likelihood (computed by the models themselves) and
  **feature expectation error**;
* route fidelity for origin–destination paths: **Hausdorff distance** (km,
  great-circle on H3 cell centroids), **cell overlap** (Jaccard), and
  **route length ratio**.
"""
from __future__ import annotations

import numpy as np

from ..data.loaders import haversine_km
from ..env.graph_mdp import GraphMDP


def _coords(mdp: GraphMDP, path: list[int]) -> np.ndarray:
    return mdp.latlng[np.asarray(path, dtype=int)]


def hausdorff_km(mdp: GraphMDP, path_a: list[int], path_b: list[int]) -> float:
    """Symmetric Hausdorff distance between two cell paths, in km."""
    A, B = _coords(mdp, path_a), _coords(mdp, path_b)
    d = np.zeros((len(A), len(B)))
    for i, (la, lo) in enumerate(A):
        for j, (lb, lj) in enumerate(B):
            d[i, j] = haversine_km(la, lo, lb, lj)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def cell_overlap(path_a: list[int], path_b: list[int]) -> float:
    """Jaccard overlap of visited cells."""
    a, b = set(path_a), set(path_b)
    return len(a & b) / max(len(a | b), 1)


def route_length_km(mdp: GraphMDP, path: list[int]) -> float:
    c = _coords(mdp, path)
    return float(sum(haversine_km(*c[i], *c[i + 1]) for i in range(len(c) - 1)))


def route_length_ratio(mdp: GraphMDP, generated: list[int],
                       reference: list[int]) -> float:
    ref = route_length_km(mdp, reference)
    return route_length_km(mdp, generated) / max(ref, 1e-9)


def route_metrics(mdp: GraphMDP, generated: list[int],
                  reference: list[int]) -> dict[str, float]:
    return {
        "hausdorff_km": hausdorff_km(mdp, generated, reference),
        "cell_overlap": cell_overlap(generated, reference),
        "length_ratio": route_length_ratio(mdp, generated, reference),
        "reached_goal": float(generated[-1] == reference[-1]),
    }


def feature_expectation_error(phi_expert: np.ndarray,
                              phi_model: np.ndarray) -> float:
    """L2 distance between mean per-episode feature expectations."""
    return float(np.linalg.norm(phi_expert - phi_model))


def aggregate(per_route: list[dict[str, float]]) -> dict[str, float]:
    if not per_route:
        return {}
    keys = per_route[0].keys()
    return {k: float(np.mean([m[k] for m in per_route])) for k in keys}
