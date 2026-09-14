"""Vessel-level train/val/test splits (paper §II: "vessel-level train/test
splits to avoid leakage, with a validation set for hyperparameter tuning").

Each MMSI is assigned to exactly one split, so no vessel contributes
demonstrations to both training and evaluation. Optionally, the most recent
``shift_months`` of *training-vessel* voyages are diverted into a separate
``temporal_shift`` evaluation set (generalization to temporal shifts, §II).

Determinism: vessels are sorted before shuffling with
``np.random.default_rng(seed)``, so the same (voyages, seed) always yields the
same split regardless of insertion order.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .loaders import Voyage
from ..utils.logging import get_logger

log = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test", "temporal_shift")


def _voyage_date_key(v: Voyage) -> tuple[int, int]:
    return (int(v.year or 0), int(v.month or 0))


def make_splits(voyages: list[Voyage], train: float = 0.7, val: float = 0.1,
                test: float = 0.2, seed: int = 0,
                temporal_shift_test: bool = True,
                shift_months: int = 3) -> dict[str, list[int]]:
    """Return {split_name: [voyage indices]}; vessel-disjoint train/val/test."""
    assert abs(train + val + test - 1.0) < 1e-6, "split fractions must sum to 1"
    by_vessel: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(voyages):
        by_vessel[v.mmsi].append(i)

    mmsis = np.array(sorted(by_vessel))
    rng = np.random.default_rng(seed)
    rng.shuffle(mmsis)
    n = len(mmsis)
    n_tr, n_va = int(round(train * n)), int(round(val * n))
    vessel_split = {
        "train": set(mmsis[:n_tr].tolist()),
        "val": set(mmsis[n_tr:n_tr + n_va].tolist()),
        "test": set(mmsis[n_tr + n_va:].tolist()),
    }

    splits: dict[str, list[int]] = {k: [] for k in SPLIT_NAMES}
    for name in ("train", "val", "test"):
        for m in vessel_split[name]:
            splits[name].extend(by_vessel[m])

    if temporal_shift_test and shift_months > 0:
        dated = [(i, _voyage_date_key(voyages[i])) for i in splits["train"]
                 if _voyage_date_key(voyages[i]) != (0, 0)]
        if dated:
            months = sorted({d for _, d in dated})
            cutoff = months[max(0, len(months) - shift_months)]
            shifted = {i for i, d in dated if d >= cutoff}
            splits["temporal_shift"] = sorted(shifted)
            splits["train"] = [i for i in splits["train"] if i not in shifted]

    log.info("Splits (voyages): train=%d val=%d test=%d temporal_shift=%d "
             "| vessels: train=%d val=%d test=%d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]),
             len(splits["temporal_shift"]),
             len(vessel_split["train"]), len(vessel_split["val"]),
             len(vessel_split["test"]))
    return splits


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: list(map(int, v)) for k, v in splits.items()}, f)


def load_splits(path: str | Path) -> dict[str, list[int]]:
    with open(path) as f:
        return {k: list(v) for k, v in json.load(f).items()}
