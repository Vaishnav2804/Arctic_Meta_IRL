"""Static vessel descriptors from ``vessel_registry_unified.json``.

The registry is a per-MMSI metadata lookup. Field names vary across AIS
registries, so extraction is defensive: each descriptor is searched under a
list of aliases. Output is a fixed-length numeric vector:

    [ length_m, width_m, aspect_ratio, type-one-hot ... ]

The vessel-type one-hot is fit on the training split (``fit``) so unseen test
vessels with unknown types map to an all-zero type block rather than crashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

_LENGTH_KEYS = ("length", "length_m", "loa", "len", "dim_a_plus_b", "ship_length")
_WIDTH_KEYS = ("width", "width_m", "beam", "breadth", "dim_c_plus_d", "ship_width")
_TYPE_KEYS = ("type", "vessel_type", "ship_type", "shiptype", "subclass", "class")


def _first(d: dict, keys: tuple[str, ...], default=None):
    """First non-empty value found under any alias (exact, then case-insensitive)."""
    for k in keys:
        if k in d and d[k] not in (None, "", "nan"):
            return d[k]
        # also try case-insensitive
        for kk in d:
            if kk.lower() == k:
                if d[kk] not in (None, "", "nan"):
                    return d[kk]
    return default


def _to_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


@dataclass
class VesselFeaturizer:
    """Maps an MMSI to a fixed-length static-descriptor vector."""

    registry: dict[int, dict]
    type_vocab: list[str] = field(default_factory=list)
    # normalization constants (meters); typical Arctic cargo/tanker scale
    length_scale: float = 200.0
    width_scale: float = 35.0

    def fit(self, mmsis: list[int]) -> "VesselFeaturizer":
        """Build the vessel-type vocabulary from the training MMSIs."""
        types = set()
        uniq = set(int(m) for m in mmsis)
        missing = sorted(m for m in uniq if m not in self.registry)
        if missing:
            log.warning("%d/%d training MMSIs not in vessel registry "
                        "-> their vessel features (length/width/type) are "
                        "ZERO-filled. e.g. %s", len(missing), len(uniq),
                        missing[:10])
        for m in mmsis:
            rec = self.registry.get(int(m), {})
            t = _first(rec, _TYPE_KEYS)
            if t is not None:
                types.add(str(t).lower())
        self.type_vocab = sorted(types)
        return self

    @property
    def dim(self) -> int:
        return 3 + len(self.type_vocab)

    @property
    def names(self) -> list[str]:
        return ["v_length", "v_width", "v_aspect"] + [f"v_type={t}" for t in self.type_vocab]

    def __call__(self, mmsi: int) -> np.ndarray:
        rec = self.registry.get(int(mmsi), {})
        length = _to_float(_first(rec, _LENGTH_KEYS), 0.0)
        width = _to_float(_first(rec, _WIDTH_KEYS), 0.0)
        aspect = length / width if width > 1e-6 else 0.0
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = length / self.length_scale
        vec[1] = width / self.width_scale
        vec[2] = aspect / 10.0
        t = _first(rec, _TYPE_KEYS)
        if t is not None:
            t = str(t).lower()
            if t in self.type_vocab:
                vec[3 + self.type_vocab.index(t)] = 1.0
        return vec
