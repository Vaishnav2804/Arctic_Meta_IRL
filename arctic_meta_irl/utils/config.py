"""YAML config loading with single-level inheritance via the ``base`` key.

``base:`` path resolution (in order — the first existing path wins):

1. The ``base`` value taken as-is (absolute, or relative to the *current
   working directory*).
2. The ``base`` value relative to the *directory of the config file* being
   loaded (e.g. ``smoke/config.yaml`` with ``base: ../configs/default.yaml``
   resolves to ``configs/default.yaml`` at the repo root).
3. Just the *filename* of the ``base`` value, looked up next to the config
   file (e.g. ``base: configs/default.yaml`` inside ``configs/mce_irl.yaml``
   falls back to ``configs/default.yaml``).

Note that step 1 makes resolution CWD-dependent by design: running from the
repo root lets configs reference ``configs/default.yaml`` directly, while
steps 2–3 keep the same configs working from any CWD.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, nested. cfg.paths.voyages_pkl etc."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, key: str) -> Path:
        return Path(self["paths"][key])


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config, applying ``base:`` inheritance and dict overrides.

    See the module docstring for the exact ``base:`` path-resolution order.
    ``overrides`` (if given) are deep-merged on top of the final config.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_key = cfg.pop("base", None)
    if base_key:
        base_path = Path(base_key)
        if not base_path.exists():  # allow paths relative to the config file
            base_path = (path.parent / base_key).resolve()
        if not base_path.exists():
            base_path = path.parent / Path(base_key).name
        base = load_config(base_path)
        cfg = _deep_update(base, cfg)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return Config(cfg)
