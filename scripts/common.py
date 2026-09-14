"""Shared helpers for the pipeline scripts (00-08).

Centralizes loading the cached MDP, episodes, splits, and the fitted
FeatureBuilder so every script reconstructs an identical pipeline state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# allow `python scripts/XX.py` from the repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arctic_meta_irl.data.dataset import Episode, load_episodes
from arctic_meta_irl.data.loaders import load_vessel_registry, load_voyages
from arctic_meta_irl.data.splits import load_splits
from arctic_meta_irl.data.vessel_features import VesselFeaturizer
from arctic_meta_irl.data.weather import WeatherStore
from arctic_meta_irl.features.builder import FeatureBuilder, FeatureSpec
from arctic_meta_irl.env.graph_mdp import GraphMDP
from arctic_meta_irl.utils.config import Config, load_config
from arctic_meta_irl.utils.logging import get_logger

log = get_logger("scripts")

# released checkpoint locations (used by the --released flag on 06/08)
RELEASED_MCE_CKPT = "models/mce_irl/theta.npz"
RELEASED_PEMIRL_CKPT = "models/pemirl/pemirl.pt"
RELEASED_TRANSFER_DIR = "models/ppo_on_pemirl"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml",
                   help="YAML config (inherits configs/default.yaml via `base`)")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--device", default=None, help="cpu | cuda | auto")
    return p


def load_cfg(args) -> Config:
    overrides = {}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "device", None) is not None:
        overrides["device"] = args.device
    return load_config(args.config, overrides)


def workdir(cfg: Config) -> Path:
    return Path(cfg["paths"]["workdir"])


def seeded_dir(path: Path, seed: int) -> Path:
    """Suffix a run directory with _s{seed} so multi-seed runs don't overwrite.

    Seed 0 keeps the unsuffixed path — the released-artifact convention."""
    if seed == 0:
        return path
    return path.with_name(f"{path.name}_s{seed}")


# --------------------------------------------------------------------------- #
# Cached pipeline state
# --------------------------------------------------------------------------- #

def load_mdp(cfg: Config) -> GraphMDP:
    cache = Path(cfg["paths"]["mdp_cache"])
    if not cache.exists():
        raise FileNotFoundError(
            f"MDP cache {cache} not found; run scripts/00_build_mdp.py first.")
    return GraphMDP.load(cache)


def build_feature_builder(cfg: Config, mdp: GraphMDP,
                          fit_mmsis: list[int] | None = None) -> FeatureBuilder:
    """FeatureBuilder wired to the weather caches + vessel registry.

    If `fit_mmsis` is given the VesselFeaturizer type vocabulary is fit on
    them; otherwise the saved feature cache (scripts/01) is loaded.
    """
    spec = FeatureSpec.from_config(cfg)
    era5 = WeatherStore(cfg["paths"]["era5_dir"],
                        variables=list(cfg["features"]["era5_vars"]), name="era5")
    oras5 = WeatherStore(cfg["paths"]["oras5_dir"],
                         variables=list(cfg["features"]["oras5_vars"]), name="oras5")
    vessels = None
    if spec.use_vessel_static:
        registry = load_vessel_registry(cfg["paths"]["vessel_registry"])
        vessels = VesselFeaturizer(registry)
        if fit_mmsis is not None:
            vessels.fit(fit_mmsis)
    fb = FeatureBuilder(mdp, era5, oras5, vessels, spec)
    if fit_mmsis is None:
        _import_feature_cache(cfg, fb)
    return fb


def save_feature_cache(cfg: Config, fb: FeatureBuilder) -> None:
    path = Path(cfg["paths"]["feature_cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    st = fb.export_state()
    np.savez(path,
             mu=st["mu"] if st["mu"] is not None else np.zeros(0),
             sd=st["sd"] if st["sd"] is not None else np.zeros(0),
             log1p_mask=st["log1p_mask"] if st["log1p_mask"] is not None
             else np.zeros(0, dtype=bool),
             passt_mask=st["passt_mask"] if st["passt_mask"] is not None
             else np.zeros(0, dtype=bool),
             era5_vars=np.array(st["era5_vars"], dtype=object),
             oras5_vars=np.array(st["oras5_vars"], dtype=object),
             type_vocab=np.array(
                 fb.vessels.type_vocab if fb.vessels is not None else [],
                 dtype=object))
    log.info("Feature cache saved to %s (D=%d)", path, fb.dim)


def _import_feature_cache(cfg: Config, fb: FeatureBuilder) -> None:
    path = Path(cfg["paths"]["feature_cache"])
    if not path.exists():
        log.warning("Feature cache %s missing; features are unstandardized and "
                    "the vessel type vocabulary is empty. Run scripts/01 first.",
                    path)
        return
    z = np.load(path, allow_pickle=True)
    fb.import_state({"mu": z["mu"] if z["mu"].size else None,
                     "sd": z["sd"] if z["sd"].size else None,
                     "log1p_mask": z["log1p_mask"] if "log1p_mask" in z.files
                     and z["log1p_mask"].size else None,
                     "passt_mask": z["passt_mask"] if "passt_mask" in z.files
                     and z["passt_mask"].size else None,
                     "era5_vars": list(z["era5_vars"]),
                     "oras5_vars": list(z["oras5_vars"])})
    fb.era5_vars = [str(v) for v in z["era5_vars"]]
    fb.oras5_vars = [str(v) for v in z["oras5_vars"]]
    if fb.vessels is not None:
        fb.vessels.type_vocab = [str(t) for t in z["type_vocab"]]


def load_split_episodes(cfg: Config) -> dict[str, list[Episode]]:
    """{split_name: [Episode]} using the cached episodes + vessel-level splits.

    Split indices refer to *voyage* indices; Episode.voyage_index links back.
    """
    episodes = load_episodes(cfg["paths"]["episodes_cache"])
    splits = load_splits(cfg["paths"]["splits_cache"])
    by_voyage = {ep.voyage_index: ep for ep in episodes}
    out = {}
    for name, idxs in splits.items():
        out[name] = [by_voyage[i] for i in idxs if i in by_voyage]
        log.info("split %-14s: %5d voyages -> %5d episodes",
                 name, len(idxs), len(out[name]))
    return out


def load_pipeline(cfg: Config):
    """(mdp, features, split_episodes) — the standard state for scripts 02-08."""
    mdp = load_mdp(cfg)
    fb = build_feature_builder(cfg, mdp)  # imports the saved feature cache
    eps = load_split_episodes(cfg)
    return mdp, fb, eps


def load_raw_voyages(cfg: Config):
    return load_voyages(cfg["paths"]["voyages_pkl"],
                        categories=tuple(cfg["data"]["categories"]),
                        min_len=int(cfg["data"]["min_episode_len"]))
