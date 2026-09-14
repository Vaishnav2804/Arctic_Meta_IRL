"""01 — Build demonstration episodes, vessel-level splits, and fit features.

    python scripts/01_build_dataset.py --config configs/default.yaml

Reads  : paths.voyages_pkl, paths.vessel_registry, paths.era5_dir,
         paths.oras5_dir, paths.mdp_cache
Writes : paths.episodes_cache  (data/cache/episodes.pkl)
         paths.splits_cache    (data/cache/splits.json)
         paths.feature_cache   (data/cache/features.npz: standardizer +
                                vessel type vocabulary fit on the TRAIN split)
"""
from __future__ import annotations

from common import (base_parser, build_feature_builder, load_cfg, load_mdp,
                    load_raw_voyages, log, save_feature_cache)

from arctic_meta_irl.data.dataset import build_episodes, save_episodes
from arctic_meta_irl.data.splits import make_splits, save_splits
from arctic_meta_irl.utils.seeding import set_seed


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg["seed"]))

    mdp = load_mdp(cfg)
    voyages = load_raw_voyages(cfg)
    log.info("Loaded %d voyages (categories=%s)",
             len(voyages), cfg["data"]["categories"])

    # ---- vessel-disjoint splits over voyage indices -------------------------
    sp = cfg["splits"]
    splits = make_splits(voyages,
                         train=float(sp["train"]), val=float(sp["val"]),
                         test=float(sp["test"]), seed=int(sp["seed"]),
                         temporal_shift_test=bool(sp["temporal_shift_test"]),
                         shift_months=int(sp["shift_months"]))
    save_splits(splits, cfg["paths"]["splits_cache"])

    # ---- voyages -> MDP episodes --------------------------------------------
    episodes = build_episodes(voyages, mdp, cfg)
    save_episodes(episodes, cfg["paths"]["episodes_cache"])

    by_voyage = {ep.voyage_index: ep for ep in episodes}
    train_eps = [by_voyage[i] for i in splits["train"] if i in by_voyage]
    n_vessels = len({ep.mmsi for ep in episodes})
    log.info("Episodes total=%d (%d vessels), train=%d",
             len(episodes), n_vessels, len(train_eps))

    # ---- features: vessel vocab + standardizer fit on TRAIN only ------------
    fb = build_feature_builder(cfg, mdp,
                               fit_mmsis=[ep.mmsi for ep in train_eps])
    mats = [fb.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month,
                              ep.mmsi) for ep in train_eps]
    fb.fit_standardizer(mats)
    save_feature_cache(cfg, fb)
    log.info("Feature dimension D=%d: %s", fb.dim, fb.names)


if __name__ == "__main__":
    main()
