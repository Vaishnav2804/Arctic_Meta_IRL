"""Guards for the context sample-complexity harness (scripts/15_support_sweep).

The whole point of the matched sweep is that the QUERY set is frozen while
only the support grows, so k arms score identical decisions. If that ever
breaks, the k-curve becomes an apples-to-oranges comparison again (the exact
flaw in the orphaned runs/eval/support_sweep.json). These tests pin it.
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ss = importlib.import_module("15_support_sweep")

from arctic_meta_irl.data.dataset import Episode


def _ep(mmsi, voyage_index, year, month, T=4):
    states = np.arange(T + 1, dtype=np.int32)
    actions = np.zeros(T, dtype=np.int32)
    return Episode(states=states, actions=actions, goal=int(states[-1]),
                   mmsi=mmsi, category="cargo", year=year, month=month,
                   voyage_index=voyage_index)


def _episodes():
    eps = []
    vi = 0
    for mmsi in (111, 222):
        for i in range(9):                      # 9 episodes/vessel -> Q=3
            eps.append(_ep(mmsi, vi, 2022, (i % 12) + 1))
            vi += 1
    return eps


def test_query_frozen_and_disjoint_from_pool():
    smap = ss.frozen_query_split(_episodes())
    assert set(smap) == {111, 222}
    for mmsi, (pool, query) in smap.items():
        pv = {e.voyage_index for e in pool}
        qv = {e.voyage_index for e in query}
        assert len(query) == 3                  # Q = 9 // 3
        assert not (pv & qv)                     # support can never leak query
        # query is the chronological tail (strictly after the pool)
        assert min(qv) > max(pv)


def test_selection_never_touches_query_across_k():
    """Every k arm must score the identical query voyages."""
    smap = ss.frozen_query_split(_episodes())
    query_ids = {m: {e.voyage_index for e in q} for m, (_, q) in smap.items()}
    rng = np.random.default_rng(0)
    for k in ss.K_GRID:
        for mmsi, (pool, query) in smap.items():
            support = ss.select_support(pool, k, "random", rng, mdp=None)
            assert {e.voyage_index for e in support} <= {e.voyage_index
                                                         for e in pool}
            assert {e.voyage_index for e in query} == query_ids[mmsi]
        assert (k == 0) == (len(ss.select_support(
            smap[111][0], k, "random", rng, mdp=None)) == 0)
