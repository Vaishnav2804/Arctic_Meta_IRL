"""Demonstration episodes + PyTorch ``DataLoader``.

Pipeline (driven by ``scripts/01_build_dataset.py``):

    voyages (.pkl)  ──map cells to graph──►  Episode(state_seq, action_seq, ...)
                     drop off-graph cells
                     bridge 1-cell gaps
                     deduplicate consecutive repeats (-> 'stay' suppressed)

Each :class:`Episode` carries everything the algorithms need:
state indices, action indices, the goal state, (year, month) for the weather
lookup, the MMSI for vessel statics / context, and category labels.

:class:`DemonstrationDataset` is a ``torch.utils.data.Dataset`` over episodes
yielding dense per-step tensors; :func:`make_dataloader` pads variable-length
trajectories and returns masks — this is the loader consumed by PEMIRL
(sequence models) while MCE-IRL consumes the raw episode lists directly.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .loaders import Voyage
from ..env.graph_mdp import GraphMDP
from ..features import FeatureBuilder
from ..utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Episode
# --------------------------------------------------------------------------- #

@dataclass
class Episode:
    """One demonstration mapped onto the MDP."""

    states: np.ndarray          # (T+1,) int32 state indices  s_0 .. s_T
    actions: np.ndarray         # (T,)   int32 action indices a_0 .. a_{T-1}
    goal: int                   # goal state index (== states[-1] for AIS demos)
    mmsi: int
    category: str
    year: int | None = None
    month: int | None = None
    speeds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    voyage_index: int = -1      # index back into the raw voyage list

    def __len__(self) -> int:
        return len(self.actions)


# --------------------------------------------------------------------------- #
# Voyage -> Episode mapping
# --------------------------------------------------------------------------- #

def voyage_to_episode(v: Voyage, mdp: GraphMDP, voyage_index: int = -1,
                      drop_offgraph: bool = True,
                      interpolate_gaps: bool = True,
                      max_horizon: int = 512) -> Episode | None:
    """Map a voyage's H3 cell sequence to a valid state/action path on the graph.

    Steps:
      1. keep only cells that are graph nodes (off-graph fixes: dropped)
      2. collapse consecutive duplicates (vessel lingering in a cell)
      3. for non-adjacent consecutive cells, optionally bridge via a single
         shared neighbor; otherwise split — we keep the *longest* contiguous
         segment (simple, deterministic, leakage-free)
      4. convert transitions to canonical action indices
    """
    idx = [mdp.cell_to_idx[c] for c in v.cells if c in mdp.cell_to_idx] \
        if drop_offgraph else [mdp.cell_to_idx.get(c, -1) for c in v.cells]
    idx = [s for s in idx if s >= 0]
    if len(idx) < 2:
        return None

    # collapse consecutive duplicates
    dedup = [idx[0]]
    for s in idx[1:]:
        if s != dedup[-1]:
            dedup.append(s)
    idx = dedup
    if len(idx) < 2:
        return None

    # stitch into contiguous-on-graph segments
    segments: list[list[int]] = [[idx[0]]]
    for s_prev, s in zip(idx, idx[1:]):
        if mdp.action_for_transition(s_prev, s) is not None:
            segments[-1].append(s)
            continue
        bridged = False
        if interpolate_gaps:
            nb_prev = set(mdp.neighbors(s_prev).tolist())
            nb_next = set(mdp.neighbors(s).tolist())
            common = nb_prev & nb_next
            if common:  # bridge through the geographically best shared neighbor
                mid = min(common, key=lambda m: mdp.distance_km(m, s))
                segments[-1].extend([mid, s])
                bridged = True
        if not bridged:
            segments.append([s])
    path = max(segments, key=len)
    if len(path) < 2:
        return None
    path = path[: max_horizon + 1]

    actions = np.array(
        [mdp.action_for_transition(a, b) for a, b in zip(path, path[1:])],
        dtype=np.int32)
    assert (actions >= 0).all()
    return Episode(states=np.array(path, dtype=np.int32), actions=actions,
                   goal=int(path[-1]), mmsi=v.mmsi, category=v.category,
                   year=v.year, month=v.month,
                   speeds=np.asarray(v.speeds[:len(path)], dtype=np.float32),
                   voyage_index=voyage_index)


def build_episodes(voyages: list[Voyage], mdp: GraphMDP, cfg) -> list[Episode]:
    """Map all voyages to episodes, dropping those too short after mapping."""
    d = cfg["data"]
    eps: list[Episode] = []
    dropped = 0
    for i, v in enumerate(voyages):
        ep = voyage_to_episode(
            v, mdp, voyage_index=i,
            drop_offgraph=d.get("drop_offgraph_cells", True),
            interpolate_gaps=d.get("interpolate_gaps", True),
            max_horizon=cfg["mdp"].get("max_horizon", 512))
        if ep is None or len(ep) < d.get("min_episode_len", 5):
            dropped += 1
            continue
        eps.append(ep)
    log.info("Episodes: %d kept, %d dropped (too short / off-graph)", len(eps), dropped)
    return eps


def save_episodes(eps: list[Episode], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(eps, f)


class _CompatUnpickler(pickle.Unpickler):
    """Resolve Episode pickled by the original research codebase (module path
    ``arctic_irl.data.dataset``) to this package, so caches produced in the
    reference environment load without it installed."""

    def find_class(self, module: str, name: str):
        if module.startswith("arctic_irl."):
            module = "arctic_meta_irl." + module[len("arctic_irl."):]
        return super().find_class(module, name)


def load_episodes(path: str | Path) -> list[Episode]:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


# --------------------------------------------------------------------------- #
# PyTorch dataset / dataloader
# --------------------------------------------------------------------------- #

class DemonstrationDataset:
    """``torch.utils.data.Dataset`` over episodes.

    Per item (one trajectory):
        phi      (T, D)  float32   state features for s_0..s_{T-1}
        actions  (T,)    int64
        states   (T,)    int64     state indices  s_0..s_{T-1}
        next_states (T,) int64     s_1..s_T
        goal     ()      int64
        mmsi     ()      int64
        length   ()      int64
    """

    def __init__(self, episodes: list[Episode], features: FeatureBuilder):
        self.episodes = episodes
        self.features = features

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int) -> dict:
        import torch
        ep = self.episodes[i]
        phi = self.features.episode_matrix(ep.states[:-1], ep.goal,
                                           ep.year, ep.month, ep.mmsi)
        return {
            "phi": torch.from_numpy(phi),
            "actions": torch.from_numpy(ep.actions.astype(np.int64)),
            "states": torch.from_numpy(ep.states[:-1].astype(np.int64)),
            "next_states": torch.from_numpy(ep.states[1:].astype(np.int64)),
            "goal": torch.tensor(ep.goal, dtype=torch.int64),
            "mmsi": torch.tensor(ep.mmsi, dtype=torch.int64),
            "length": torch.tensor(len(ep), dtype=torch.int64),
        }


def pad_collate(batch: list[dict]) -> dict:
    """Pad variable-length trajectories; returns a boolean ``mask`` (B, Tmax)."""
    import torch
    B = len(batch)
    Tmax = max(int(b["length"]) for b in batch)
    D = batch[0]["phi"].shape[1]
    A_pad = torch.zeros(B, Tmax, dtype=torch.int64)
    PHI = torch.zeros(B, Tmax, D, dtype=torch.float32)
    S = torch.zeros(B, Tmax, dtype=torch.int64)
    SN = torch.zeros(B, Tmax, dtype=torch.int64)
    M = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = int(b["length"])
        PHI[i, :t] = b["phi"]
        A_pad[i, :t] = b["actions"]
        S[i, :t] = b["states"]
        SN[i, :t] = b["next_states"]
        M[i, :t] = True
    return {
        "phi": PHI, "actions": A_pad, "states": S, "next_states": SN, "mask": M,
        "goal": torch.stack([b["goal"] for b in batch]),
        "mmsi": torch.stack([b["mmsi"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
    }


def make_dataloader(episodes: list[Episode], features: FeatureBuilder,
                    batch_size: int = 32, shuffle: bool = True,
                    num_workers: int = 0):
    from torch.utils.data import DataLoader
    ds = DemonstrationDataset(episodes, features)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=pad_collate)
