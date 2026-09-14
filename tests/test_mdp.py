"""GraphMDP construction, action semantics, episode mapping, features."""
import numpy as np

from arctic_meta_irl.data.dataset import build_episodes
from arctic_meta_irl.data.loaders import load_voyages


def test_mdp_shapes(mdp):
    S, A = mdp.next_state.shape
    assert S == 40 and A >= 2
    assert (mdp.next_state[:, 0] == np.arange(S)).all()  # action 0 = stay
    assert mdp.action_mask[:, 0].all()


def test_transitions_symmetric(mdp):
    for s in range(mdp.n_states):
        for a in range(1, mdp.n_actions):
            ns = mdp.next_state[s, a]
            if ns < 0:
                continue
            assert s in mdp.next_state[ns], "graph edges must be bidirectional"


def test_action_for_transition(mdp):
    s = 0
    ns = int(mdp.neighbors(s)[0])
    a = mdp.action_for_transition(s, ns)
    assert a is not None and mdp.next_state[s, a] == ns


def test_save_load_roundtrip(mdp, tmp_path):
    from arctic_meta_irl.env.graph_mdp import GraphMDP
    p = tmp_path / "mdp.npz"
    mdp.save(p)
    m2 = GraphMDP.load(p)
    assert m2.cells == mdp.cells
    assert (m2.next_state == mdp.next_state).all()


def test_episodes_valid_transitions(cfg, mdp):
    voyages = load_voyages(cfg["paths"]["voyages_pkl"])
    eps = build_episodes(voyages, mdp, cfg)
    assert len(eps) > 0
    for ep in eps:
        assert len(ep.states) == len(ep.actions) + 1
        for t, a in enumerate(ep.actions):
            assert mdp.next_state[ep.states[t], a] == ep.states[t + 1]
        assert ep.goal == ep.states[-1]


def test_features(pipeline, mdp):
    fb, eps = pipeline
    ep = eps[0]
    phi = fb.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month, ep.mmsi)
    assert phi.shape == (len(ep), fb.dim)
    assert np.isfinite(phi).all()
    assert len(fb.names) == fb.dim


def test_dataloader(pipeline):
    import torch
    from arctic_meta_irl.data.dataset import make_dataloader
    fb, eps = pipeline
    dl = make_dataloader(eps, fb, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    assert batch["phi"].dim() == 3 and batch["phi"].shape[0] == 4
    assert batch["mask"].dtype == torch.bool
    # padding mask covers exactly `length` steps
    assert (batch["mask"].sum(dim=1) == batch["length"]).all()
