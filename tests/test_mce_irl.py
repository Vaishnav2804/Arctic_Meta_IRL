"""MCE-IRL: soft VI sanity + training improves data likelihood."""
import numpy as np

from arctic_meta_irl.algos.mce_irl import (MCEIRL, expected_visitation,
                                           soft_value_iteration)


def test_soft_vi_prefers_reward(mdp):
    goal = mdp.n_states - 1
    r = np.zeros(mdp.n_states)
    r[goal] = 5.0
    vi = soft_value_iteration(mdp, r, goal, horizon=30)
    assert np.isfinite(vi.log_pi[np.isfinite(vi.log_pi)]).all()
    # occupancy must reach the goal from state 0 within the horizon
    D = expected_visitation(mdp, vi.log_pi, start=0, goal=goal, horizon=30)
    assert D.sum() > 0


def test_mce_irl_improves_ll(pipeline, mdp):
    fb, eps = pipeline
    model = MCEIRL(mdp, fb, lr=0.1, seed=0)
    ll0, n = model.log_likelihood(eps)
    model.fit(eps, n_iters=15, log_every=100)
    ll1, _ = model.log_likelihood(eps)
    assert ll1 > ll0, f"LL should improve: {ll0/n:.4f} -> {ll1/n:.4f}"


def test_mce_irl_route_and_io(pipeline, mdp, tmp_path):
    fb, eps = pipeline
    model = MCEIRL(mdp, fb, lr=0.1, seed=0)
    model.fit(eps, n_iters=5, log_every=100)
    ep = eps[0]
    path = model.greedy_route(int(ep.states[0]), int(ep.goal),
                              ep.year, ep.month, ep.mmsi, horizon=64)
    assert path[0] == ep.states[0] and len(path) >= 1
    p = tmp_path / "theta.npz"
    model.save(p)
    theta = model.theta.copy()
    model.theta[:] = 0
    model.load(p)
    assert np.allclose(model.theta, theta)
