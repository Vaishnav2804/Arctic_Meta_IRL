"""PEMIRL + PPO baseline smoke tests on the synthetic MDP (CPU)."""
import numpy as np
import torch

from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.algos.ppo import PPO
from arctic_meta_irl.algos.sampler import Sampler
from arctic_meta_irl.env.gcrl_env import GCRLNavEnv, tasks_from_episodes
from arctic_meta_irl.eval.rollout import (_episode_tensors,
                                          pemirl_log_likelihood)
from arctic_meta_irl.models.policy import MaskedPolicy

PEM_CFG = {"context_dim": 4, "context_limit": 2.0, "bi_lstm_hidden": 32,
           "disc_hidden": [64, 64], "policy_hidden": [64, 64],
           "info_coeff": 0.1, "info_training_iters": 1,
           "cnt_training_iterations": 1, "cnt_starting_iter": 0,
           "context_repeat_num": 2, "mini_batch_size": 256,
           "optimizer_lr_discriminator": 3e-4, "optimizer_lr_policy": 3e-4,
           "grad_penalty": 1.0, "state_only": False}


def test_pemirl_two_outer_iters(pipeline, mdp, tmp_path):
    torch.manual_seed(0)
    fb, eps = pipeline
    model = PEMIRL(fb.dim, mdp.n_actions, PEM_CFG, device="cpu")
    env = GCRLNavEnv(mdp, fb, max_horizon=24)
    sampler = Sampler(env)
    ppo = PPO(model.policy, mini_batch_size=256, epochs=2)
    demo = [_episode_tensors(ep, fb) for ep in eps[:8]]
    tasks = tasks_from_episodes(eps[:8])

    R = PEM_CFG["context_repeat_num"]
    for it in range(2):
        z = model.sample_prior(2).repeat_interleave(R, dim=0)
        batch = [tasks[i] for i in (0, 1) for _ in range(R)]
        ros = sampler.collect(model.policy, batch, z)
        stats = model.step(ros, demo, training_itr=it, n_step=1)
        assert np.isfinite(stats["disc_loss"])

        def airl(ro):
            t = ro.tensors("cpu")
            zz = ro.z.unsqueeze(0).expand(len(ro), -1)
            return model.airl_reward(t["obs"], t["actions"], zz)
        ps = ppo.step(ros, reward_override=airl)
        assert np.isfinite(ps["pi_loss"])

    # support-conditioned LL + checkpoint round-trip
    ll, n = pemirl_log_likelihood(model, eps, fb, mdp, support_size=2)
    assert n > 0 and np.isfinite(ll)
    p = tmp_path / "pem.pt"
    model.save(p)
    m2 = PEMIRL(fb.dim, mdp.n_actions, PEM_CFG, device="cpu")
    m2.load(p)
    for a, b in zip(model.policy.parameters(), m2.policy.parameters()):
        assert torch.allclose(a, b)


def test_ppo_baseline_handreward(pipeline, mdp):
    torch.manual_seed(0)
    fb, eps = pipeline
    env = GCRLNavEnv(mdp, fb, max_horizon=24,
                     hand_reward_cfg={"w_dist": 1.0, "w_ice": 2.0,
                                      "w_wind": 0.5, "goal_bonus": 10.0,
                                      "step_penalty": 0.01})
    policy = MaskedPolicy(fb.dim, mdp.n_actions, hidden=(64, 64))
    sampler = Sampler(env)
    ppo = PPO(policy, mini_batch_size=256, epochs=2)
    tasks = tasks_from_episodes(eps[:6])
    ros = sampler.collect(policy, tasks)
    assert all(len(ro) > 0 for ro in ros)
    stats = ppo.step(ros)
    assert np.isfinite(stats["pi_loss"]) and np.isfinite(stats["v_loss"])


def test_eval_metrics(pipeline, mdp):
    from arctic_meta_irl.eval.metrics import route_metrics
    fb, eps = pipeline
    ep = eps[0]
    m = route_metrics(mdp, list(ep.states), list(ep.states))
    assert m["hausdorff_km"] == 0.0
    assert m["cell_overlap"] == 1.0
    assert abs(m["length_ratio"] - 1.0) < 1e-9


def test_heterogeneity(pipeline, mdp, cfg):
    from arctic_meta_irl.data.loaders import load_vessel_registry
    from arctic_meta_irl.eval.heterogeneity import (behavior_descriptors,
                                                    eta_squared_table)
    fb, eps = pipeline
    reg = load_vessel_registry(cfg["paths"]["vessel_registry"])
    df = behavior_descriptors(eps, mdp, reg)
    assert len(df) == len(eps)
    table = eta_squared_table(df)
    assert ((table >= 0) & (table <= 1 + 1e-9)).all()
