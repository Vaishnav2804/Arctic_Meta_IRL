"""Tests for MAML_IRL algorithm module."""
import torch
from arctic_meta_irl.algos.maml_irl import MAML_IRL


def test_maml_irl_init():
    cfg = {
        "disc_hidden": [64, 64],
        "policy_hidden": [64, 64],
        "inner_lr": 0.01,
        "inner_steps": 1,
        "first_order": True,
        "mini_batch_size": 32,
    }
    obs_dim = 20
    n_actions = 7

    algo = MAML_IRL(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, device="cpu")
    assert algo.discriminator is not None
    assert algo.policy is not None


def test_maml_irl_inner_adaptation():
    cfg = {
        "disc_hidden": [32, 32],
        "policy_hidden": [32, 32],
        "inner_lr": 0.01,
        "inner_steps": 1,
        "first_order": True,
    }
    algo = MAML_IRL(obs_dim=20, n_actions=7, cfg=cfg, device="cpu")

    supp_obs = torch.randn(10, 20)
    supp_act = torch.randint(0, 7, (10,))
    policy_obs = torch.randn(10, 20)
    policy_act = torch.randint(0, 7, (10,))

    adapted_disc = algo.adapt_vessel_discriminator(
        supp_obs, supp_act, policy_obs, policy_act
    )

    assert adapted_disc is not None
    # Verify adapted weights differ from original
    orig_param = next(algo.discriminator.parameters()).data
    adapted_param = next(adapted_disc.parameters()).data
    assert not torch.allclose(orig_param, adapted_param)
