"""PEMIRL stability knobs (configs/pemirl.yaml) are wired into the module.

The knobs were added after a 500-iter run diverged (unbounded f in the
info-max term -> disc loss ~1e11). Assert the production values reach the
module attributes and shape the AIRL reward path.
"""
import numpy as np
import torch

from arctic_meta_irl.algos.pemirl import PEMIRL

OBS_DIM, N_ACT, Z_DIM = 6, 5, 4

# toy-sized nets + the production stability knobs from configs/pemirl.yaml
STAB_CFG = {"context_dim": Z_DIM, "context_limit": 2.0, "bi_lstm_hidden": 16,
            "disc_hidden": [32, 32], "policy_hidden": [32, 32],
            "info_training_iters": 1, "cnt_training_iterations": 1,
            "context_repeat_num": 2, "mini_batch_size": 64,
            "optimizer_lr_discriminator": 3e-4, "optimizer_lr_policy": 3e-4,
            "grad_penalty": 1.0, "state_only": False,
            # ---- stability knobs under test ----
            "f_clamp": 10.0, "disc_grad_clip": 1.0,
            "reward_logit": True, "reward_norm": True,
            "info_coeff": 0.01, "cnt_starting_iter": 10}


def _model(**overrides) -> PEMIRL:
    torch.manual_seed(0)
    return PEMIRL(OBS_DIM, N_ACT, {**STAB_CFG, **overrides}, device="cpu")


def _batch(n=128, scale=1.0):
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(n, OBS_DIM, generator=g) * scale
    act = torch.randint(0, N_ACT, (n,), generator=g)
    z = torch.randn(n, Z_DIM, generator=g)
    return obs, act, z


def test_knobs_wired_to_attributes():
    m = _model()
    assert m.f_clamp == 10.0
    assert m.disc_grad_clip == 1.0
    assert m.reward_logit is True
    assert m.reward_norm is True
    assert np.isclose(m.info_coeff, 0.01)
    assert m.cnt_starting_iter == 10


def test_airl_reward_standardized_per_batch():
    # reward_norm=True: PPO consumes a per-batch standardized reward
    m = _model()
    r = m.airl_reward(*_batch())
    assert torch.isfinite(r).all()
    assert abs(float(r.mean())) < 1e-4
    assert abs(float(r.std()) - 1.0) < 1e-2


def test_airl_reward_clamped_to_f_clamp():
    # reward_logit without normalization: |r| = |clamp(f)| <= f_clamp even
    # when the raw discriminator output is made to explode
    m = _model(reward_norm=False)
    with torch.no_grad():
        for p in m.discriminator.parameters():
            p.mul_(100.0)
    r = m.airl_reward(*_batch(scale=10.0))
    assert torch.isfinite(r).all()
    assert float(r.abs().max()) <= m.f_clamp + 1e-5
    assert float(r.abs().max()) >= m.f_clamp - 1e-4  # the clamp actually bit


def test_reward_logit_toggle():
    # reward_logit=False falls back to the released-code sigmoid(f) in (0, 1)
    m = _model(reward_logit=False, reward_norm=False)
    r = m.airl_reward(*_batch())
    assert ((r > 0) & (r < 1)).all()
    # logit mode is NOT sigmoid-bounded
    m2 = _model(reward_logit=True, reward_norm=False)
    with torch.no_grad():
        for p in m2.discriminator.parameters():
            p.mul_(100.0)
    r2 = m2.airl_reward(*_batch(scale=10.0))
    assert float(r2.abs().max()) > 1.0


def test_disc_grad_clip_is_positive_float():
    m = _model()
    assert isinstance(m.disc_grad_clip, float) and m.disc_grad_clip > 0
    # the clip is applied to the discriminator optimizer's parameters —
    # ensure the optimizer actually owns those parameters
    disc_params = {id(p) for p in m.discriminator.parameters()}
    opt_params = {id(p) for grp in m.optim.param_groups for p in grp["params"]}
    assert disc_params <= opt_params
