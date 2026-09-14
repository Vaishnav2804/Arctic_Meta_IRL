"""WP0 guards: the pooled-AIRL ablation path (context_dim=0) and the k=0
prior fallback in support-set context inference."""
import torch

from arctic_meta_irl.algos.pemirl import PEMIRL
from arctic_meta_irl.algos.sampler import Rollout

OBS_DIM, N_ACTIONS = 6, 5

BASE_CFG = {
    "context_dim": 4, "context_limit": 2.0, "bi_lstm_hidden": 16,
    "cnt_sampling_fixed": False, "disc_hidden": [16, 16], "state_only": False,
    "grad_penalty": 1.0, "optimizer_lr_discriminator": 3.0e-4,
    "weight_decay": 3.0e-5, "info_coeff": 0.01, "info_training_iters": 1,
    "cnt_training_iterations": 1, "cnt_starting_iter": 0,
    "context_repeat_num": 2, "policy_hidden": [16, 16],
    "optimizer_lr_policy": 3.0e-4, "ppo_clip": 0.2, "ppo_epochs": 1,
    "gae_lambda": 0.95, "entropy_coef": 0.01, "value_coef": 0.5,
    "mini_batch_size": 64, "support_set_size": 3,
    "f_clamp": 10.0, "disc_grad_clip": 1.0,
    "reward_logit": True, "reward_norm": True,
}


def _rollout(model, T=7):
    obs = torch.randn(T, OBS_DIM)
    act = torch.randint(0, N_ACTIONS, (T,))
    z = model.sample_prior(1).squeeze(0)
    return Rollout(obs=list(obs.numpy()), actions=act.tolist(),
                   masks=[m.numpy() for m in
                          torch.ones(T, N_ACTIONS, dtype=torch.bool)],
                   logps=[0.0] * T, values=[0.0] * T, rewards=[0.0] * T,
                   states=list(range(T)), reached=False, z=z)


def _demo(T=7):
    return [(torch.randn(T, OBS_DIM), torch.randint(0, N_ACTIONS, (T,)))
            for _ in range(4)]


def test_noctx_construction_and_step():
    """context_dim=0 (pooled AIRL) builds, steps, and rewards without error."""
    cfg = {**BASE_CFG, "context_dim": 0}
    model = PEMIRL(obs_dim=OBS_DIM, n_actions=N_ACTIONS, cfg=cfg, device="cpu")
    assert model.dim_cnt == 0
    assert model.sample_prior(3).shape == (3, 0)

    rollouts = [_rollout(model) for _ in range(4)]
    stats = model.step(rollouts, _demo(), training_itr=5)
    # posterior / info-max branches must be skipped entirely
    assert "cnt_loss" not in stats and "info_loss" not in stats
    assert "disc_loss" in stats

    obs, act = _demo(T=5)[0]
    r = model.airl_reward(obs, act, torch.zeros(5, 0))
    assert r.shape[0] == 5 and torch.isfinite(r).all()


def test_ctx_model_still_trains_posterior():
    """Sanity: with context_dim>0 the posterior/info-max branches still run."""
    model = PEMIRL(obs_dim=OBS_DIM, n_actions=N_ACTIONS, cfg=BASE_CFG,
                   device="cpu")
    rollouts = [_rollout(model) for _ in range(4)]
    stats = model.step(rollouts, _demo(), training_itr=5)
    assert "cnt_loss" in stats


def test_empty_support_prior_fallback():
    """k=0 support (no-adaptation arm): prior mean z=0 instead of a crash."""
    model = PEMIRL(obs_dim=OBS_DIM, n_actions=N_ACTIONS, cfg=BASE_CFG,
                   device="cpu")
    z = model.infer_context_support([])
    assert z.shape == (1, BASE_CFG["context_dim"])
    assert torch.all(z == 0)
