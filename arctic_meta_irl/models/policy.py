"""Action-masked categorical policy + value critic, context-conditionable.

pi(a | s, z): input = [phi(s), z]; invalid graph actions are masked to -inf
before the softmax. Used as the PEMIRL generator and the PPO baseline
(baseline uses context_dim=0).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .discriminator import mlp

NEG_INF = -1e9


class MaskedPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, context_dim: int = 0,
                 hidden=(256, 256)):
        super().__init__()
        self.context_dim = context_dim
        in_dim = obs_dim + context_dim
        self.pi = mlp([in_dim, *hidden, n_actions])
        self.v = mlp([in_dim, *hidden, 1])

    def _x(self, obs: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if self.context_dim:
            assert z is not None
            return torch.cat([obs, z], dim=-1)
        return obs

    def dist(self, obs, mask, z=None) -> Categorical:
        logits = self.pi(self._x(obs, z))
        logits = torch.where(mask.bool(), logits, torch.full_like(logits, NEG_INF))
        return Categorical(logits=logits)

    def act(self, obs, mask, z=None, deterministic: bool = False):
        d = self.dist(obs, mask, z)
        a = d.probs.argmax(dim=-1) if deterministic else d.sample()
        return a, d.log_prob(a), self.value(obs, z)

    def log_prob_action(self, obs, mask, act, z=None) -> torch.Tensor:
        return self.dist(obs, mask, z).log_prob(act.long()).unsqueeze(-1)

    def entropy(self, obs, mask, z=None) -> torch.Tensor:
        return self.dist(obs, mask, z).entropy()

    def value(self, obs, z=None) -> torch.Tensor:
        return self.v(self._x(obs, z)).squeeze(-1)
