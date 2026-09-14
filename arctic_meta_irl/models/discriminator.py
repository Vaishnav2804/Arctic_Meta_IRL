"""Context-conditioned AIRL discriminator f(s, a, z).

f doubles as the learned reward (PEMIRL: the discriminator output is the
context-conditioned reward signal). Includes a WGAN-style gradient penalty,
as in the reference implementation.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, context_dim: int,
                 hidden=(256, 256), state_only: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.state_only = state_only
        in_dim = obs_dim + context_dim + (0 if state_only else n_actions)
        self.f = mlp([in_dim, *hidden, 1])

    def _inp(self, obs, act, z):
        parts = [obs, z]
        if not self.state_only:
            a1h = torch.nn.functional.one_hot(act.long(), self.n_actions).float()
            parts.insert(1, a1h)
        return torch.cat(parts, dim=-1)

    def get_unnormed_d(self, obs: torch.Tensor, act: torch.Tensor,
                       z: torch.Tensor) -> torch.Tensor:
        """f(s,a,z): (N, 1)."""
        return self.f(self._inp(obs, act, z))

    def reward(self, obs, act, z) -> torch.Tensor:
        """AIRL reward used by the generator: D = sigma(f); r = log D - log(1-D) = f.
        Following the reference implementation we use D = exp(f)/(exp(f)+1)."""
        with torch.no_grad():
            f = self.get_unnormed_d(obs, act, z)
            return torch.sigmoid(f)

    def gradient_penalty(self, obs, act, z, lam: float = 10.0) -> torch.Tensor:
        x = self._inp(obs, act, z).detach().requires_grad_(True)
        out = self.f(x)
        grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        return lam * ((grad.norm(2, dim=-1) - 1.0) ** 2).mean()
