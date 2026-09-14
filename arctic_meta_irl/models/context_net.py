"""Context posterior q(z | tau) — bi-LSTM over the trajectory.

Adapted from the PEMIRL baseline in Multi-task-Hierarchical-AIRL
(model/context_net.py): a bidirectional LSTM reads the sequence of
(state-feature, one-hot action) pairs; per-step logits are average-pooled
(location invariance) into the mean of a diagonal Gaussian with a learned,
state-independent log-std. Supports padded batches via a boolean mask.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ContextPosterior(nn.Module):
    """q(z | tau). Optionally conditions the mean on a static vessel-metadata
    prior (``meta_slice``): the metadata is already inside ``seq`` (it is the
    vessel-static block of phi(s), constant over t), so when ``meta_slice`` is
    given we give it a *dedicated* encoder pathway into z, concatenated with the
    pooled bi-LSTM representation before the mean head. With ``meta_slice=None``
    the module is bit-identical to the original PEMIRL posterior."""

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int,
                 context_limit: float = 2.0,
                 meta_slice: tuple[int, int] | None = None,
                 meta_hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
        self.meta_slice = meta_slice
        if meta_slice is not None:
            md = meta_slice[1] - meta_slice[0]
            self.meta_enc = nn.Sequential(nn.Linear(md, meta_hidden), nn.Tanh())
            self.linear = nn.Linear(hidden_dim * 2 + meta_hidden, context_dim)
        else:
            self.linear = nn.Linear(hidden_dim * 2, context_dim)
        nn.init.zeros_(self.linear.bias)
        self.a_log_std = nn.Parameter(torch.zeros(1, context_dim))
        self.context_limit = context_limit

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None):
        """seq: (B, T, input_dim); mask: (B, T) bool. -> (mean, log_std)."""
        h, _ = self.lstm(seq)                       # (B, T, 2H)
        if self.meta_slice is None:
            logits = self.linear(h)                 # (B, T, Z)
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                mean = (logits * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            else:
                mean = logits.mean(dim=1)           # (B, Z)
        else:                                        # metadata-prior variant
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            else:
                pooled = h.mean(dim=1)               # (B, 2H)
            lo, hi = self.meta_slice
            meta = seq[:, 0, lo:hi]                  # (B, md) static, constant over t
            mean = self.linear(torch.cat([pooled, self.meta_enc(meta)], dim=-1))
        return mean, self.a_log_std.expand_as(mean)

    def log_prob_context(self, seq: torch.Tensor, cnt: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
        """log q(cnt | seq): (B, 1)."""
        mean, logstd = self.forward(seq, mask)
        lp = (-((cnt - mean) ** 2) / (2 * (logstd * 2).exp())
              - logstd - math.log(math.sqrt(2 * math.pi)))
        return lp.sum(dim=-1, keepdim=True)

    def sample_context(self, seq: torch.Tensor, mask: torch.Tensor | None = None,
                       fixed: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(seq, mask)
        z = mean if fixed else mean + log_std.exp() * torch.randn_like(mean)
        return torch.clamp(z, -self.context_limit, self.context_limit)
