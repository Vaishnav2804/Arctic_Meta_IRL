"""Action-masked sequence policies pi(a_t | phi(s_1..t), a_1..t-1).

Supervised sequence baselines ("modern deep learning baseline" arm): unlike
the MLP policies (BC / AIRL / PEMIRL generator), these condition on the
trajectory's own past through a recurrent (LSTM) or causal-attention
(Transformer) backbone. Input per step is [phi(s_t), onehot(a_{t-1})] with a
zero vector for a_0's predecessor; outputs are per-step action logits masked
to the valid graph actions before the softmax.

Teacher-forced training/evaluation factorizes the exact per-decision
log-likelihood, so the metric is directly comparable to the MLP policy LLs
(the sequence models simply see strictly more information — the episode's own
prefix — which is the point of the baseline: within-trajectory context vs.
PEMIRL's cross-trajectory support-set context).
"""
from __future__ import annotations

import torch
from torch import nn

NEG_INF = -1e9


class SequencePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, arch: str = "lstm",
                 hidden: int = 128, layers: int = 1, nhead: int = 4,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        assert arch in ("lstm", "transformer"), arch
        self.arch = arch
        self.n_actions = n_actions
        in_dim = obs_dim + n_actions          # phi(s_t) + onehot(a_{t-1})
        if arch == "lstm":
            self.backbone = nn.LSTM(in_dim, hidden, num_layers=layers,
                                    batch_first=True,
                                    dropout=dropout if layers > 1 else 0.0)
        else:
            self.in_proj = nn.Linear(in_dim, hidden)
            self.pos = nn.Embedding(max_len, hidden)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=4 * hidden,
                dropout=dropout, batch_first=True, norm_first=True)
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(hidden, n_actions)
        self.max_len = max_len

    def _inputs(self, phi: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """[phi_t, onehot(a_{t-1})]; a_{-1} = zero vector. phi (B,T,D), actions (B,T)."""
        a1h = torch.nn.functional.one_hot(actions.long(), self.n_actions).float()
        prev = torch.zeros_like(a1h)
        prev[:, 1:] = a1h[:, :-1]
        return torch.cat([phi, prev], dim=-1)

    def logits(self, phi: torch.Tensor, actions: torch.Tensor,
               action_mask: torch.Tensor) -> torch.Tensor:
        """Per-step masked logits (B, T, A), causal in t (teacher-forced)."""
        x = self._inputs(phi, actions)
        if self.arch == "lstm":
            h, _ = self.backbone(x)
        else:
            T = x.size(1)
            pos = self.pos(torch.arange(T, device=x.device))
            h = self.in_proj(x) + pos.unsqueeze(0)
            causal = nn.Transformer.generate_square_subsequent_mask(
                T, device=x.device)
            h = self.backbone(h, mask=causal, is_causal=True)
        logits = self.head(h)
        return torch.where(action_mask.bool(), logits,
                           torch.full_like(logits, NEG_INF))

    def log_probs(self, phi: torch.Tensor, actions: torch.Tensor,
                  action_mask: torch.Tensor) -> torch.Tensor:
        """Teacher-forced log pi(a_t | .) per step: (B, T)."""
        logits = self.logits(phi, actions, action_mask)
        logp = torch.log_softmax(logits, dim=-1)
        return logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
