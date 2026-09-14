"""Trajectory collection in the GCRL environment (the PEMIRL/PPO "sampler")."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..env.gcrl_env import GCRLNavEnv, Task


@dataclass
class Rollout:
    """A single on-policy trajectory.

    Note: ``states`` records the graph state *before* each step, so it has
    length T (the terminal state is not appended), unlike demonstration
    episodes whose state arrays have length T+1.
    """

    obs: list = field(default_factory=list)        # (T, D) float32
    actions: list = field(default_factory=list)    # (T,)
    masks: list = field(default_factory=list)      # (T, A) bool
    logps: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    states: list = field(default_factory=list)     # graph state ids
    z: torch.Tensor | None = None                  # context used (PEMIRL)
    task: Task | None = None
    reached: bool = False

    def tensors(self, device="cpu") -> dict[str, torch.Tensor]:
        return dict(
            obs=torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.array(self.actions), dtype=torch.long, device=device),
            masks=torch.as_tensor(np.array(self.masks), dtype=torch.bool, device=device),
            logps=torch.as_tensor(np.array(self.logps), dtype=torch.float32, device=device),
            values=torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device),
            rewards=torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.actions)


class Sampler:
    def __init__(self, env: GCRLNavEnv, device: str = "cpu"):
        self.env = env
        self.device = device

    @torch.no_grad()
    def collect(self, policy, tasks: list[Task], z_per_task=None,
                deterministic: bool = False) -> list[Rollout]:
        """Roll the policy on each task; z_per_task: optional (N, Z) tensor."""
        out = []
        for i, task in enumerate(tasks):
            ro = Rollout(task=task)
            z = None
            if z_per_task is not None:
                z = z_per_task[i:i + 1].to(self.device)
                ro.z = z.squeeze(0).cpu()
            obs = self.env.reset(task)
            done = False
            while not done:
                mask = self.env.action_mask()
                o = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                m = torch.as_tensor(mask, dtype=torch.bool,
                                    device=self.device).unsqueeze(0)
                a, logp, v = policy.act(o, m, z, deterministic=deterministic)
                ro.obs.append(obs)
                ro.masks.append(mask)
                ro.actions.append(int(a.item()))
                ro.logps.append(float(logp.item()))
                ro.values.append(float(v.item()))
                ro.states.append(self.env.s)
                obs, r, done, info = self.env.step(int(a.item()))
                ro.rewards.append(r)
            ro.reached = (self.env.s == task.goal)
            out.append(ro)
        return out
