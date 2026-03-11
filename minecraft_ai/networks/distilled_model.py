"""Lightweight obs->action MLP for runtime inference.

This model replaces CL1 hardware at runtime after distillation.
It's a simple MLP trained to mimic the CL1's input-output mapping.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class DistilledModel(nn.Module):
    """Compact MLP distilled from CL1 hardware responses.

    Architecture:
        obs (59) -> [Linear+SiLU] * N -> Linear -> action_logits (9)

    Default: 59 -> 128 -> 64 -> 9
    """

    def __init__(
        self,
        obs_dim: int = 59,
        hidden_dims: List[int] = None,
        num_actions: int = 9,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_actions))

        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            obs: (batch, obs_dim) or (obs_dim,) observation tensor

        Returns:
            logits: (batch, num_actions) action logits
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)

    def get_action(self, obs: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Get greedy action from observation.

        Returns:
            (action_index, action_probabilities)
        """
        with torch.no_grad():
            logits = self.forward(obs)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            action = int(np.argmax(probs))
        return action, probs

    def get_action_from_numpy(self, obs_np: np.ndarray) -> Tuple[int, np.ndarray]:
        """Convenience: numpy observation in, action out."""
        obs_tensor = torch.from_numpy(obs_np).float()
        return self.get_action(obs_tensor)
