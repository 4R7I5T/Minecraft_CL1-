"""State value estimation network for PPO training."""

import torch
import torch.nn as nn


class MCValueNet(nn.Module):
    """Estimates state value V(s) for PPO advantage computation.

    Architecture:
        obs (59) -> Linear+SiLU -> Linear+SiLU -> Linear -> scalar value
    """

    def __init__(self, obs_dim: int = 59, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            obs: (batch, 59) observation tensor

        Returns:
            value: (batch, 1) state value estimates
        """
        return self.net(obs)

    def get_value(self, obs: torch.Tensor) -> float:
        """Get scalar value for a single observation."""
        with torch.no_grad():
            return self.forward(obs.unsqueeze(0)).item()
