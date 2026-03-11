"""Spike counts -> action logits decoder (linear readout).

Takes spike count responses from CL1 channel groups and maps them
to action logits via a lightweight linear readout layer.
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..config import DecoderConfig


class MCDecoder(nn.Module):
    """Decodes CL1 spike counts into action logits.

    Architecture:
        spike_counts (48 or 10) -> Linear -> LayerNorm -> Linear -> action_logits (9)

    Designed for minimal transformation - the CL1 hardware does
    the heavy computational lifting, this just reads out the result.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.readout = nn.Sequential(
            nn.Linear(config.spike_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions),
        )

    def forward(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """Decode spike counts to action logits.

        Args:
            spike_counts: (batch, spike_input_dim) spike counts from CL1

        Returns:
            action_logits: (batch, num_actions) raw logits
        """
        return self.readout(spike_counts)

    def get_action_probs(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """Get softmax action probabilities."""
        logits = self.forward(spike_counts)
        return torch.softmax(logits, dim=-1)

    def get_action(self, spike_counts: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Get greedy action and probabilities.

        Returns:
            (action_index, action_probabilities)
        """
        probs = self.get_action_probs(spike_counts)
        action = int(torch.argmax(probs, dim=-1).item())
        return action, probs


class ChannelGroupDecoder(nn.Module):
    """Decoder that operates on per-channel-group spike counts.

    Simpler variant that takes the 10 channel group spike counts
    directly (9 action groups + encoding) and maps to action logits.
    """

    def __init__(self, num_groups: int = 10, num_actions: int = 9):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_groups, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, group_counts: torch.Tensor) -> torch.Tensor:
        """Decode channel group spike counts to action logits.

        Args:
            group_counts: (batch, 10) spike counts per channel group

        Returns:
            action_logits: (batch, 9) raw logits
        """
        return self.decoder(group_counts)
