"""Observation -> stimulation parameter encoder.

Maps 59-dim Minecraft observations to per-channel-group stimulation
parameters using Beta distribution heads and SiLU activations.
Each channel group gets (alpha, beta) parameters for frequency and amplitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..config import EncoderConfig


class MCEncoder(nn.Module):
    """Encodes observations into CL1 stimulation parameters.

    Architecture:
        obs (59) -> Linear+SiLU -> Linear+SiLU -> per-group Beta heads

    Each of the 10 channel groups gets:
        - frequency_alpha, frequency_beta (Beta distribution params)
        - amplitude_alpha, amplitude_beta
    Output shape: (batch, 10, 4) reshaped from (batch, 40)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.shared = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Per-group output heads: 4 params each (freq_alpha, freq_beta, amp_alpha, amp_beta)
        self.group_heads = nn.Linear(
            config.hidden_dim,
            config.num_channel_groups * 4,
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to stimulation parameters.

        Args:
            obs: (batch, 59) observation tensor

        Returns:
            frequencies: (batch, 10) stimulation frequencies
            amplitudes: (batch, 10) stimulation amplitudes
        """
        h = self.shared(obs)
        raw = self.group_heads(h)  # (batch, 40)

        # Reshape to (batch, 10, 4)
        params = raw.view(-1, self.config.num_channel_groups, 4)

        # Softplus to ensure positive Beta params (minimum 1.0 for valid Beta)
        alpha_freq = F.softplus(params[:, :, 0]) + 1.0
        beta_freq = F.softplus(params[:, :, 1]) + 1.0
        alpha_amp = F.softplus(params[:, :, 2]) + 1.0
        beta_amp = F.softplus(params[:, :, 3]) + 1.0

        # Beta distribution mean = alpha / (alpha + beta), range [0, 1]
        freq_mean = alpha_freq / (alpha_freq + beta_freq)
        amp_mean = alpha_amp / (alpha_amp + beta_amp)

        # Scale to hardware ranges
        frequencies = freq_mean * 50.0   # 0-50 Hz
        amplitudes = amp_mean * 800.0    # 0-800 mV

        return frequencies, amplitudes

    def get_beta_params(self, obs: torch.Tensor) -> torch.Tensor:
        """Get raw Beta distribution parameters for analysis.

        Returns:
            (batch, 10, 4) tensor of [alpha_f, beta_f, alpha_a, beta_a] per group
        """
        h = self.shared(obs)
        raw = self.group_heads(h)
        params = raw.view(-1, self.config.num_channel_groups, 4)
        return F.softplus(params) + 1.0
