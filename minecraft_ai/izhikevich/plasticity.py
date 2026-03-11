"""Reward-modulated spike-timing-dependent plasticity (R-STDP).

Implements three-factor learning rule:
    1. STDP eligibility traces from pre/post spike timing
    2. Eligibility traces decay over time
    3. Reward signal modulates trace -> weight update
"""

import numpy as np
from dataclasses import dataclass
from ..config import STDPConfig


class EligibilityTrace:
    """Per-synapse eligibility trace for R-STDP."""

    def __init__(self, shape: tuple, config: STDPConfig):
        self.config = config
        self.traces = np.zeros(shape, dtype=np.float64)
        self.pre_traces = np.zeros(shape[0], dtype=np.float64)  # per pre-neuron
        self.post_traces = np.zeros(shape[1], dtype=np.float64)  # per post-neuron

    def update_pre_spike(self, pre_mask: np.ndarray, dt: float):
        """Update pre-synaptic spike traces."""
        decay = np.exp(-dt / self.config.tau_plus)
        self.pre_traces *= decay
        self.pre_traces[pre_mask] += self.config.a_plus

    def update_post_spike(self, post_mask: np.ndarray, dt: float):
        """Update post-synaptic spike traces."""
        decay = np.exp(-dt / self.config.tau_minus)
        self.post_traces *= decay
        self.post_traces[post_mask] += self.config.a_minus

    def compute_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Compute STDP updates and accumulate into eligibility traces.

        LTP: post fires after pre -> potentiation (pre_trace at post spike time)
        LTD: pre fires after post -> depression (post_trace at pre spike time)
        """
        # LTP: post-synaptic spikes potentiate using pre-synaptic traces
        if np.any(post_spikes):
            ltp = np.outer(self.pre_traces, post_spikes.astype(np.float64))
            self.traces += ltp

        # LTD: pre-synaptic spikes depress using post-synaptic traces
        if np.any(pre_spikes):
            ltd = np.outer(pre_spikes.astype(np.float64), self.post_traces)
            self.traces -= ltd

    def decay(self, dt: float):
        """Decay eligibility traces over time."""
        factor = np.exp(-dt / self.config.tau_eligibility)
        self.traces *= factor

    def reset(self):
        """Clear all traces."""
        self.traces[:] = 0.0
        self.pre_traces[:] = 0.0
        self.post_traces[:] = 0.0


class RewardModulatedSTDP:
    """Reward-modulated STDP learning rule for SNN weight updates."""

    def __init__(self, weight_shape: tuple, config: STDPConfig):
        self.config = config
        self.eligibility = EligibilityTrace(weight_shape, config)
        self._reward_baseline = 0.0
        self._baseline_alpha = 0.01

    def on_spikes(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float):
        """Process spike events and update eligibility traces."""
        self.eligibility.update_pre_spike(pre_spikes, dt)
        self.eligibility.update_post_spike(post_spikes, dt)
        self.eligibility.compute_stdp(pre_spikes, post_spikes)
        self.eligibility.decay(dt)

    def apply_reward(
        self, weights: np.ndarray, reward: float,
        min_weight: float = 0.0, max_weight: float = 10.0
    ) -> np.ndarray:
        """Apply reward-modulated weight update.

        Uses reward prediction error (reward - baseline) to modulate
        eligibility traces into actual weight changes.
        """
        # Reward prediction error
        rpe = reward - self._reward_baseline
        self._reward_baseline += self._baseline_alpha * rpe

        # Weight update: dW = lr * RPE * eligibility - decay * W
        lr = self.config.reward_learning_rate
        dw = lr * rpe * self.eligibility.traces
        dw -= self.config.weight_decay * weights

        weights += dw
        np.clip(weights, min_weight, max_weight, out=weights)

        return weights

    def reset(self):
        """Reset learning state."""
        self.eligibility.reset()
        self._reward_baseline = 0.0
