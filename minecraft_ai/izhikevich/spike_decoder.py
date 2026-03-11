"""Motor spike rates -> action selection for Izhikevich brains."""

import numpy as np
from typing import Dict, List, Tuple
from ..environment.mc_actions import ActionType, NUM_ACTIONS


# Output neuron group assignments (48 output neurons -> 9 action groups)
# Each action gets ~5 output neurons; their firing rates are averaged
OUTPUT_GROUP_RANGES: Dict[ActionType, Tuple[int, int]] = {
    ActionType.MOVE_FORWARD:  (0, 5),
    ActionType.MOVE_BACKWARD: (5, 10),
    ActionType.STRAFE_LEFT:   (10, 15),
    ActionType.STRAFE_RIGHT:  (15, 20),
    ActionType.LOOK_LEFT:     (20, 25),
    ActionType.LOOK_RIGHT:    (25, 30),
    ActionType.ATTACK:        (30, 36),   # 6 neurons for attack (higher resolution)
    ActionType.USE_ITEM:      (36, 42),   # 6 neurons for use_item
    ActionType.JUMP_SNEAK:    (42, 48),   # 6 neurons for jump/sneak
}


class SpikeDecoder:
    """Decode output layer spike rates into discrete and continuous actions."""

    def __init__(self, temperature: float = 1.0, threshold: float = 0.1):
        self.temperature = temperature
        self.threshold = threshold
        self._group_ranges = OUTPUT_GROUP_RANGES

    def decode_rates(self, firing_rates: np.ndarray) -> Tuple[int, np.ndarray]:
        """Convert output neuron firing rates to action selection.

        Args:
            firing_rates: Array of firing rates for output neurons (48,)

        Returns:
            (action_index, action_probabilities)
        """
        action_strengths = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for action_type, (start, end) in self._group_ranges.items():
            group_rates = firing_rates[start:end]
            action_strengths[action_type.value] = np.mean(group_rates)

        # Softmax with temperature
        action_probs = self._softmax(action_strengths)
        action_idx = int(np.argmax(action_probs))

        return action_idx, action_probs

    def decode_multi_action(self, firing_rates: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Decode firing rates allowing multiple simultaneous actions.

        Returns actions whose average rate exceeds the threshold.
        """
        action_strengths = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for action_type, (start, end) in self._group_ranges.items():
            group_rates = firing_rates[start:end]
            action_strengths[action_type.value] = np.mean(group_rates)

        active_actions = [
            i for i in range(NUM_ACTIONS) if action_strengths[i] > self.threshold
        ]
        if not active_actions:
            active_actions = [int(np.argmax(action_strengths))]

        return active_actions, action_strengths

    def add_aggression_bias(
        self, firing_rates: np.ndarray, bias: float
    ) -> np.ndarray:
        """Add bias current to attack-related output neurons."""
        rates = firing_rates.copy()
        attack_start, attack_end = self._group_ranges[ActionType.ATTACK]
        rates[attack_start:attack_end] += bias
        return rates

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Temperature-scaled softmax."""
        scaled = x / max(self.temperature, 1e-8)
        shifted = scaled - np.max(scaled)
        exp_x = np.exp(shifted)
        return exp_x / (np.sum(exp_x) + 1e-8)
