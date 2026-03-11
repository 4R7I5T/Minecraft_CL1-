"""Izhikevich SNN brain for hostile mobs.

Each hostile mob (zombie, skeleton, creeper, spider) gets a dedicated
spiking neural network with mob-specific neuron parameters driving
distinct behavioral patterns through neural dynamics.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional

from .base_brain import BaseBrain, BrainAction, BrainObservation
from ..config import IzhikevichConfig, STDPConfig
from ..izhikevich.network import IzhikevichNetwork
from ..izhikevich.mob_profiles import MobProfile, get_mob_profile
from ..izhikevich.spike_decoder import SpikeDecoder

logger = logging.getLogger(__name__)


class IzhikevichBrain(BaseBrain):
    """Spiking neural network brain for hostile mob AI.

    Uses Izhikevich neuron model with reward-modulated STDP learning.
    Mob-specific neuron parameters create emergent behavioral differences:
        - Zombie (RS): steady pursuit
        - Skeleton (CH): burst-pause aiming
        - Creeper (IB): buildup-to-explosion
        - Spider (FS): rapid reactions
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: int,
        config: Optional[IzhikevichConfig] = None,
        stdp_config: Optional[STDPConfig] = None,
        profile: Optional[MobProfile] = None,
    ):
        super().__init__(entity_type, entity_id, config)

        self.iz_config = config or IzhikevichConfig()
        self.stdp_config = stdp_config or STDPConfig()
        self.profile = profile or get_mob_profile(entity_type)

        # Build SNN with mob-specific parameters
        layer_sizes = [
            59,  # input: observation dim
            self.profile.num_hidden_neurons,
            48,  # output: spike decoder expects 48
        ]
        exc_ratios = [
            1.0,  # input: all excitatory
            self.profile.excitatory_ratio,
            1.0,  # output: all excitatory
        ]

        self.network = IzhikevichNetwork(
            layer_sizes=layer_sizes,
            neuron_params=self.profile.neuron_params,
            config=self.iz_config,
            stdp_config=self.stdp_config,
            excitatory_ratios=exc_ratios,
        )

        self.decoder = SpikeDecoder(temperature=1.0)
        self._last_rates: Optional[np.ndarray] = None
        self._pending_reward = 0.0

    def act(self, observation: BrainObservation) -> BrainAction:
        """Choose an action by simulating the SNN for one tick."""
        obs = observation.obs_vector.astype(np.float64)

        # Scale observation into input currents
        input_currents = obs * self.profile.base_current

        # Simulate SNN
        firing_rates = self.network.simulate_tick(input_currents)
        self._last_rates = firing_rates

        # Add mob-specific aggression bias
        biased_rates = self.decoder.add_aggression_bias(
            firing_rates, self.profile.aggression_bias
        )

        # Decode to action
        action_idx, action_probs = self.decoder.decode_rates(biased_rates)

        return BrainAction(
            action_idx=action_idx,
            action_probs=action_probs,
            metadata={
                "mob_type": self.profile.name,
                "mean_rate": float(np.mean(firing_rates)),
                "max_rate": float(np.max(firing_rates)),
            },
        )

    def learn(self, reward: float) -> Dict[str, float]:
        """Apply reward signal to modulate STDP weight updates."""
        self._pending_reward += reward
        self.network.apply_reward(self._pending_reward)
        applied = self._pending_reward
        self._pending_reward = 0.0

        stats = self.network.get_stats()
        stats["applied_reward"] = applied
        return stats

    def reset(self):
        """Reset SNN state for a new episode."""
        self.network.reset()
        self._last_rates = None
        self._pending_reward = 0.0
        self._step_count = 0
        self._total_reward = 0.0

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base["mob_profile"] = self.profile.name
        base.update(self.network.get_stats())
        if self._last_rates is not None:
            base["output_mean_rate"] = float(np.mean(self._last_rates))
        return base
