"""Multi-layer spiking neural network with excitatory/inhibitory populations."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .neuron import NeuronParams, NeuronPopulation
from .plasticity import RewardModulatedSTDP
from ..config import IzhikevichConfig, STDPConfig


class SNNLayer:
    """A single layer with excitatory and inhibitory neuron populations."""

    def __init__(
        self,
        num_neurons: int,
        excitatory_ratio: float,
        params: NeuronParams,
        noise_amplitude: float = 0.0,
    ):
        self.num_neurons = num_neurons
        self.n_exc = int(num_neurons * excitatory_ratio)
        self.n_inh = num_neurons - self.n_exc

        # Inhibitory neurons get faster recovery (FS-like)
        inh_params = NeuronParams(
            a=0.1, b=0.2, c=params.c, d=params.d
        )

        self.excitatory = NeuronPopulation(self.n_exc, params, noise_amplitude)
        self.inhibitory = NeuronPopulation(self.n_inh, inh_params, noise_amplitude)

    def step(self, currents: np.ndarray, dt: float) -> np.ndarray:
        """Advance layer by dt ms. Returns combined spike mask."""
        exc_spikes = self.excitatory.step(currents[:self.n_exc], dt)
        inh_spikes = self.inhibitory.step(currents[self.n_exc:], dt)
        return np.concatenate([exc_spikes, inh_spikes])

    def get_spike_counts(self) -> np.ndarray:
        return np.concatenate([
            self.excitatory.spike_counts,
            self.inhibitory.spike_counts,
        ])

    def get_firing_rates(self, window_steps: int) -> np.ndarray:
        return np.concatenate([
            self.excitatory.get_firing_rates(window_steps),
            self.inhibitory.get_firing_rates(window_steps),
        ])

    def reset(self):
        self.excitatory.reset()
        self.inhibitory.reset()

    def reset_spike_counts(self):
        self.excitatory.reset_spike_counts()
        self.inhibitory.reset_spike_counts()


class IzhikevichNetwork:
    """Multi-layer SNN with reward-modulated STDP learning."""

    def __init__(
        self,
        layer_sizes: List[int],
        neuron_params: NeuronParams,
        config: IzhikevichConfig,
        stdp_config: Optional[STDPConfig] = None,
        excitatory_ratios: Optional[List[float]] = None,
    ):
        self.config = config
        self.stdp_config = stdp_config or STDPConfig()
        self.dt = config.dt
        self.steps_per_tick = config.steps_per_tick

        if excitatory_ratios is None:
            excitatory_ratios = [1.0] + [0.8] * (len(layer_sizes) - 2) + [1.0]

        # Build layers
        self.layers: List[SNNLayer] = []
        for i, size in enumerate(layer_sizes):
            layer = SNNLayer(
                num_neurons=size,
                excitatory_ratio=excitatory_ratios[i],
                params=neuron_params,
                noise_amplitude=config.noise_amplitude,
            )
            self.layers.append(layer)

        # Build inter-layer weight matrices and STDP modules
        self.weights: List[np.ndarray] = []
        self.stdp_modules: List[RewardModulatedSTDP] = []

        for i in range(len(layer_sizes) - 1):
            pre_n = layer_sizes[i]
            post_n = layer_sizes[i + 1]
            # Random sparse initialization
            w = np.random.randn(pre_n, post_n).astype(np.float64) * 0.5
            w = np.clip(w, config.min_weight, config.max_weight)
            # Sparsify
            mask = np.random.rand(pre_n, post_n) < 0.2
            w *= mask
            self.weights.append(w)

            stdp = RewardModulatedSTDP((pre_n, post_n), self.stdp_config)
            self.stdp_modules.append(stdp)

        # Recurrent weights within hidden layers
        self.recurrent_weights: List[Optional[np.ndarray]] = [None]  # no recurrence for input
        for i in range(1, len(layer_sizes) - 1):
            n = layer_sizes[i]
            rw = np.random.randn(n, n).astype(np.float64) * 0.3
            np.fill_diagonal(rw, 0)  # no self-connections
            mask = np.random.rand(n, n) < 0.1
            rw *= mask
            rw = np.clip(rw, config.min_weight, config.max_weight)
            self.recurrent_weights.append(rw)
        self.recurrent_weights.append(None)  # no recurrence for output

        self._tick_step_count = 0

    def simulate_tick(self, input_currents: np.ndarray) -> np.ndarray:
        """Run network for one game tick (steps_per_tick simulation steps).

        Args:
            input_currents: Current injection into input layer (n_input,)

        Returns:
            Output layer firing rates over the tick window.
        """
        # Reset spike counts for this tick window
        for layer in self.layers:
            layer.reset_spike_counts()

        for step in range(self.steps_per_tick):
            # Input layer
            layer_spikes = [self.layers[0].step(input_currents, self.dt)]

            # Hidden and output layers
            for i in range(1, len(self.layers)):
                # Feed-forward current from previous layer spikes
                pre_spikes = layer_spikes[i - 1].astype(np.float64)
                ff_current = pre_spikes @ self.weights[i - 1]

                # Recurrent current within layer
                if self.recurrent_weights[i] is not None:
                    prev_spikes_this_layer = self.layers[i].get_spike_counts() > 0
                    rec_current = prev_spikes_this_layer.astype(np.float64) @ self.recurrent_weights[i]
                    ff_current += rec_current

                ff_current += self.config.base_current

                post_spikes = self.layers[i].step(ff_current, self.dt)
                layer_spikes.append(post_spikes)

                # Update STDP eligibility
                self.stdp_modules[i - 1].on_spikes(
                    layer_spikes[i - 1], post_spikes, self.dt
                )

        self._tick_step_count += self.steps_per_tick

        # Return output layer firing rates
        return self.layers[-1].get_firing_rates(self.steps_per_tick)

    def apply_reward(self, reward: float):
        """Apply reward signal to modulate weight updates via R-STDP."""
        for i, stdp in enumerate(self.stdp_modules):
            self.weights[i] = stdp.apply_reward(
                self.weights[i], reward,
                self.config.min_weight, self.config.max_weight,
            )

    def reset(self):
        """Reset all network state."""
        for layer in self.layers:
            layer.reset()
        for stdp in self.stdp_modules:
            stdp.reset()
        self._tick_step_count = 0

    def get_stats(self) -> Dict[str, float]:
        """Get network statistics for logging."""
        stats = {}
        for i, layer in enumerate(self.layers):
            rates = layer.get_firing_rates(max(self.steps_per_tick, 1))
            stats[f"layer_{i}_mean_rate"] = float(np.mean(rates))
            stats[f"layer_{i}_max_rate"] = float(np.max(rates))
        for i, w in enumerate(self.weights):
            stats[f"weight_{i}_mean"] = float(np.mean(w))
            stats[f"weight_{i}_std"] = float(np.std(w))
        return stats
