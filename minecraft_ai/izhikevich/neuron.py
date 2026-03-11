"""Core Izhikevich neuron model equations.

Implements the 2003 Izhikevich simple model:
    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30 mV: v = c, u = u + d

Parameters (a, b, c, d) define neuron type:
    Regular Spiking (RS), Intrinsically Bursting (IB),
    Chattering (CH), Fast Spiking (FS), etc.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuronParams:
    """Parameters defining an Izhikevich neuron type."""
    a: float  # recovery time scale
    b: float  # recovery sensitivity to v
    c: float  # post-spike reset voltage (mV)
    d: float  # post-spike recovery increment
    v_thresh: float = 30.0  # spike threshold (mV)
    v_init: float = -65.0  # initial membrane potential
    u_init: Optional[float] = None  # initial recovery (defaults to b * v_init)

    def __post_init__(self):
        if self.u_init is None:
            self.u_init = self.b * self.v_init


# Standard neuron type presets
REGULAR_SPIKING = NeuronParams(a=0.02, b=0.2, c=-65.0, d=8.0)
INTRINSICALLY_BURSTING = NeuronParams(a=0.02, b=0.2, c=-55.0, d=4.0)
CHATTERING = NeuronParams(a=0.02, b=0.2, c=-50.0, d=2.0)
FAST_SPIKING = NeuronParams(a=0.1, b=0.2, c=-65.0, d=2.0)
LOW_THRESHOLD_SPIKING = NeuronParams(a=0.02, b=0.25, c=-65.0, d=2.0)


class IzhikevichNeuron:
    """Single Izhikevich neuron with state tracking."""

    def __init__(self, params: NeuronParams):
        self.params = params
        self.v = params.v_init
        self.u = params.u_init
        self.fired = False

    def step(self, I: float, dt: float = 0.5) -> bool:
        """Advance neuron by dt milliseconds with input current I.

        Uses two half-steps for numerical stability (Izhikevich's recommendation).
        Returns True if neuron fired.
        """
        p = self.params
        self.fired = False

        # Two half-steps for v (numerical stability)
        half_dt = dt / 2.0
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        self.u += dt * p.a * (p.b * self.v - self.u)

        if self.v >= p.v_thresh:
            self.v = p.c
            self.u += p.d
            self.fired = True

        return self.fired

    def reset(self):
        """Reset to initial conditions."""
        self.v = self.params.v_init
        self.u = self.params.u_init
        self.fired = False


class NeuronPopulation:
    """Vectorized population of Izhikevich neurons for efficient simulation."""

    def __init__(self, n: int, params: NeuronParams, noise_amplitude: float = 0.0):
        self.n = n
        self.params = params
        self.noise_amplitude = noise_amplitude

        self.v = np.full(n, params.v_init, dtype=np.float64)
        self.u = np.full(n, params.u_init, dtype=np.float64)
        self.fired = np.zeros(n, dtype=bool)
        self.spike_counts = np.zeros(n, dtype=np.int32)

    def step(self, I: np.ndarray, dt: float = 0.5) -> np.ndarray:
        """Advance all neurons by dt ms. Returns boolean spike mask."""
        p = self.params

        if self.noise_amplitude > 0:
            I = I + np.random.randn(self.n) * self.noise_amplitude

        half_dt = dt / 2.0
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        self.u += dt * p.a * (p.b * self.v - self.u)

        self.fired = self.v >= p.v_thresh
        self.v[self.fired] = p.c
        self.u[self.fired] += p.d
        self.spike_counts += self.fired.astype(np.int32)

        return self.fired

    def reset(self):
        """Reset all neurons to initial conditions."""
        self.v[:] = self.params.v_init
        self.u[:] = self.params.u_init
        self.fired[:] = False
        self.spike_counts[:] = 0

    def reset_spike_counts(self):
        """Reset spike counters without resetting membrane state."""
        self.spike_counts[:] = 0

    def get_firing_rates(self, window_steps: int) -> np.ndarray:
        """Compute firing rates from spike counts over a window."""
        if window_steps <= 0:
            return np.zeros(self.n, dtype=np.float32)
        return self.spike_counts.astype(np.float32) / window_steps
