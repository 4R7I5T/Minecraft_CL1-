"""Short burst CL1 training session manager.

Manages 2-10 minute training sessions on CL1 hardware:
1. Stimulate with encoded observations
2. Record spike responses
3. Collect (obs, spikes, reward) tuples
4. Session data feeds into distillation pipeline
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..config import CL1Config
from .mc_cl1_interface import MCCL1Interface
from .channel_mapping import NUM_CHANNEL_GROUPS

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """Single observation-spike-reward record from CL1 training."""
    observation: np.ndarray     # (59,) observation vector
    spike_counts: np.ndarray    # (10,) spike counts per channel group
    reward: float
    timestamp: float
    tick: int


@dataclass
class SessionResult:
    """Results from a completed CL1 training session."""
    records: List[TrainingRecord]
    duration_seconds: float
    total_ticks: int
    mean_reward: float
    total_reward: float
    mean_latency_ms: float

    @property
    def num_samples(self) -> int:
        return len(self.records)

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract arrays for distillation: (observations, spike_counts, rewards)."""
        obs = np.array([r.observation for r in self.records], dtype=np.float32)
        spikes = np.array([r.spike_counts for r in self.records], dtype=np.float32)
        rewards = np.array([r.reward for r in self.records], dtype=np.float32)
        return obs, spikes, rewards


class CL1TrainingSession:
    """Manages a short CL1 hardware training session."""

    def __init__(
        self,
        cl1_interface: MCCL1Interface,
        config: CL1Config,
        max_duration_s: float = 600.0,
        min_samples: int = 500,
    ):
        self.cl1 = cl1_interface
        self.config = config
        self.max_duration_s = max_duration_s
        self.min_samples = min_samples
        self._records: List[TrainingRecord] = []
        self._running = False
        self._start_time = 0.0
        self._tick = 0
        self._total_latency = 0.0

    def start(self):
        """Begin a new training session."""
        if not self.cl1.is_connected:
            raise RuntimeError("CL1 not connected")
        self._records = []
        self._running = True
        self._start_time = time.time()
        self._tick = 0
        self._total_latency = 0.0
        logger.info("CL1 training session started")

    def record_tick(
        self,
        observation: np.ndarray,
        reward: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Process one training tick: stimulate, record, store.

        Args:
            observation: (59,) observation vector
            reward: reward signal for this tick

        Returns:
            Spike counts if successful, None otherwise
        """
        if not self._running:
            return None

        # Encode observation to stimulation params
        frequencies, amplitudes = self.cl1.encode_observation(observation)

        # Send reward feedback
        if reward != 0.0:
            self.cl1.send_reward(reward)

        # Stimulate and read spikes
        spike_counts = self.cl1.stimulate_and_read(frequencies, amplitudes)
        if spike_counts is None:
            spike_counts = np.zeros(NUM_CHANNEL_GROUPS, dtype=np.float32)

        # Store record
        record = TrainingRecord(
            observation=observation.copy(),
            spike_counts=spike_counts.copy(),
            reward=reward,
            timestamp=time.time(),
            tick=self._tick,
        )
        self._records.append(record)
        self._total_latency += self.cl1.latency_ms
        self._tick += 1

        # Check stopping conditions
        elapsed = time.time() - self._start_time
        if elapsed >= self.max_duration_s:
            logger.info(f"Session time limit reached ({elapsed:.1f}s)")
            self._running = False

        return spike_counts

    def stop(self) -> SessionResult:
        """End the training session and return results."""
        self._running = False
        duration = time.time() - self._start_time

        rewards = [r.reward for r in self._records]
        total_reward = sum(rewards)
        mean_reward = total_reward / max(len(rewards), 1)
        mean_latency = self._total_latency / max(self._tick, 1)

        result = SessionResult(
            records=self._records,
            duration_seconds=duration,
            total_ticks=self._tick,
            mean_reward=mean_reward,
            total_reward=total_reward,
            mean_latency_ms=mean_latency,
        )

        logger.info(
            f"CL1 session ended: {result.num_samples} samples, "
            f"{duration:.1f}s, mean_reward={mean_reward:.3f}, "
            f"latency={mean_latency:.1f}ms"
        )
        return result

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_enough_samples(self) -> bool:
        return len(self._records) >= self.min_samples

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time
