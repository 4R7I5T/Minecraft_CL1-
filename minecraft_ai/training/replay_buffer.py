"""Experience storage for training."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Experience:
    """Single step of experience."""
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 10000, obs_dim: int = 59):
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

    def add(self, exp: Experience):
        """Add a single experience to the buffer."""
        idx = self._pos % self.capacity
        self.observations[idx] = exp.observation
        self.actions[idx] = exp.action
        self.rewards[idx] = exp.reward
        self.next_observations[idx] = exp.next_observation
        self.dones[idx] = exp.done
        self.log_probs[idx] = exp.log_prob
        self.values[idx] = exp.value

        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of experiences."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
        }

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all stored experiences."""
        s = self._size
        return {
            "observations": self.observations[:s],
            "actions": self.actions[:s],
            "rewards": self.rewards[:s],
            "next_observations": self.next_observations[:s],
            "dones": self.dones[:s],
            "log_probs": self.log_probs[:s],
            "values": self.values[:s],
        }

    def clear(self):
        """Clear the buffer."""
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity


class RolloutBuffer:
    """Collects rollout data for on-policy training (PPO)."""

    def __init__(self, rollout_length: int = 128, obs_dim: int = 59):
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self._experiences: List[Experience] = []
        self._advantages: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None

    def add(self, exp: Experience):
        """Add experience to rollout."""
        self._experiences.append(exp)

    @property
    def is_full(self) -> bool:
        return len(self._experiences) >= self.rollout_length

    def compute_gae(
        self, gamma: float = 0.99, lam: float = 0.95,
        last_value: float = 0.0,
    ):
        """Compute Generalized Advantage Estimation."""
        n = len(self._experiences)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_adv = 0.0
        last_val = last_value

        for t in reversed(range(n)):
            exp = self._experiences[t]
            next_val = last_val if t == n - 1 else self._experiences[t + 1].value
            non_terminal = 0.0 if exp.done else 1.0

            delta = exp.reward + gamma * next_val * non_terminal - exp.value
            last_adv = delta + gamma * lam * non_terminal * last_adv
            advantages[t] = last_adv
            returns[t] = advantages[t] + exp.value

        self._advantages = advantages
        self._returns = returns

    def get_batches(
        self, batch_size: int
    ) -> List[Dict[str, np.ndarray]]:
        """Get shuffled mini-batches for PPO epochs."""
        n = len(self._experiences)
        indices = np.random.permutation(n)

        obs = np.array([e.observation for e in self._experiences], dtype=np.float32)
        actions = np.array([e.action for e in self._experiences], dtype=np.int64)
        old_log_probs = np.array([e.log_prob for e in self._experiences], dtype=np.float32)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batches.append({
                "observations": obs[idx],
                "actions": actions[idx],
                "old_log_probs": old_log_probs[idx],
                "advantages": self._advantages[idx],
                "returns": self._returns[idx],
            })

        return batches

    def clear(self):
        """Clear rollout data."""
        self._experiences = []
        self._advantages = None
        self._returns = None
