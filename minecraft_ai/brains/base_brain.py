"""Abstract brain interface for Minecraft AI entities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class BrainObservation:
    """Observation passed to a brain for decision-making."""
    obs_vector: np.ndarray  # 59-dim observation
    tick: int = 0
    entity_id: int = 0
    entity_type: str = ""
    raw_state: Optional[Dict[str, Any]] = None


@dataclass
class BrainAction:
    """Action output from a brain."""
    action_idx: int  # primary discrete action
    action_probs: np.ndarray = field(default_factory=lambda: np.zeros(9))
    value_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def intensity(self) -> float:
        """Confidence/intensity of the chosen action."""
        if len(self.action_probs) > self.action_idx:
            return float(self.action_probs[self.action_idx])
        return 1.0


class BaseBrain(ABC):
    """Abstract base class for all Minecraft AI brains."""

    def __init__(self, entity_type: str, entity_id: int, config: Any = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.config = config
        self._step_count = 0
        self._total_reward = 0.0

    @abstractmethod
    def act(self, observation: BrainObservation) -> BrainAction:
        """Choose an action given an observation."""
        ...

    @abstractmethod
    def learn(self, reward: float) -> Dict[str, float]:
        """Update internal state based on reward signal.

        Returns dict of learning metrics (e.g., loss values).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset brain state for a new episode."""
        ...

    def step(self, observation: BrainObservation, reward: float = 0.0) -> BrainAction:
        """Combined observe-learn-act cycle."""
        if self._step_count > 0:
            self.learn(reward)
        self._total_reward += reward
        self._step_count += 1
        return self.act(observation)

    def get_stats(self) -> Dict[str, Any]:
        """Return brain statistics for logging."""
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
        }

    def save(self, path: str) -> None:
        """Persist brain state to disk. Override in subclasses."""
        pass

    def load(self, path: str) -> None:
        """Load brain state from disk. Override in subclasses."""
        pass
