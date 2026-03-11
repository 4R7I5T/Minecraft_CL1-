"""Survival/combat/resource reward profiles.

Computes shaped rewards from state transitions for different entity roles:
- Survival: health maintenance, hunger management
- Combat: damage dealt/taken, kills, deaths
- Resource: block breaking, item collection
- Exploration: distance traveled, new chunks
"""

import numpy as np
from typing import Dict, Optional

from ..config import RewardConfig
from .mc_state import WorldSnapshot


class RewardShaper:
    """Computes shaped rewards from consecutive world snapshots."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_snapshot: Optional[WorldSnapshot] = None
        self._prev_health: float = 20.0
        self._prev_hunger: float = 20.0
        self._prev_position: Optional[np.ndarray] = None
        self._total_distance: float = 0.0
        self._kill_count: int = 0

    def compute_reward(
        self,
        snapshot: WorldSnapshot,
        events: Optional[Dict] = None,
    ) -> float:
        """Compute shaped reward from current state and events.

        Args:
            snapshot: Current world state
            events: Optional game events (kills, deaths, blocks broken, etc.)

        Returns:
            Scalar reward value
        """
        reward = 0.0
        events = events or {}
        player = snapshot.player

        # Survival rewards
        reward += self._survival_reward(player.health, player.hunger)

        # Combat rewards
        reward += self._combat_reward(player.health, events)

        # Resource rewards
        reward += self._resource_reward(events)

        # Exploration rewards
        reward += self._exploration_reward(player.position)

        # Update state tracking
        self._prev_health = player.health
        self._prev_hunger = player.hunger
        self._prev_position = player.position.copy()
        self._prev_snapshot = snapshot

        return reward

    def _survival_reward(self, health: float, hunger: float) -> float:
        """Reward for staying alive and maintaining health/hunger."""
        reward = 0.0
        c = self.config

        # Small positive reward for being alive
        reward += 0.01 * c.survival_weight

        # Penalty for health loss
        health_delta = health - self._prev_health
        if health_delta < 0:
            reward += health_delta * c.damage_taken_scale * c.survival_weight

        # Death penalty
        if health <= 0:
            reward += c.death_penalty * c.survival_weight

        return reward

    def _combat_reward(self, health: float, events: Dict) -> float:
        """Reward for combat performance."""
        reward = 0.0
        c = self.config

        # Kill reward
        kills = events.get("kills", 0)
        if kills > 0:
            reward += kills * c.kill_reward * c.combat_weight
            self._kill_count += kills

        # Damage dealt reward
        damage_dealt = events.get("damage_dealt", 0.0)
        if damage_dealt > 0:
            reward += damage_dealt * c.damage_dealt_scale * c.combat_weight

        return reward

    def _resource_reward(self, events: Dict) -> float:
        """Reward for resource gathering."""
        reward = 0.0
        c = self.config

        # Block breaking
        blocks_broken = events.get("blocks_broken", 0)
        reward += blocks_broken * c.block_break_reward * c.resource_weight

        # Food consumption
        food_eaten = events.get("food_eaten", 0)
        reward += food_eaten * c.food_reward * c.resource_weight

        return reward

    def _exploration_reward(self, position: np.ndarray) -> float:
        """Reward for exploring new areas."""
        if self._prev_position is None:
            return 0.0

        distance = float(np.linalg.norm(position - self._prev_position))
        self._total_distance += distance

        # Small reward for movement (encourages exploration)
        reward = min(distance * 0.01, 0.05) * self.config.exploration_weight
        return reward

    def reset(self):
        """Reset reward state for a new episode."""
        self._prev_snapshot = None
        self._prev_health = 20.0
        self._prev_hunger = 20.0
        self._prev_position = None
        self._total_distance = 0.0
        self._kill_count = 0

    def get_stats(self) -> Dict[str, float]:
        return {
            "total_distance": self._total_distance,
            "kill_count": self._kill_count,
        }


class HostileMobRewardShaper(RewardShaper):
    """Reward shaper tuned for hostile mob behavior.

    Hostile mobs get rewarded for:
    - Dealing damage to players
    - Staying close to targets
    - Successful kills
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__(config)
        if self.config.combat_weight < 2.0:
            self.config.combat_weight = 3.0
        self.config.exploration_weight = 0.1
        self.config.resource_weight = 0.0

    def compute_mob_reward(
        self,
        mob_health: float,
        target_distance: float,
        events: Optional[Dict] = None,
    ) -> float:
        """Compute reward specifically for hostile mob behavior.

        Args:
            mob_health: Current mob health
            target_distance: Distance to nearest target
            events: Combat events (damage_dealt, kills)

        Returns:
            Scalar reward
        """
        reward = 0.0
        events = events or {}

        # Proximity reward: closer to target = better
        if target_distance < 5.0:
            reward += (5.0 - target_distance) * 0.1
        elif target_distance < 20.0:
            reward += (20.0 - target_distance) * 0.02

        # Combat rewards
        damage_dealt = events.get("damage_dealt", 0.0)
        reward += damage_dealt * self.config.damage_dealt_scale * self.config.combat_weight

        kills = events.get("kills", 0)
        reward += kills * self.config.kill_reward * self.config.combat_weight

        # Penalize mob death
        if mob_health <= 0:
            reward += self.config.death_penalty * 0.5

        return reward
