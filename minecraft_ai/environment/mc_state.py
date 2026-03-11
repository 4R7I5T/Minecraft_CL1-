"""Minecraft state representation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EntityState:
    """State of a single entity in the world."""
    entity_id: int
    entity_type: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    health: float = 20.0
    is_hostile: bool = False
    distance: float = 0.0  # distance from observer

    def as_feature_vector(self) -> np.ndarray:
        """5-dim feature: [rel_x, rel_y, rel_z, health_norm, is_hostile]."""
        return np.array([
            self.x, self.y, self.z,
            self.health / 20.0,
            float(self.is_hostile),
        ], dtype=np.float32)


@dataclass
class PlayerState:
    """State of the controlled player/entity."""
    health: float = 20.0
    hunger: float = 20.0
    x: float = 0.0
    y: float = 64.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    time_of_day: float = 0.0  # 0-24000 normalized to 0-1
    light_level: float = 15.0  # 0-15 normalized to 0-1
    is_on_ground: bool = True
    is_in_water: bool = False
    is_sneaking: bool = False
    held_item_id: int = 0
    nearby_entities: List[EntityState] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float32)


@dataclass
class WorldSnapshot:
    """Complete world state at a single tick."""
    tick: int
    player: PlayerState
    entities: Dict[int, EntityState] = field(default_factory=dict)
    blocks_nearby: Optional[np.ndarray] = None

    def get_observation_vector(self, max_entities: int = 8) -> np.ndarray:
        """Build 59-dim observation vector.

        Layout:
            [0]     health (normalized 0-1)
            [1]     hunger (normalized 0-1)
            [2:5]   position (x, y, z) / 1000
            [5:8]   velocity (vx, vy, vz)
            [8:10]  orientation (yaw/360, pitch/180)
            [10]    time_of_day (0-1)
            [11]    light_level (0-1)
            [12:15] flags (on_ground, in_water, sneaking)
            [15]    held_item_id (normalized)
            [16:18] padding (reserved)
            [18]    num_nearby_entities (normalized)
            [19:59] 8 entities x 5 features
        """
        obs = np.zeros(59, dtype=np.float32)
        p = self.player

        obs[0] = p.health / 20.0
        obs[1] = p.hunger / 20.0
        obs[2] = p.x / 1000.0
        obs[3] = p.y / 256.0
        obs[4] = p.z / 1000.0
        obs[5] = p.vx
        obs[6] = p.vy
        obs[7] = p.vz
        obs[8] = p.yaw / 360.0
        obs[9] = p.pitch / 180.0
        obs[10] = p.time_of_day
        obs[11] = p.light_level / 15.0
        obs[12] = float(p.is_on_ground)
        obs[13] = float(p.is_in_water)
        obs[14] = float(p.is_sneaking)
        obs[15] = min(p.held_item_id / 1000.0, 1.0)

        # Sort entities by distance, take closest
        sorted_entities = sorted(
            p.nearby_entities, key=lambda e: e.distance
        )[:max_entities]

        obs[18] = len(sorted_entities) / max_entities

        for i, ent in enumerate(sorted_entities):
            base = 19 + i * 5
            features = ent.as_feature_vector()
            # Normalize relative positions
            features[0] /= 50.0  # rel_x
            features[1] /= 50.0  # rel_y
            features[2] /= 50.0  # rel_z
            obs[base:base + 5] = np.clip(features, -1.0, 1.0)

        return obs
