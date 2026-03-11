"""Build 59-dim observation vectors from raw Minecraft state."""

import numpy as np
from typing import Dict, List, Optional

from .mc_state import EntityState, PlayerState, WorldSnapshot


class ObservationBuilder:
    """Constructs normalized 59-dim observation vectors from world state.

    Vector layout (59 dimensions):
        [0]      health (0-1)
        [1]      hunger (0-1)
        [2:5]    position (x/1000, y/256, z/1000)
        [5:8]    velocity (vx, vy, vz)
        [8:10]   orientation (yaw/360, pitch/180)
        [10]     time_of_day (0-1)
        [11]     light_level (0-1)
        [12:15]  flags (on_ground, in_water, sneaking)
        [15]     held_item_id (normalized)
        [16:18]  reserved padding
        [18]     num_nearby_entities / max_entities
        [19:59]  8 entities x 5 features
    """

    def __init__(self, max_entities: int = 8, obs_dim: int = 59):
        self.max_entities = max_entities
        self.obs_dim = obs_dim

    def build(self, snapshot: WorldSnapshot) -> np.ndarray:
        """Build observation vector from a world snapshot."""
        return snapshot.get_observation_vector(max_entities=self.max_entities)

    def build_from_raw(self, raw_state: dict) -> np.ndarray:
        """Build observation from raw JSON state from bot server.

        Args:
            raw_state: Dictionary with keys matching bot server output

        Returns:
            (59,) normalized observation vector
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Player state
        player = raw_state.get("player", {})
        obs[0] = player.get("health", 20.0) / 20.0
        obs[1] = player.get("food", 20.0) / 20.0

        pos = player.get("position", {})
        obs[2] = pos.get("x", 0.0) / 1000.0
        obs[3] = pos.get("y", 64.0) / 256.0
        obs[4] = pos.get("z", 0.0) / 1000.0

        vel = player.get("velocity", {})
        obs[5] = vel.get("x", 0.0)
        obs[6] = vel.get("y", 0.0)
        obs[7] = vel.get("z", 0.0)

        obs[8] = player.get("yaw", 0.0) / 360.0
        obs[9] = player.get("pitch", 0.0) / 180.0
        obs[10] = player.get("timeOfDay", 0) / 24000.0
        obs[11] = player.get("lightLevel", 15) / 15.0
        obs[12] = float(player.get("onGround", True))
        obs[13] = float(player.get("isInWater", False))
        obs[14] = float(player.get("isSneaking", False))
        obs[15] = min(player.get("heldItem", 0) / 1000.0, 1.0)

        # Nearby entities
        entities = raw_state.get("entities", [])
        player_pos = np.array([
            pos.get("x", 0.0), pos.get("y", 64.0), pos.get("z", 0.0)
        ])

        # Sort by distance, take closest
        entity_features = []
        for ent in entities:
            ent_pos = np.array([
                ent.get("x", 0.0),
                ent.get("y", 0.0),
                ent.get("z", 0.0),
            ])
            rel_pos = ent_pos - player_pos
            dist = float(np.linalg.norm(rel_pos))
            entity_features.append((dist, rel_pos, ent))

        entity_features.sort(key=lambda x: x[0])

        obs[18] = min(len(entity_features), self.max_entities) / self.max_entities

        for i, (dist, rel_pos, ent) in enumerate(entity_features[:self.max_entities]):
            base = 19 + i * 5
            obs[base] = np.clip(rel_pos[0] / 50.0, -1.0, 1.0)
            obs[base + 1] = np.clip(rel_pos[1] / 50.0, -1.0, 1.0)
            obs[base + 2] = np.clip(rel_pos[2] / 50.0, -1.0, 1.0)
            obs[base + 3] = ent.get("health", 20.0) / 20.0
            obs[base + 4] = float(ent.get("isHostile", False))

        return obs

    def build_from_entity_perspective(
        self,
        entity: EntityState,
        all_entities: List[EntityState],
        time_of_day: float = 0.0,
        light_level: float = 15.0,
    ) -> np.ndarray:
        """Build observation from an entity's perspective (for mob brains).

        Uses the entity's own position as the reference frame.
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        obs[0] = entity.health / 20.0
        obs[1] = 1.0  # mobs don't have hunger
        obs[2] = entity.x / 1000.0
        obs[3] = entity.y / 256.0
        obs[4] = entity.z / 1000.0
        obs[8] = entity.yaw / 360.0
        obs[9] = entity.pitch / 180.0
        obs[10] = time_of_day
        obs[11] = light_level / 15.0
        obs[12] = 1.0  # assume on ground
        obs[15] = 0.0  # no held item

        # Other entities relative to this one
        others = [
            e for e in all_entities
            if e.entity_id != entity.entity_id
        ]
        entity_pos = np.array([entity.x, entity.y, entity.z])

        entity_dists = []
        for other in others:
            other_pos = np.array([other.x, other.y, other.z])
            rel = other_pos - entity_pos
            dist = float(np.linalg.norm(rel))
            entity_dists.append((dist, rel, other))

        entity_dists.sort(key=lambda x: x[0])

        obs[18] = min(len(entity_dists), self.max_entities) / self.max_entities

        for i, (dist, rel, other) in enumerate(entity_dists[:self.max_entities]):
            base = 19 + i * 5
            obs[base] = np.clip(rel[0] / 50.0, -1.0, 1.0)
            obs[base + 1] = np.clip(rel[1] / 50.0, -1.0, 1.0)
            obs[base + 2] = np.clip(rel[2] / 50.0, -1.0, 1.0)
            obs[base + 3] = other.health / 20.0
            obs[base + 4] = float(other.is_hostile)

        return obs
