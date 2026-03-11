"""Track entity spawn/despawn/state in the Minecraft world."""

import logging
import time
from typing import Callable, Dict, List, Optional, Set

from .mc_state import EntityState

logger = logging.getLogger(__name__)


class EntityTracker:
    """Tracks entity lifecycle and state updates.

    Monitors entity spawns, despawns, and state changes,
    triggering callbacks for the brain manager to create/destroy
    brain instances accordingly.
    """

    def __init__(
        self,
        despawn_timeout_s: float = 30.0,
        max_entities: int = 32,
    ):
        self.despawn_timeout_s = despawn_timeout_s
        self.max_entities = max_entities

        self._entities: Dict[int, EntityState] = {}
        self._last_seen: Dict[int, float] = {}
        self._entity_types: Dict[int, str] = {}

        # Callbacks
        self._on_spawn: Optional[Callable[[EntityState], None]] = None
        self._on_despawn: Optional[Callable[[int, str], None]] = None
        self._on_update: Optional[Callable[[EntityState], None]] = None

    def on_spawn(self, callback: Callable[[EntityState], None]):
        """Register callback for entity spawn events."""
        self._on_spawn = callback

    def on_despawn(self, callback: Callable[[int, str], None]):
        """Register callback for entity despawn events."""
        self._on_despawn = callback

    def on_update(self, callback: Callable[[EntityState], None]):
        """Register callback for entity state updates."""
        self._on_update = callback

    def update(self, entities: List[EntityState]):
        """Process a batch of entity state updates from the bot server.

        Detects spawns, updates, and (via timeout) despawns.
        """
        now = time.time()
        seen_ids: Set[int] = set()

        for entity in entities:
            eid = entity.entity_id
            seen_ids.add(eid)
            self._last_seen[eid] = now

            if eid not in self._entities:
                # New entity spawn
                self._entities[eid] = entity
                self._entity_types[eid] = entity.entity_type
                if self._on_spawn:
                    self._on_spawn(entity)
                logger.debug(f"Entity spawned: {entity.entity_type}#{eid}")
            else:
                # Update existing entity
                self._entities[eid] = entity
                if self._on_update:
                    self._on_update(entity)

        # Check for despawns (entities not seen recently)
        self._check_despawns(now)

    def _check_despawns(self, now: float):
        """Remove entities that haven't been seen within the timeout."""
        despawned = []
        for eid, last_seen in list(self._last_seen.items()):
            if now - last_seen > self.despawn_timeout_s:
                despawned.append(eid)

        for eid in despawned:
            entity_type = self._entity_types.get(eid, "unknown")
            del self._entities[eid]
            del self._last_seen[eid]
            del self._entity_types[eid]
            if self._on_despawn:
                self._on_despawn(eid, entity_type)
            logger.debug(f"Entity despawned: {entity_type}#{eid}")

    def get_entity(self, entity_id: int) -> Optional[EntityState]:
        return self._entities.get(entity_id)

    def get_all_entities(self) -> List[EntityState]:
        return list(self._entities.values())

    def get_entities_by_type(self, entity_type: str) -> List[EntityState]:
        return [
            e for e in self._entities.values()
            if e.entity_type == entity_type
        ]

    def get_hostile_entities(self) -> List[EntityState]:
        return [e for e in self._entities.values() if e.is_hostile]

    def get_friendly_entities(self) -> List[EntityState]:
        return [e for e in self._entities.values() if not e.is_hostile]

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def tracked_ids(self) -> Set[int]:
        return set(self._entities.keys())

    def clear(self):
        """Remove all tracked entities."""
        self._entities.clear()
        self._last_seen.clear()
        self._entity_types.clear()

    def get_stats(self) -> Dict[str, int]:
        type_counts: Dict[str, int] = {}
        for etype in self._entity_types.values():
            type_counts[etype] = type_counts.get(etype, 0) + 1
        return {
            "total_entities": len(self._entities),
            "type_counts": type_counts,
        }
