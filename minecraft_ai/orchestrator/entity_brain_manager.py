"""Manages all brain instances with thread pool for parallel processing."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional, Set

from ..brains.base_brain import BaseBrain, BrainAction, BrainObservation
from ..brains.brain_registry import create_brain
from ..config import GameConfig
from ..environment.mc_state import EntityState

logger = logging.getLogger(__name__)


class EntityBrainManager:
    """Manages brain instances for all tracked entities.

    Creates/destroys brains as entities spawn/despawn, and processes
    all brain decisions in parallel using a thread pool.
    """

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self._brains: Dict[int, BaseBrain] = {}
        self._pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self._pending_futures: Dict[int, Future] = {}
        self._entity_types: Dict[int, str] = {}

    def on_entity_spawn(self, entity: EntityState):
        """Create a brain for a newly spawned entity."""
        if entity.entity_id in self._brains:
            return

        if len(self._brains) >= self.config.max_entities:
            logger.warning(
                f"Max entities ({self.config.max_entities}) reached, "
                f"skipping {entity.entity_type}#{entity.entity_id}"
            )
            return

        brain = create_brain(entity.entity_type, entity.entity_id)
        self._brains[entity.entity_id] = brain
        self._entity_types[entity.entity_id] = entity.entity_type
        logger.info(
            f"Brain created: {type(brain).__name__} for "
            f"{entity.entity_type}#{entity.entity_id}"
        )

    def on_entity_despawn(self, entity_id: int, entity_type: str):
        """Destroy brain for a despawned entity."""
        if entity_id in self._brains:
            del self._brains[entity_id]
            del self._entity_types[entity_id]
            # Cancel any pending future
            if entity_id in self._pending_futures:
                self._pending_futures[entity_id].cancel()
                del self._pending_futures[entity_id]
            logger.info(f"Brain destroyed: {entity_type}#{entity_id}")

    def process_tick(
        self,
        observations: Dict[int, BrainObservation],
        rewards: Dict[int, float],
    ) -> Dict[int, BrainAction]:
        """Process one tick for all active brains in parallel.

        Args:
            observations: entity_id -> observation mapping
            rewards: entity_id -> reward mapping

        Returns:
            entity_id -> action mapping
        """
        futures: Dict[int, Future] = {}

        for entity_id, brain in self._brains.items():
            obs = observations.get(entity_id)
            if obs is None:
                continue
            reward = rewards.get(entity_id, 0.0)
            future = self._pool.submit(self._process_single, brain, obs, reward)
            futures[entity_id] = future

        # Collect results
        actions: Dict[int, BrainAction] = {}
        for entity_id, future in futures.items():
            try:
                action = future.result(timeout=0.08)  # 80ms timeout (< 100ms tick)
                actions[entity_id] = action
            except Exception as e:
                logger.warning(
                    f"Brain error for entity {entity_id}: {e}"
                )

        return actions

    def _process_single(
        self, brain: BaseBrain, obs: BrainObservation, reward: float
    ) -> BrainAction:
        """Process a single brain's tick (runs in thread pool)."""
        return brain.step(obs, reward)

    def get_brain(self, entity_id: int) -> Optional[BaseBrain]:
        return self._brains.get(entity_id)

    def get_all_brains(self) -> Dict[int, BaseBrain]:
        return dict(self._brains)

    @property
    def active_count(self) -> int:
        return len(self._brains)

    @property
    def active_ids(self) -> Set[int]:
        return set(self._brains.keys())

    def reset_all(self):
        """Reset all brain instances."""
        for brain in self._brains.values():
            brain.reset()

    def get_stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for etype in self._entity_types.values():
            type_counts[etype] = type_counts.get(etype, 0) + 1

        return {
            "active_brains": len(self._brains),
            "type_counts": type_counts,
            "pool_size": self.config.thread_pool_size,
        }

    def shutdown(self):
        """Shut down the thread pool."""
        self._pool.shutdown(wait=False)
        self._brains.clear()
        self._entity_types.clear()
        logger.info("EntityBrainManager shut down")
