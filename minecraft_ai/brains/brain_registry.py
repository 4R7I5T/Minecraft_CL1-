"""Entity type -> brain class mapping.

Routes each Minecraft entity type to the appropriate brain backend:
- Hostile mobs -> IzhikevichBrain with mob-specific profiles
- Players, villagers, passive mobs -> CL1HybridBrain
"""

from typing import Any, Dict, Optional, Type

from .base_brain import BaseBrain
from .cl1_hybrid_brain import CL1HybridBrain
from .izhikevich_brain import IzhikevichBrain
from ..izhikevich.mob_profiles import HOSTILE_ENTITY_TYPES, get_mob_profile


# Entity types mapped to CL1 hybrid brain
CL1_ENTITY_TYPES = {
    "player", "villager",
    "pig", "cow", "sheep", "chicken", "horse",
    "wolf", "cat", "rabbit", "fox",
    "iron_golem", "snow_golem",
}

# Entity types mapped to Izhikevich SNN brain
IZHIKEVICH_ENTITY_TYPES = HOSTILE_ENTITY_TYPES  # zombie, skeleton, creeper, spider

# Complete assignment registry
BRAIN_ASSIGNMENTS: Dict[str, str] = {}
for e in CL1_ENTITY_TYPES:
    BRAIN_ASSIGNMENTS[e] = "cl1_hybrid"
for e in IZHIKEVICH_ENTITY_TYPES:
    BRAIN_ASSIGNMENTS[e] = "izhikevich"


def get_brain_class(entity_type: str) -> Type[BaseBrain]:
    """Get the brain class for an entity type."""
    normalized = entity_type.lower().replace(" ", "_")

    if normalized in IZHIKEVICH_ENTITY_TYPES:
        return IzhikevichBrain
    # Default to CL1 hybrid for any non-hostile entity
    return CL1HybridBrain


def create_brain(entity_type: str, entity_id: int, config: Any = None) -> BaseBrain:
    """Factory: create the appropriate brain for an entity.

    Args:
        entity_type: Minecraft entity type string
        entity_id: Unique entity identifier
        config: Optional configuration object

    Returns:
        Instantiated brain matching the entity type
    """
    normalized = entity_type.lower().replace(" ", "_")
    brain_class = get_brain_class(normalized)

    if brain_class is IzhikevichBrain:
        profile = get_mob_profile(normalized)
        return IzhikevichBrain(
            entity_type=normalized,
            entity_id=entity_id,
            profile=profile,
        )
    else:
        return CL1HybridBrain(
            entity_type=normalized,
            entity_id=entity_id,
        )


def get_assignment(entity_type: str) -> str:
    """Get the brain assignment string for an entity type."""
    normalized = entity_type.lower().replace(" ", "_")
    return BRAIN_ASSIGNMENTS.get(normalized, "cl1_hybrid")
