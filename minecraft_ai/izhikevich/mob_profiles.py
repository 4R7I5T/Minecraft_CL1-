"""Per-mob Izhikevich neuron parameters.

Each hostile mob type maps to a distinct spiking neuron behavior:
    Zombie  -> Regular Spiking (RS): persistent, relentless pursuit
    Skeleton -> Chattering (CH): burst-pause for repositioning/aiming
    Creeper -> Intrinsically Bursting (IB): slow buildup to explosive ignition
    Spider  -> Fast Spiking (FS): rapid reactions, high-frequency decisions
"""

from dataclasses import dataclass
from typing import Dict
from .neuron import NeuronParams


@dataclass
class MobProfile:
    """Complete neural profile for a mob type."""
    name: str
    neuron_params: NeuronParams
    excitatory_ratio: float
    num_hidden_neurons: int
    connection_density: float
    noise_level: float
    base_current: float
    aggression_bias: float  # added to attack-related output neurons


# Zombie: Regular Spiking - steady, persistent drive
ZOMBIE_PROFILE = MobProfile(
    name="zombie",
    neuron_params=NeuronParams(a=0.02, b=0.2, c=-65.0, d=8.0),
    excitatory_ratio=0.85,
    num_hidden_neurons=150,
    connection_density=0.15,
    noise_level=1.5,
    base_current=5.0,
    aggression_bias=3.0,
)

# Skeleton: Chattering - burst firing for aim-shoot-reposition cycles
SKELETON_PROFILE = MobProfile(
    name="skeleton",
    neuron_params=NeuronParams(a=0.02, b=0.2, c=-50.0, d=2.0),
    excitatory_ratio=0.75,
    num_hidden_neurons=180,
    connection_density=0.12,
    noise_level=2.0,
    base_current=4.0,
    aggression_bias=1.5,
)

# Creeper: Intrinsically Bursting - buildup to explosion
CREEPER_PROFILE = MobProfile(
    name="creeper",
    neuron_params=NeuronParams(a=0.02, b=0.2, c=-55.0, d=4.0),
    excitatory_ratio=0.9,
    num_hidden_neurons=120,
    connection_density=0.2,
    noise_level=1.0,
    base_current=3.0,
    aggression_bias=5.0,
)

# Spider: Fast Spiking - quick, reactive decisions
SPIDER_PROFILE = MobProfile(
    name="spider",
    neuron_params=NeuronParams(a=0.1, b=0.2, c=-65.0, d=2.0),
    excitatory_ratio=0.7,
    num_hidden_neurons=200,
    connection_density=0.1,
    noise_level=2.5,
    base_current=6.0,
    aggression_bias=2.0,
)


MOB_PROFILES: Dict[str, MobProfile] = {
    "zombie": ZOMBIE_PROFILE,
    "skeleton": SKELETON_PROFILE,
    "creeper": CREEPER_PROFILE,
    "spider": SPIDER_PROFILE,
}

# Entity types that map to hostile mob profiles
HOSTILE_ENTITY_TYPES = set(MOB_PROFILES.keys())


def get_mob_profile(entity_type: str) -> MobProfile:
    """Get the neural profile for a mob type, defaulting to zombie."""
    normalized = entity_type.lower().replace("_", "")
    for key, profile in MOB_PROFILES.items():
        if key in normalized:
            return profile
    return ZOMBIE_PROFILE
