"""59-electrode mapping for 10 Minecraft channel groups.

Maps the 59 usable electrodes (out of 64, excluding dead electrodes
0, 1, 6, 7, 56) into 10 functional channel groups for Minecraft control,
plus 2 reward feedback groups.

Channel Groups (12 total, 59 electrodes):
    0: encoding     (8 electrodes) - observation stimulation
    1: move_fwd     (5 electrodes) - forward movement readout
    2: move_back    (5 electrodes) - backward movement readout
    3: strafe_l     (5 electrodes) - strafe left readout
    4: strafe_r     (5 electrodes) - strafe right readout
    5: look_l       (5 electrodes) - look left readout
    6: look_r       (5 electrodes) - look right readout
    7: attack       (5 electrodes) - attack readout
    8: use_item     (5 electrodes) - use item readout
    9: jump_sneak   (5 electrodes) - jump/sneak readout
   10: reward_pos   (3 electrodes) - positive reward feedback
   11: reward_neg   (3 electrodes) - negative reward feedback
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# Dead electrodes: 0, 7, 56, 63 are unused; 4 is the reference channel
DEAD_ELECTRODES = {0, 4, 7, 56, 63}

# All usable electrodes (59 out of 64)
ALL_USABLE = sorted([i for i in range(64) if i not in DEAD_ELECTRODES])
assert len(ALL_USABLE) == 59

# Channel group definitions: name -> list of electrode indices
CHANNEL_GROUPS: Dict[str, List[int]] = {
    "encoding":    [1, 2, 3, 5, 6, 8, 9, 10],        # 8 electrodes
    "move_fwd":    [11, 12, 13, 14, 15],               # 5 electrodes
    "move_back":   [16, 17, 18, 19, 20],               # 5 electrodes
    "strafe_l":    [21, 22, 23, 24, 25],               # 5 electrodes
    "strafe_r":    [26, 27, 28, 29, 30],               # 5 electrodes
    "look_l":      [31, 32, 33, 34, 35],               # 5 electrodes
    "look_r":      [36, 37, 38, 39, 40],               # 5 electrodes
    "attack":      [41, 42, 43, 44, 45],               # 5 electrodes
    "use_item":    [46, 47, 48, 49, 50],               # 5 electrodes
    "jump_sneak":  [51, 52, 53, 54, 55],               # 5 electrodes
    "reward_pos":  [57, 58, 59],                        # 3 electrodes
    "reward_neg":  [60, 61, 62],                        # 3 electrodes
}

# Ordered group names (index corresponds to channel set index)
GROUP_NAMES = [
    "encoding", "move_fwd", "move_back", "strafe_l", "strafe_r",
    "look_l", "look_r", "attack", "use_item", "jump_sneak",
    "reward_pos", "reward_neg",
]

NUM_CHANNEL_GROUPS = 10  # action readout groups (excluding reward)
NUM_TOTAL_GROUPS = 12    # including reward feedback

# Action group name -> action index mapping
ACTION_GROUP_TO_INDEX: Dict[str, int] = {
    "move_fwd": 0,
    "move_back": 1,
    "strafe_l": 2,
    "strafe_r": 3,
    "look_l": 4,
    "look_r": 5,
    "attack": 6,
    "use_item": 7,
    "jump_sneak": 8,
}


@dataclass
class ChannelGroupInfo:
    """Metadata about a single channel group."""
    name: str
    index: int
    electrodes: List[int]
    num_electrodes: int
    is_input: bool  # True for encoding, False for readout
    is_reward: bool

    @property
    def electrode_mask(self) -> List[bool]:
        """64-element boolean mask for this group's electrodes."""
        mask = [False] * 64
        for e in self.electrodes:
            mask[e] = True
        return mask


def get_group_info(name: str) -> ChannelGroupInfo:
    """Get detailed info about a channel group."""
    electrodes = CHANNEL_GROUPS[name]
    idx = GROUP_NAMES.index(name)
    return ChannelGroupInfo(
        name=name,
        index=idx,
        electrodes=electrodes,
        num_electrodes=len(electrodes),
        is_input=(name == "encoding"),
        is_reward=(name in ("reward_pos", "reward_neg")),
    )


def get_all_group_infos() -> List[ChannelGroupInfo]:
    """Get info for all channel groups in order."""
    return [get_group_info(name) for name in GROUP_NAMES]


def electrode_to_group(electrode: int) -> str:
    """Find which group an electrode belongs to."""
    for name, electrodes in CHANNEL_GROUPS.items():
        if electrode in electrodes:
            return name
    raise ValueError(f"Electrode {electrode} not assigned to any group")


def validate_mapping():
    """Verify the channel mapping is consistent."""
    all_assigned = set()
    for name, electrodes in CHANNEL_GROUPS.items():
        for e in electrodes:
            assert e not in DEAD_ELECTRODES, f"Dead electrode {e} in group {name}"
            assert e not in all_assigned, f"Electrode {e} assigned to multiple groups"
            assert 0 <= e < 64, f"Electrode {e} out of range"
            all_assigned.add(e)

    assert all_assigned == set(ALL_USABLE), (
        f"Mapping mismatch: {len(all_assigned)} assigned vs {len(ALL_USABLE)} usable"
    )
    total = sum(len(v) for v in CHANNEL_GROUPS.values())
    assert total == 59, f"Total electrodes {total} != 59"


# Validate on import
validate_mapping()
