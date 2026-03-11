"""Minecraft action space definition."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional


class ActionType(IntEnum):
    """Discrete action types for Minecraft bot control."""
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    STRAFE_LEFT = 2
    STRAFE_RIGHT = 3
    LOOK_LEFT = 4
    LOOK_RIGHT = 5
    ATTACK = 6
    USE_ITEM = 7
    JUMP_SNEAK = 8


NUM_ACTIONS = len(ActionType)

ACTION_NAMES: Dict[int, str] = {a.value: a.name.lower() for a in ActionType}

# Map action types to CL1 channel group indices
ACTION_TO_CHANNEL_GROUP: Dict[ActionType, int] = {
    ActionType.MOVE_FORWARD: 1,
    ActionType.MOVE_BACKWARD: 2,
    ActionType.STRAFE_LEFT: 3,
    ActionType.STRAFE_RIGHT: 4,
    ActionType.LOOK_LEFT: 5,
    ActionType.LOOK_RIGHT: 6,
    ActionType.ATTACK: 7,
    ActionType.USE_ITEM: 8,
    ActionType.JUMP_SNEAK: 9,
}


@dataclass
class BotCommand:
    """Command to send to the Node.js bot server."""
    action_type: ActionType
    intensity: float = 1.0  # 0.0-1.0 scaling for continuous actions
    duration_ticks: int = 1

    def to_dict(self) -> dict:
        return {
            "action": ACTION_NAMES[self.action_type.value],
            "intensity": self.intensity,
            "duration": self.duration_ticks,
        }


@dataclass
class CompoundAction:
    """Multiple simultaneous actions (e.g., move forward + look left)."""
    commands: List[BotCommand]

    def to_dict(self) -> dict:
        return {
            "commands": [cmd.to_dict() for cmd in self.commands],
            "compound": True,
        }


def action_index_to_command(action_idx: int, intensity: float = 1.0) -> BotCommand:
    """Convert a discrete action index to a BotCommand."""
    action_type = ActionType(action_idx)
    return BotCommand(action_type=action_type, intensity=intensity)


def decode_multi_action(action_probs: List[float], threshold: float = 0.5) -> CompoundAction:
    """Decode per-action probabilities into a compound action."""
    commands = []
    for idx, prob in enumerate(action_probs):
        if prob > threshold:
            commands.append(BotCommand(
                action_type=ActionType(idx),
                intensity=min(prob, 1.0),
            ))
    return CompoundAction(commands=commands)
