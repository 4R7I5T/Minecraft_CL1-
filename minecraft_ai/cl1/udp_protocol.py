"""Binary packet protocol for CL1 communication (10 channel sets).

Packet Formats:
    Stimulation Command (Training -> CL1):
        [8 bytes timestamp][40 bytes frequencies][40 bytes amplitudes]
        Total: 88 bytes (10 channel groups * 2 * 4 bytes + 8 byte timestamp)

    Spike Data (CL1 -> Training):
        [8 bytes timestamp][40 bytes spike_counts]
        Total: 48 bytes

    Reward Feedback (Training -> CL1):
        [8 bytes timestamp][1 byte type][4 bytes value][3 bytes channels]
        Total: 16 bytes
"""

import struct
import time
import numpy as np
from typing import Tuple, List

from .channel_mapping import NUM_CHANNEL_GROUPS, NUM_TOTAL_GROUPS

# Packet constants
TIMESTAMP_SIZE = 8
FLOAT_SIZE = 4

STIM_PACKET_SIZE = TIMESTAMP_SIZE + (NUM_CHANNEL_GROUPS * 2 * FLOAT_SIZE)  # 88 bytes
SPIKE_PACKET_SIZE = TIMESTAMP_SIZE + (NUM_CHANNEL_GROUPS * FLOAT_SIZE)     # 48 bytes
REWARD_PACKET_SIZE = 16

# Struct format strings (little-endian)
STIM_FORMAT = '<Q' + ('f' * NUM_CHANNEL_GROUPS * 2)
SPIKE_FORMAT = '<Q' + ('f' * NUM_CHANNEL_GROUPS)
REWARD_FORMAT = '<QBf3B'

# Reward types
REWARD_POSITIVE = 1
REWARD_NEGATIVE = 2
REWARD_NEUTRAL = 0


def _timestamp_us() -> int:
    """Current time in microseconds."""
    return int(time.time() * 1_000_000)


def pack_stimulation(frequencies: np.ndarray, amplitudes: np.ndarray) -> bytes:
    """Pack stimulation parameters for 10 channel groups.

    Args:
        frequencies: (10,) array of Hz values per group
        amplitudes: (10,) array of amplitude values per group

    Returns:
        88-byte binary packet
    """
    assert frequencies.shape == (NUM_CHANNEL_GROUPS,), f"Expected ({NUM_CHANNEL_GROUPS},), got {frequencies.shape}"
    assert amplitudes.shape == (NUM_CHANNEL_GROUPS,), f"Expected ({NUM_CHANNEL_GROUPS},), got {amplitudes.shape}"

    freq_list = frequencies.astype(np.float32).tolist()
    amp_list = amplitudes.astype(np.float32).tolist()

    packet = struct.pack(STIM_FORMAT, _timestamp_us(), *freq_list, *amp_list)
    return packet


def unpack_stimulation(packet: bytes) -> Tuple[int, np.ndarray, np.ndarray]:
    """Unpack stimulation command packet.

    Returns:
        (timestamp_us, frequencies, amplitudes)
    """
    if len(packet) != STIM_PACKET_SIZE:
        raise ValueError(f"Expected {STIM_PACKET_SIZE} bytes, got {len(packet)}")

    values = struct.unpack(STIM_FORMAT, packet)
    timestamp = values[0]
    frequencies = np.array(values[1:NUM_CHANNEL_GROUPS + 1], dtype=np.float32)
    amplitudes = np.array(values[NUM_CHANNEL_GROUPS + 1:], dtype=np.float32)
    return timestamp, frequencies, amplitudes


def pack_spike_data(spike_counts: np.ndarray) -> bytes:
    """Pack spike counts from 10 channel groups.

    Args:
        spike_counts: (10,) array of spike counts per group

    Returns:
        48-byte binary packet
    """
    assert spike_counts.shape == (NUM_CHANNEL_GROUPS,), f"Expected ({NUM_CHANNEL_GROUPS},), got {spike_counts.shape}"

    count_list = spike_counts.astype(np.float32).tolist()
    packet = struct.pack(SPIKE_FORMAT, _timestamp_us(), *count_list)
    return packet


def unpack_spike_data(packet: bytes) -> Tuple[int, np.ndarray]:
    """Unpack spike count packet.

    Returns:
        (timestamp_us, spike_counts)
    """
    if len(packet) != SPIKE_PACKET_SIZE:
        raise ValueError(f"Expected {SPIKE_PACKET_SIZE} bytes, got {len(packet)}")

    values = struct.unpack(SPIKE_FORMAT, packet)
    timestamp = values[0]
    spike_counts = np.array(values[1:], dtype=np.float32)
    return timestamp, spike_counts


def pack_reward_feedback(reward_type: int, value: float, channels: List[int]) -> bytes:
    """Pack reward feedback for CL1 stimulation.

    Args:
        reward_type: REWARD_POSITIVE, REWARD_NEGATIVE, or REWARD_NEUTRAL
        value: reward magnitude (0.0-1.0)
        channels: up to 3 electrode indices for reward stimulation

    Returns:
        16-byte binary packet
    """
    ch = [0, 0, 0]
    for i, c in enumerate(channels[:3]):
        ch[i] = c

    packet = struct.pack(REWARD_FORMAT, _timestamp_us(), reward_type, value, *ch)
    return packet


def unpack_reward_feedback(packet: bytes) -> Tuple[int, int, float, List[int]]:
    """Unpack reward feedback packet.

    Returns:
        (timestamp_us, reward_type, value, channels)
    """
    if len(packet) != REWARD_PACKET_SIZE:
        raise ValueError(f"Expected {REWARD_PACKET_SIZE} bytes, got {len(packet)}")

    values = struct.unpack(REWARD_FORMAT, packet)
    timestamp = values[0]
    reward_type = values[1]
    value = values[2]
    channels = [values[3], values[4], values[5]]
    return timestamp, reward_type, value, channels


def get_latency_ms(packet_timestamp_us: int) -> float:
    """Calculate latency from packet timestamp to now in milliseconds."""
    return (_timestamp_us() - packet_timestamp_us) / 1000.0
