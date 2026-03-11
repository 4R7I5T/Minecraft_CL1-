"""CL1 hardware interface adapted for Minecraft.

Supports two connection modes:
- UDP: Direct connection to local CL1 device (original protocol)
- Cloud: Remote connection via Jupyter kernel WebSocket (cloud-hosted CL1)

Both modes provide the same stimulate/read/reward API.
"""

import logging
import socket
import time
import numpy as np
from typing import Dict, Optional, Tuple

from ..config import CL1Config
from .channel_mapping import (
    CHANNEL_GROUPS, GROUP_NAMES, NUM_CHANNEL_GROUPS,
    ACTION_GROUP_TO_INDEX, get_all_group_infos,
)
from .udp_protocol import (
    pack_stimulation, unpack_spike_data,
    pack_reward_feedback, REWARD_POSITIVE, REWARD_NEGATIVE,
)

logger = logging.getLogger(__name__)


class MCCL1Interface:
    """Interface to CL1 biological neural hardware for Minecraft.

    Supports both local UDP and cloud WebSocket connections.
    """

    def __init__(self, config: CL1Config):
        self.config = config
        self._stim_socket: Optional[socket.socket] = None
        self._spike_socket: Optional[socket.socket] = None
        self._cloud_bridge = None
        self._mode: Optional[str] = None  # "udp" or "cloud"
        self._connected = False
        self._stim_count = 0
        self._spike_count = 0
        self._last_spike_data: Optional[np.ndarray] = None
        self._latency_ms = 0.0

    def connect(self, stim_port: int = 5001, spike_port: int = 5002):
        """Open UDP sockets for local CL1 stimulation and spike data."""
        try:
            self._stim_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._spike_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._spike_socket.bind(("0.0.0.0", spike_port))
            self._spike_socket.settimeout(self.config.recording_timeout_ms / 1000.0)
            self._connected = True
            self._mode = "udp"
            logger.info(f"CL1 connected (UDP): stim->{self.config.host}:{stim_port}, spikes<-:{spike_port}")
        except OSError as e:
            logger.error(f"CL1 UDP connection failed: {e}")
            self._connected = False
            raise

    def connect_cloud(
        self,
        host: Optional[str] = None,
        kernel_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Connect to a cloud-hosted CL1 device via Jupyter kernel."""
        from .cloud_bridge import CL1CloudBridge, DEFAULT_CL1_HOST

        bridge = CL1CloudBridge(
            host=host or DEFAULT_CL1_HOST,
            kernel_id=kernel_id,
            token=token,
            loop_hz=100,
        )
        bridge.connect()
        self._cloud_bridge = bridge
        self._connected = True
        self._mode = "cloud"
        logger.info(f"CL1 connected (cloud): {bridge.host}")

    def disconnect(self):
        """Close connection (UDP or cloud)."""
        if self._mode == "cloud" and self._cloud_bridge:
            self._cloud_bridge.disconnect()
            self._cloud_bridge = None
        else:
            if self._stim_socket:
                self._stim_socket.close()
            if self._spike_socket:
                self._spike_socket.close()
        self._connected = False
        self._mode = None
        logger.info("CL1 disconnected")

    def stimulate(self, frequencies: np.ndarray, amplitudes: np.ndarray) -> bool:
        """Send stimulation command to CL1.

        Args:
            frequencies: (10,) array of Hz values for each channel group
            amplitudes: (10,) array of amplitude values for each group

        Returns:
            True if stimulation sent successfully
        """
        if not self._connected:
            return False

        # Clip amplitudes to hardware limits
        amplitudes = np.clip(
            amplitudes,
            self.config.min_amplitude_mv,
            self.config.max_amplitude_mv,
        )

        if self._mode == "cloud":
            # Cloud mode: stim is combined with read in stimulate_and_read
            self._pending_stim = (frequencies.copy(), amplitudes.copy())
            return True

        # UDP mode
        packet = pack_stimulation(frequencies, amplitudes)
        try:
            self._stim_socket.sendto(packet, (self.config.host, self.config.port))
            self._stim_count += 1
            return True
        except OSError as e:
            logger.warning(f"Stimulation send failed: {e}")
            return False

    def read_spikes(self) -> Optional[np.ndarray]:
        """Read spike counts from CL1.

        Returns:
            (10,) array of spike counts per channel group, or None on timeout
        """
        if not self._connected:
            return None

        if self._mode == "cloud":
            # Cloud mode: use pending stim params
            freqs, amps = getattr(self, "_pending_stim", (np.zeros(10), np.zeros(10)))
            result = self._cloud_bridge.stimulate_and_read(freqs, amps)
            if result is not None:
                self._spike_count += 1
                self._last_spike_data = result
                self._latency_ms = self._cloud_bridge._latency_ms
            return result

        # UDP mode
        try:
            data, addr = self._spike_socket.recvfrom(1024)
            timestamp, spike_counts = unpack_spike_data(data)
            from .udp_protocol import get_latency_ms
            self._latency_ms = get_latency_ms(timestamp)
            self._spike_count += 1
            self._last_spike_data = spike_counts
            return spike_counts
        except socket.timeout:
            return None
        except Exception as e:
            logger.warning(f"Spike read error: {e}")
            return None

    def stimulate_and_read(
        self, frequencies: np.ndarray, amplitudes: np.ndarray
    ) -> Optional[np.ndarray]:
        """Combined stimulate + read cycle for one tick."""
        if self._mode == "cloud":
            amplitudes = np.clip(
                amplitudes,
                self.config.min_amplitude_mv,
                self.config.max_amplitude_mv,
            )
            result = self._cloud_bridge.stimulate_and_read(frequencies, amplitudes)
            if result is not None:
                self._stim_count += 1
                self._spike_count += 1
                self._last_spike_data = result
                self._latency_ms = self._cloud_bridge._latency_ms
            return result

        if self.stimulate(frequencies, amplitudes):
            return self.read_spikes()
        return None

    def send_reward(self, reward: float):
        """Send reward feedback stimulation."""
        if not self._connected:
            return

        if self._mode == "cloud":
            self._cloud_bridge.send_reward(reward)
            return

        if self._stim_socket is None:
            return

        if reward > 0:
            channels = CHANNEL_GROUPS["reward_pos"]
            reward_type = REWARD_POSITIVE
        elif reward < 0:
            channels = CHANNEL_GROUPS["reward_neg"]
            reward_type = REWARD_NEGATIVE
        else:
            return

        packet = pack_reward_feedback(reward_type, abs(reward), channels)
        try:
            self._stim_socket.sendto(packet, (self.config.host, self.config.port))
        except OSError as e:
            logger.warning(f"Reward feedback send failed: {e}")

    def encode_observation(self, obs_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a 59-dim observation into stimulation parameters.

        Maps observation values to frequency/amplitude pairs for the
        encoding channel group. Non-encoding groups get zero stimulation.

        Args:
            obs_vector: (59,) normalized observation

        Returns:
            (frequencies, amplitudes) each of shape (10,)
        """
        frequencies = np.zeros(NUM_CHANNEL_GROUPS, dtype=np.float32)
        amplitudes = np.zeros(NUM_CHANNEL_GROUPS, dtype=np.float32)

        # Use first 8 observation dims for encoding channels
        encoding_obs = obs_vector[:8]

        # Map to frequency range [5, 50] Hz based on observation magnitude
        freq_base = 5.0 + np.mean(np.abs(encoding_obs)) * 45.0
        frequencies[0] = freq_base

        # Map to amplitude range [min, max] mV
        amp_range = self.config.max_amplitude_mv - self.config.min_amplitude_mv
        amp_base = self.config.min_amplitude_mv + np.mean(np.abs(encoding_obs)) * amp_range
        amplitudes[0] = amp_base

        return frequencies, amplitudes

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def get_stats(self) -> Dict[str, float]:
        stats = {
            "stim_count": self._stim_count,
            "spike_count": self._spike_count,
            "latency_ms": self._latency_ms,
            "connected": float(self._connected),
            "mode": self._mode or "none",
        }
        if self._mode == "cloud" and self._cloud_bridge:
            stats.update(self._cloud_bridge.get_stats())
        return stats
