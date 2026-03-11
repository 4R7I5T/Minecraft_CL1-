"""Cloud bridge for CL1 device access via Jupyter kernel WebSocket.

Connects to a CL1 device's Jupyter kernel through Cloudflare Access
and executes stimulation/spike-reading code remotely. This replaces
the UDP protocol for cloud-hosted CL1 devices.

Architecture:
    Local Python -> WebSocket -> Jupyter Kernel -> cl module -> Neurons
"""

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import websocket
except ImportError:
    websocket = None

from .channel_mapping import CHANNEL_GROUPS, GROUP_NAMES, NUM_CHANNEL_GROUPS

logger = logging.getLogger(__name__)

# Default CL1 cloud device hostname
DEFAULT_CL1_HOST = "cl1-2544-144.device.cloud.corticallabs-test.com"


class CL1CloudBridge:
    """Bridge to CL1 hardware via Jupyter kernel WebSocket.

    Executes Python code on the CL1 device through the Jupyter kernel
    messaging protocol. Manages a persistent cl.open() session and
    provides stimulate/read_spikes operations.
    """

    def __init__(
        self,
        host: str = DEFAULT_CL1_HOST,
        kernel_id: Optional[str] = None,
        token: Optional[str] = None,
        loop_hz: int = 100,
    ):
        self.host = host
        self.kernel_id = kernel_id
        self.token = token
        self.loop_hz = loop_hz
        self._ws: Optional[Any] = None
        self._session_id = str(uuid.uuid4())
        self._connected = False
        self._neurons_open = False
        self._stim_count = 0
        self._spike_count = 0
        self._latency_ms = 0.0

    def connect(self):
        """Connect to the CL1 device via Jupyter kernel WebSocket."""
        if websocket is None:
            raise ImportError(
                "websocket-client required for cloud bridge. "
                "Install with: pip install websocket-client"
            )

        # Auto-discover token if not provided
        if self.token is None:
            self.token = self._load_cf_token()

        # Auto-discover kernel if not provided
        if self.kernel_id is None:
            self.kernel_id = self._discover_kernel()

        url = f"wss://{self.host}/_/jupyter/api/kernels/{self.kernel_id}/channels"
        headers = [
            f"cf-access-token: {self.token}",
            f"Cookie: CF_Authorization={self.token}",
        ]

        self._ws = websocket.create_connection(url, header=headers, timeout=15)
        self._connected = True
        logger.info(f"Connected to CL1 cloud bridge (kernel={self.kernel_id[:8]}...)")

        # Open persistent neurons session on the device
        self._open_neurons()

    def disconnect(self):
        """Close the neurons session and WebSocket."""
        if self._neurons_open:
            try:
                self._execute(
                    "_cl1_ctx.__exit__(None,None,None)\n"
                    "print('neurons closed')",
                    timeout=5,
                )
            except Exception:
                pass
            self._neurons_open = False

        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected = False
        logger.info("CL1 cloud bridge disconnected")

    def stimulate_and_read(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        read_duration_ms: float = 100.0,
    ) -> Optional[np.ndarray]:
        """Stimulate encoding channels and read spike responses.

        Sends stim command, runs a short loop to collect spikes,
        returns spike counts per action channel group.

        Args:
            frequencies: (10,) Hz values per channel group (only encoding used for stim)
            amplitudes: (10,) amplitude values per group (only encoding used for stim)
            read_duration_ms: How long to read spikes after stim (ms)

        Returns:
            (10,) array of spike counts per action channel group, or None on error
        """
        if not self._connected or not self._neurons_open:
            return None

        encoding_channels = CHANNEL_GROUPS["encoding"]
        freq = float(frequencies[0]) if frequencies[0] > 0 else 0.0
        amp = float(amplitudes[0]) if amplitudes[0] > 0 else 0.0

        # Clip amplitude to safe range (uA)
        amp = max(0.0, min(amp, 5.0))
        read_sec = read_duration_ms / 1000.0

        # Build compact code: stim then short loop to read spikes
        # Pre-build channel->group lookup on first call (int keys!)
        if not hasattr(self, "_ch_lookup_ready"):
            ch_map = {}
            for i, name in enumerate(GROUP_NAMES[:NUM_CHANNEL_GROUPS]):
                if name == "encoding":
                    continue
                for ch in CHANNEL_GROUPS[name]:
                    ch_map[ch] = i
            # Use repr() not json.dumps() to preserve int keys
            self._execute(f"_cg={repr(ch_map)}", timeout=5)
            self._ch_lookup_ready = True

        code = (
            f"_r=[0]*{NUM_CHANNEL_GROUPS}\n"
            f"if {amp}>0:_n.stim(cl.ChannelSet(*{encoding_channels}),"
            f"cl.StimDesign(180,-{amp:.3f},180,{amp:.3f}),"
            f"cl.BurstDesign(max(1,int({freq:.1f}*{read_sec:.3f})),{freq:.1f}))\n"
            f"for _t in _n.loop(500,stop_after_seconds={read_sec},ignore_jitter=True):\n"
            f" for _s in _t.analysis.spikes:\n"
            f"  _i=_cg.get(_s.channel,-1)\n"
            f"  if _i>=0:_r[_i]+=1\n"
            f"print(json.dumps(_r))"
        )

        t0 = time.time()
        try:
            result = self._execute(code, timeout=max(8, read_sec * 3 + 5))
            self._latency_ms = (time.time() - t0) * 1000
        except Exception as e:
            logger.warning(f"Stim+read failed: {e}")
            return None

        # Parse spike counts
        try:
            result = result.strip()
            lines = [l for l in result.split("\n") if l.strip().startswith("[")]
            if lines:
                spike_counts = np.array(json.loads(lines[-1]), dtype=np.float32)
                self._stim_count += 1
                self._spike_count += int(np.sum(spike_counts))
                return spike_counts
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse spike data: {result[:200]}")

        return None

    def send_reward(self, reward: float):
        """Send reward feedback stimulation."""
        if not self._connected or not self._neurons_open:
            return

        if reward == 0:
            return

        if reward > 0:
            channels = CHANNEL_GROUPS["reward_pos"]
        else:
            channels = CHANNEL_GROUPS["reward_neg"]

        amp = min(abs(reward) * 2.0, 5.0)  # Scale reward to uA, max 5

        code = f"""
_n.stim(
    cl.ChannelSet(*{channels}),
    cl.StimDesign(180, -{amp}, 180, {amp}),
    cl.BurstDesign(3, 50))
"""
        try:
            self._execute(code, timeout=5)
        except Exception as e:
            logger.warning(f"Reward stim failed: {e}")

    def get_stats(self) -> Dict[str, float]:
        return {
            "stim_count": self._stim_count,
            "spike_count": self._spike_count,
            "latency_ms": self._latency_ms,
            "connected": float(self._connected),
            "neurons_open": float(self._neurons_open),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._neurons_open

    # --- Internal methods ---

    def _open_neurons(self):
        """Open a persistent cl.open() session on the device."""
        code = """
import cl, json, time
_cl1_ctx = cl.open()
_n = _cl1_ctx.__enter__()
print(f"Neurons opened: {type(_n).__name__}")
print(f"Channels: {_n.get_channel_count()}")
print(f"FPS: {_n.get_frames_per_second()}")
"""
        result = self._execute(code, timeout=15)
        if "Neurons opened" in result:
            self._neurons_open = True
            logger.info(f"CL1 neurons session opened: {result.strip()}")
        else:
            raise ConnectionError(f"Failed to open neurons: {result}")

    def _execute(self, code: str, timeout: float = 15) -> str:
        """Execute Python code on the CL1 device via Jupyter kernel."""
        if not self._ws:
            raise ConnectionError("Not connected to Jupyter kernel")

        msg_id = str(uuid.uuid4())
        msg = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "",
                "session": self._session_id,
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }

        self._ws.send(json.dumps(msg))

        outputs = []
        start = time.time()
        while time.time() - start < timeout:
            try:
                self._ws.settimeout(min(timeout, 8))
                resp = self._ws.recv()
                data = json.loads(resp)
                msg_type = data.get("msg_type", data.get("header", {}).get("msg_type", ""))
                parent_id = data.get("parent_header", {}).get("msg_id", "")

                if parent_id == msg_id:
                    if msg_type == "stream":
                        outputs.append(data["content"]["text"])
                    elif msg_type == "execute_result":
                        outputs.append(data["content"]["data"].get("text/plain", ""))
                    elif msg_type == "error":
                        ename = data["content"]["ename"]
                        evalue = data["content"]["evalue"]
                        raise RuntimeError(f"CL1 execution error: {ename}: {evalue}")
                    elif msg_type == "status":
                        if data["content"]["execution_state"] == "idle":
                            break
            except websocket.WebSocketTimeoutException:
                break

        return "".join(outputs)

    def _load_cf_token(self) -> str:
        """Load Cloudflare Access token from cloudflared cache."""
        token_pattern = Path.home() / ".cloudflared"
        token_files = list(token_pattern.glob(f"{self.host}*-token"))
        if token_files:
            return token_files[0].read_text().strip()
        raise FileNotFoundError(
            f"No CF access token found for {self.host}. "
            "Run: cloudflared access login " + f"https://{self.host}"
        )

    def _discover_kernel(self) -> str:
        """Find an available Jupyter kernel on the device."""
        url = f"https://{self.host}/_/jupyter/api/kernels"

        # Use subprocess curl to avoid SSL/auth issues with urllib
        result = subprocess.run(
            ["curl", "-s",
             "-H", f"cf-access-token: {self.token}",
             "-H", f"cookie: CF_Authorization={self.token}",
             url],
            capture_output=True, text=True, timeout=15,
        )

        if result.returncode != 0:
            raise ConnectionError(f"Failed to query kernels: {result.stderr}")

        kernels = json.loads(result.stdout)

        if not kernels:
            raise ConnectionError("No Jupyter kernels available on CL1 device")

        # Prefer idle kernels
        for k in kernels:
            if k.get("execution_state") == "idle":
                return k["id"]
        return kernels[0]["id"]
