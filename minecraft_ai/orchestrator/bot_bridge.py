"""HTTP/WebSocket bridge to the Node.js mineflayer bot server."""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import requests
import websocket

from ..config import BotServerConfig
from ..environment.mc_actions import BotCommand, CompoundAction

logger = logging.getLogger(__name__)


class BotBridge:
    """Manages communication with the Node.js bot server.

    Uses HTTP REST for commands and WebSocket for real-time state updates.
    """

    def __init__(self, config: Optional[BotServerConfig] = None):
        self.config = config or BotServerConfig()
        self._base_url = f"http://{self.config.host}:{self.config.http_port}"
        self._ws_url = f"ws://{self.config.host}:{self.config.ws_port}"
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._connected = False
        self._state_callback: Optional[Callable[[dict], None]] = None
        self._event_callback: Optional[Callable[[dict], None]] = None
        self._latest_state: Optional[dict] = None
        self._lock = threading.Lock()

    def connect(self):
        """Connect to the bot server via WebSocket."""
        self._ws = websocket.WebSocketApp(
            self._ws_url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open,
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True,
        )
        self._ws_thread.start()

        # Wait for connection
        deadline = time.time() + 5.0
        while not self._connected and time.time() < deadline:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError("Failed to connect to bot server WebSocket")

        logger.info(f"Connected to bot server at {self._ws_url}")

    def disconnect(self):
        """Disconnect from bot server."""
        if self._ws:
            self._ws.close()
        self._connected = False

    def on_state(self, callback: Callable[[dict], None]):
        """Register callback for state updates."""
        self._state_callback = callback

    def on_event(self, callback: Callable[[dict], None]):
        """Register callback for game events."""
        self._event_callback = callback

    def send_action(self, command: BotCommand, bot_id: str = "default") -> bool:
        """Send a single action command to the bot server.

        Args:
            command: Action command to send
            bot_id: Target bot identifier

        Returns:
            True if the command was accepted
        """
        payload = {
            "botId": bot_id,
            **command.to_dict(),
        }
        return self._post("/action", payload)

    def send_compound_action(
        self, action: CompoundAction, bot_id: str = "default"
    ) -> bool:
        """Send multiple simultaneous actions."""
        payload = {
            "botId": bot_id,
            **action.to_dict(),
        }
        return self._post("/action", payload)

    def get_state(self, bot_id: str = "default") -> Optional[dict]:
        """Get current bot state via HTTP (fallback if WebSocket unavailable)."""
        try:
            resp = requests.get(
                f"{self._base_url}/state/{bot_id}",
                timeout=2.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException as e:
            logger.warning(f"State request failed: {e}")
        return None

    def get_latest_state(self) -> Optional[dict]:
        """Get the most recent state from WebSocket updates."""
        with self._lock:
            return self._latest_state

    def spawn_bot(self, bot_id: str, options: Optional[dict] = None) -> bool:
        """Request the bot server to spawn a new bot."""
        payload = {"botId": bot_id, **(options or {})}
        return self._post("/bots/spawn", payload)

    def despawn_bot(self, bot_id: str) -> bool:
        """Request the bot server to remove a bot."""
        return self._post("/bots/despawn", {"botId": bot_id})

    def list_bots(self) -> List[str]:
        """Get list of active bot IDs."""
        try:
            resp = requests.get(f"{self._base_url}/bots", timeout=2.0)
            if resp.status_code == 200:
                return resp.json().get("bots", [])
        except requests.RequestException:
            pass
        return []

    def _post(self, path: str, payload: dict) -> bool:
        """Send a POST request to the bot server."""
        try:
            resp = requests.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=2.0,
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"POST {path} failed: {e}")
            return False

    def _on_ws_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "state":
                with self._lock:
                    self._latest_state = data.get("data")
                if self._state_callback:
                    self._state_callback(data.get("data", {}))

            elif msg_type == "event":
                if self._event_callback:
                    self._event_callback(data.get("data", {}))

        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WebSocket")

    def _on_ws_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status, close_msg):
        self._connected = False
        logger.info("WebSocket closed")

    def _on_ws_open(self, ws):
        self._connected = True
        logger.info("WebSocket connected")

    @property
    def is_connected(self) -> bool:
        return self._connected
