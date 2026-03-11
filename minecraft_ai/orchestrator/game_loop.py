"""Main 10Hz tick loop orchestrating the Minecraft CL1/Izhikevich AI system."""

import logging
import signal
import time
from typing import Dict, Optional

from ..brains.base_brain import BrainObservation
from ..config import MinecraftCL1Config
from ..environment.entity_tracker import EntityTracker
from ..environment.mc_state import EntityState, PlayerState, WorldSnapshot
from ..environment.observation_builder import ObservationBuilder
from ..environment.reward_shaper import RewardShaper, HostileMobRewardShaper
from .bot_bridge import BotBridge
from .entity_brain_manager import EntityBrainManager
from ..environment.mc_actions import action_index_to_command

logger = logging.getLogger(__name__)


class GameLoop:
    """Main orchestration loop running at 10Hz.

    Each tick:
    1. Receive world state from bot server
    2. Track entity spawns/despawns
    3. Build observations for each entity
    4. Run all brains in parallel (thread pool)
    5. Send actions back to bot server
    """

    def __init__(self, config: Optional[MinecraftCL1Config] = None):
        self.config = config or MinecraftCL1Config()

        # Components
        self.bridge = BotBridge(self.config.bot_server)
        self.tracker = EntityTracker(
            despawn_timeout_s=self.config.game.entity_despawn_timeout_s,
            max_entities=self.config.game.max_entities,
        )
        self.brain_manager = EntityBrainManager(self.config.game)
        self.obs_builder = ObservationBuilder()
        self.reward_shaper = RewardShaper(self.config.reward)
        self.mob_reward_shaper = HostileMobRewardShaper(self.config.reward)

        # State
        self._running = False
        self._tick = 0
        self._tick_interval = 1.0 / self.config.game.tick_rate_hz
        self._last_snapshot: Optional[WorldSnapshot] = None

        # Wire up entity tracking callbacks
        self.tracker.on_spawn(self.brain_manager.on_entity_spawn)
        self.tracker.on_despawn(self.brain_manager.on_entity_despawn)

    def start(self):
        """Connect to bot server and start the game loop."""
        logger.info("Starting Minecraft CL1/Izhikevich AI system...")

        # Connect to bot server
        self.bridge.connect()
        self.bridge.on_state(self._on_state_update)
        self.bridge.on_event(self._on_game_event)

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        logger.info(f"Game loop started at {self.config.game.tick_rate_hz}Hz")

        try:
            self._run_loop()
        finally:
            self.stop()

    def _run_loop(self):
        """Main tick loop."""
        while self._running:
            tick_start = time.time()

            try:
                self._process_tick()
            except Exception as e:
                logger.error(f"Tick {self._tick} error: {e}", exc_info=True)

            # Maintain tick rate
            elapsed = time.time() - tick_start
            sleep_time = self._tick_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > self._tick_interval * 2:
                logger.warning(
                    f"Tick {self._tick} took {elapsed*1000:.0f}ms "
                    f"(target: {self._tick_interval*1000:.0f}ms)"
                )

            self._tick += 1

    def _process_tick(self):
        """Process a single game tick."""
        # 1. Get latest world state
        raw_state = self.bridge.get_latest_state()
        if raw_state is None:
            return

        # 2. Build snapshot
        snapshot = self._build_snapshot(raw_state)
        if snapshot is None:
            return

        # 3. Update entity tracker
        entities = [
            EntityState(
                entity_id=e.get("id", 0),
                entity_type=e.get("type", "unknown"),
                x=e.get("x", 0.0),
                y=e.get("y", 0.0),
                z=e.get("z", 0.0),
                health=e.get("health", 20.0),
                is_hostile=e.get("isHostile", False),
                distance=e.get("distance", 0.0),
            )
            for e in raw_state.get("entities", [])
        ]
        self.tracker.update(entities)

        # 4. Build observations for all tracked entities
        observations: Dict[int, BrainObservation] = {}
        rewards: Dict[int, float] = {}
        all_entities = self.tracker.get_all_entities()

        for entity in all_entities:
            if entity.entity_id not in self.brain_manager.active_ids:
                continue

            obs_vector = self.obs_builder.build_from_entity_perspective(
                entity=entity,
                all_entities=all_entities,
                time_of_day=snapshot.player.time_of_day,
                light_level=snapshot.player.light_level,
            )

            observations[entity.entity_id] = BrainObservation(
                obs_vector=obs_vector,
                tick=self._tick,
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
            )

            # Compute rewards
            if entity.is_hostile:
                target_dist = self._nearest_player_distance(entity, all_entities)
                rewards[entity.entity_id] = self.mob_reward_shaper.compute_mob_reward(
                    mob_health=entity.health,
                    target_distance=target_dist,
                    events=raw_state.get("events", {}),
                )
            else:
                rewards[entity.entity_id] = self.reward_shaper.compute_reward(
                    snapshot,
                    events=raw_state.get("events", {}),
                )

        # 5. Run all brains in parallel
        actions = self.brain_manager.process_tick(observations, rewards)

        # 6. Send actions to bot server
        for entity_id, brain_action in actions.items():
            cmd = action_index_to_command(
                brain_action.action_idx,
                intensity=brain_action.intensity,
            )
            self.bridge.send_action(cmd, bot_id=str(entity_id))

        self._last_snapshot = snapshot

    def _build_snapshot(self, raw_state: dict) -> Optional[WorldSnapshot]:
        """Build a WorldSnapshot from raw bot server state."""
        player_data = raw_state.get("player")
        if player_data is None:
            return None

        pos = player_data.get("position", {})
        vel = player_data.get("velocity", {})

        player = PlayerState(
            health=player_data.get("health", 20.0),
            hunger=player_data.get("food", 20.0),
            x=pos.get("x", 0.0),
            y=pos.get("y", 64.0),
            z=pos.get("z", 0.0),
            vx=vel.get("x", 0.0),
            vy=vel.get("y", 0.0),
            vz=vel.get("z", 0.0),
            yaw=player_data.get("yaw", 0.0),
            pitch=player_data.get("pitch", 0.0),
            time_of_day=player_data.get("timeOfDay", 0) / 24000.0,
            light_level=player_data.get("lightLevel", 15),
            is_on_ground=player_data.get("onGround", True),
            is_in_water=player_data.get("isInWater", False),
            is_sneaking=player_data.get("isSneaking", False),
            held_item_id=player_data.get("heldItem", 0),
        )

        return WorldSnapshot(tick=self._tick, player=player)

    def _nearest_player_distance(
        self, mob: EntityState, all_entities: list
    ) -> float:
        """Find distance to nearest player-type entity."""
        import numpy as np
        mob_pos = np.array([mob.x, mob.y, mob.z])
        min_dist = 100.0

        for ent in all_entities:
            if ent.entity_type == "player" and ent.entity_id != mob.entity_id:
                ent_pos = np.array([ent.x, ent.y, ent.z])
                dist = float(np.linalg.norm(ent_pos - mob_pos))
                min_dist = min(min_dist, dist)

        return min_dist

    def _on_state_update(self, state: dict):
        """Handle real-time state updates from WebSocket."""
        pass  # State is consumed in _process_tick via get_latest_state

    def _on_game_event(self, event: dict):
        """Handle game events (kills, deaths, etc.)."""
        event_type = event.get("type", "")
        logger.debug(f"Game event: {event_type}")

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def stop(self):
        """Shut down all components."""
        self._running = False
        self.brain_manager.shutdown()
        self.bridge.disconnect()
        logger.info(f"Game loop stopped after {self._tick} ticks")

    def get_stats(self) -> dict:
        return {
            "tick": self._tick,
            "brain_stats": self.brain_manager.get_stats(),
            "entity_stats": self.tracker.get_stats(),
            "bridge_connected": self.bridge.is_connected,
        }
