#!/usr/bin/env python3
"""Neuron-driven Minecraft bot with closed-loop learning.

Full DishBrain-style loop:
  1. SENSE:  Encode world state -> stimulate encoding channels (1,2,3,5,6,8,9,10)
  2. ACT:    Read spikes from active motor channels (jump, attack, look)
  3. REWARD: Stimulate reward channels based on outcome
             reward_pos (57,58,59) for good, reward_neg (60,61,62) for bad

Over time neurons learn: sensory pattern -> spike output -> reward association.
"""

import json
import logging
import math
import signal
import sys
import time

import numpy as np
import requests

sys.path.insert(0, "/Users/chi/Documents/Dev/BioLLM_MC/minecraft_CL1")

from minecraft_ai.cl1.cloud_bridge import CL1CloudBridge
from minecraft_ai.cl1.channel_mapping import CHANNEL_GROUPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neurons")

BOT_URL = "http://127.0.0.1:3002"

# Channel assignments
ENC_CHANNELS = CHANNEL_GROUPS["encoding"]       # [1,2,3,5,6,8,9,10] sensory input
JUMP_CHANNELS = CHANNEL_GROUPS["jump_sneak"]     # [51,52,53,54,55] motor output
ATTACK_CHANNELS = CHANNEL_GROUPS["attack"]       # [41,42,43,44,45] motor output
LOOK_L_CHANNELS = CHANNEL_GROUPS["look_l"]       # [31,32,33,34,35] motor output
LOOK_R_CHANNELS = CHANNEL_GROUPS["look_r"]       # [36,37,38,39,40] motor output
MOVE_FWD_CHANNELS = CHANNEL_GROUPS["move_fwd"]   # [11,12,13,14,15] motor output
REWARD_POS = CHANNEL_GROUPS["reward_pos"]        # [57,58,59]
REWARD_NEG = CHANNEL_GROUPS["reward_neg"]        # [60,61,62]

# All motor channels we read from
MOTOR_GROUPS = {
    "jump":    JUMP_CHANNELS,
    "attack":  ATTACK_CHANNELS,
    "look_l":  LOOK_L_CHANNELS,
    "look_r":  LOOK_R_CHANNELS,
    "move_fwd": MOVE_FWD_CHANNELS,
}

running = True


def signal_handler(sig, frame):
    global running
    running = False


def get_bot_state():
    try:
        r = requests.get(f"{BOT_URL}/state/default", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def send_compound(commands):
    try:
        requests.post(
            f"{BOT_URL}/action",
            json={"botId": "default", "compound": True, "commands": commands},
            timeout=1,
        )
    except Exception:
        pass


def encode_world(state):
    """Encode world state into per-electrode stimulation amplitudes (uA).

    8 encoding electrodes, each encodes a different aspect:
      ch 1: nearest entity distance (close=strong)
      ch 2: nearest entity direction (left=neg, right=pos)
      ch 3: health level
      ch 5: hunger level
      ch 6: forward velocity
      ch 8: is on ground
      ch 9: time of day
      ch 10: number of nearby entities
    """
    p = state.get("player", {})
    pos = p.get("position", {})
    vel = p.get("velocity", {})
    entities = state.get("entities", [])
    entities.sort(key=lambda e: e.get("distance", 999))

    amps = {}  # electrode -> amplitude in uA

    # Nearest entity distance: closer = stronger stim (0-3 uA)
    if entities:
        nearest = entities[0]
        dist = nearest.get("distance", 50)
        amps[1] = max(0, 3.0 - dist / 10.0)  # 3uA at 0m, 0 at 30m

        # Direction: angle to nearest entity
        dx = nearest.get("x", 0) - pos.get("x", 0)
        dz = nearest.get("z", 0) - pos.get("z", 0)
        angle = math.atan2(dx, dz)
        bot_yaw = p.get("yaw", 0)
        rel_angle = angle - bot_yaw
        while rel_angle > math.pi: rel_angle -= 2 * math.pi
        while rel_angle < -math.pi: rel_angle += 2 * math.pi
        # Encode as amplitude: center=0, left/right = stronger
        amps[2] = abs(rel_angle) / math.pi * 2.5
    else:
        amps[1] = 0.0
        amps[2] = 0.0

    # Health: low health = strong stim (alarm signal)
    health = p.get("health", 20) / 20.0
    amps[3] = (1.0 - health) * 3.0

    # Hunger
    hunger = p.get("food", 20) / 20.0
    amps[5] = (1.0 - hunger) * 2.0

    # Forward velocity: moving = stim
    vx = vel.get("x", 0)
    vz = vel.get("z", 0)
    speed = math.sqrt(vx*vx + vz*vz)
    amps[6] = min(speed * 5.0, 2.5)

    # On ground
    amps[8] = 2.0 if p.get("onGround", True) else 0.5

    # Time of day (day=low, night=high)
    tod = p.get("timeOfDay", 0) / 24000.0
    night = 1.0 if (tod > 0.5) else 0.0
    amps[9] = night * 2.5

    # Entity count nearby
    nearby_count = sum(1 for e in entities if e.get("distance", 999) < 16)
    amps[10] = min(nearby_count * 0.8, 3.0)

    return amps


def compute_reward(prev_state, curr_state, action_taken):
    """Compute reward based on what changed between ticks.

    Rewards (positive):
      +2.0  moved to new position (exploration)
      +3.0  got closer to nearest entity (engagement)
      +1.0  any spike-driven action taken (activity bonus)

    Punishments (negative):
      -2.0  took damage
      -1.0  stayed completely still
      -0.5  health low
    """
    if prev_state is None:
        return 0.0

    reward = 0.0
    pp = prev_state["player"]
    cp = curr_state["player"]
    ppos = pp["position"]
    cpos = cp["position"]

    # Movement reward
    dx = cpos["x"] - ppos["x"]
    dz = cpos["z"] - ppos["z"]
    dist_moved = math.sqrt(dx*dx + dz*dz)

    if dist_moved > 0.3:
        reward += 2.0  # moved
    elif dist_moved < 0.05:
        reward -= 1.0  # stuck

    # Getting closer to entities
    prev_ents = prev_state.get("entities", [])
    curr_ents = curr_state.get("entities", [])
    if prev_ents and curr_ents:
        prev_nearest = min(e.get("distance", 999) for e in prev_ents)
        curr_nearest = min(e.get("distance", 999) for e in curr_ents)
        if curr_nearest < prev_nearest - 0.5:
            reward += 3.0  # approaching entity

    # Damage taken
    if cp.get("health", 20) < pp.get("health", 20):
        reward -= 2.0

    # Low health alarm
    if cp.get("health", 20) < 10:
        reward -= 0.5

    # Activity bonus: neurons produced an action
    if action_taken:
        reward += 1.0

    return np.clip(reward, -5.0, 5.0)


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bridge = CL1CloudBridge(loop_hz=100)

    print("=" * 55)
    print("  NEURON BOT v3 - Closed-Loop Learning")
    print("  Sense -> Act -> Reward")
    print("  Connecting to biological neurons...")
    print("=" * 55)

    try:
        bridge.connect()
    except Exception as e:
        print(f"Failed: {e}")
        return

    print("Live! Full closed-loop: sense -> spike -> act -> reward\n")

    tick = 0
    total_spikes = 0
    total_reward = 0.0
    prev_state = None
    reward_history = []

    while running:
        t0 = time.time()
        tick += 1

        state = get_bot_state()
        if not state:
            time.sleep(0.5)
            continue

        # === 1. SENSE: Encode world into encoding channel stimulation ===
        stim_amps = encode_world(state)

        # Build stim command: stimulate each encoding electrode with its amplitude
        stim_parts = []
        for ch, amp in stim_amps.items():
            if amp > 0.1:
                stim_parts.append(f"_n.stim(cl.ChannelSet({ch}),cl.StimDesign(180,-{amp:.2f},180,{amp:.2f}))")

        stim_code = ";".join(stim_parts) if stim_parts else "pass"

        # === 2. ACT: Read spikes from motor channels ===
        # Build channel -> group lookup
        ch_to_group = {}
        for gname, channels in MOTOR_GROUPS.items():
            for ch in channels:
                ch_to_group[ch] = gname

        code = (
            f"import cl,json\n"
            f"{stim_code}\n"
            f"_m={repr(ch_to_group)}\n"
            f"_r={{k:0 for k in {repr(list(MOTOR_GROUPS.keys()))}}}\n"
            f"for _t in _n.loop(500,stop_after_seconds=0.1,ignore_jitter=True):\n"
            f" for _s in _t.analysis.spikes:\n"
            f"  _g=_m.get(_s.channel)\n"
            f"  if _g:_r[_g]+=1\n"
            f"print(json.dumps(_r))"
        )

        try:
            result = bridge._execute(code, timeout=10)
        except Exception as e:
            log.warning(f"Tick {tick} exec error: {e}")
            continue

        # Parse motor spikes
        try:
            lines = [l for l in result.strip().split("\n") if l.strip().startswith("{")]
            if not lines:
                continue
            motor = json.loads(lines[-1])
        except (json.JSONDecodeError, ValueError):
            continue

        # Map spikes to actions
        commands = []
        fired = []

        if motor.get("jump", 0) > 0:
            commands.append({"action": "jump_sneak", "intensity": min(motor["jump"] / 3, 1.0)})
            fired.append(f"J={motor['jump']}")

        if motor.get("attack", 0) > 0:
            commands.append({"action": "attack", "intensity": min(motor["attack"] / 2, 1.0)})
            fired.append(f"A={motor['attack']}")

        if motor.get("look_l", 0) > 0:
            commands.append({"action": "look_left", "intensity": min(motor["look_l"] / 2, 1.0)})
            fired.append(f"Ll={motor['look_l']}")

        if motor.get("look_r", 0) > 0:
            commands.append({"action": "look_right", "intensity": min(motor["look_r"] / 2, 1.0)})
            fired.append(f"Lr={motor['look_r']}")

        if motor.get("move_fwd", 0) > 0:
            commands.append({"action": "move_forward", "intensity": min(motor["move_fwd"] / 2, 1.0), "duration": 2})
            fired.append(f"F={motor['move_fwd']}")

        action_taken = len(commands) > 0

        # No spikes -> drift forward
        if not commands:
            commands.append({"action": "move_forward", "intensity": 0.3, "duration": 1})

        send_compound(commands)

        # === 3. REWARD: Compute outcome and stimulate reward channels ===
        reward = compute_reward(prev_state, state, action_taken)
        total_reward += reward
        reward_history.append(reward)
        if len(reward_history) > 20:
            reward_history.pop(0)

        # Send reward stimulation to neurons
        if abs(reward) > 0.5:
            if reward > 0:
                channels = REWARD_POS
                amp = min(abs(reward) * 0.8, 4.0)
            else:
                channels = REWARD_NEG
                amp = min(abs(reward) * 0.8, 4.0)

            reward_code = (
                f"_n.stim(cl.ChannelSet(*{channels}),"
                f"cl.StimDesign(180,-{amp:.2f},180,{amp:.2f}),"
                f"cl.BurstDesign(3,50))"
            )
            try:
                bridge._execute(reward_code, timeout=5)
            except Exception:
                pass

        prev_state = state

        # Log
        spk_total = sum(motor.values())
        total_spikes += spk_total
        dt = (time.time() - t0) * 1000
        pos = state["player"]["position"]
        neural_str = " ".join(fired) if fired else "drift"
        avg_reward = np.mean(reward_history) if reward_history else 0

        reward_sym = "+" if reward > 0.5 else ("-" if reward < -0.5 else ".")

        log.info(
            f"T{tick:03d} {reward_sym} | {neural_str:<25s} | "
            f"r={reward:+.1f} avg={avg_reward:+.1f} | "
            f"({pos['x']:.0f},{pos['y']:.0f},{pos['z']:.0f}) | "
            f"{dt:.0f}ms"
        )

    # Summary
    print(f"\n{'='*55}")
    print(f"  {tick} ticks | {total_spikes} spikes | reward={total_reward:+.1f}")
    print(f"  Avg reward/tick: {total_reward/max(tick,1):+.2f}")
    print(f"{'='*55}")
    bridge.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
