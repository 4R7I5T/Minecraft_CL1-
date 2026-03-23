/**
 * Action type -> mineflayer API calls.
 * Translates brain action commands into bot movement/combat actions.
 * Supports both creative and survival mode actions.
 */

const { Vec3 } = require('vec3');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');

const ACTION_HANDLERS = {
  move_forward(bot, intensity, duration) {
    bot.setControlState('forward', true);
    setTimeout(() => bot.setControlState('forward', false), duration * 100);
  },

  move_backward(bot, intensity, duration) {
    bot.setControlState('back', true);
    setTimeout(() => bot.setControlState('back', false), duration * 100);
  },

  strafe_left(bot, intensity, duration) {
    bot.setControlState('left', true);
    setTimeout(() => bot.setControlState('left', false), duration * 100);
  },

  strafe_right(bot, intensity, duration) {
    bot.setControlState('right', true);
    setTimeout(() => bot.setControlState('right', false), duration * 100);
  },

  look_left(bot, intensity, duration) {
    const yawDelta = intensity * 0.15; // radians
    bot.look(bot.entity.yaw + yawDelta, bot.entity.pitch, false);
  },

  look_right(bot, intensity, duration) {
    const yawDelta = intensity * 0.15;
    bot.look(bot.entity.yaw - yawDelta, bot.entity.pitch, false);
  },

  attack(bot, intensity, duration) {
    try {
      const nearestEntity = bot.nearestEntity((e) => {
        if (!e.position) return false;
        if (!e.isValid) return false;
        const dist = e.position.distanceTo(bot.entity.position);
        return dist < 4.0 && e.type === 'mob';
      });
      if (nearestEntity && nearestEntity.isValid) {
        bot.attack(nearestEntity);
      } else {
        bot.swingArm('right');
      }
    } catch (err) {
      bot.swingArm('right');
    }
  },

  use_item(bot, intensity, duration) {
    bot.activateItem();
    setTimeout(() => bot.deactivateItem(), duration * 100);
  },

  jump_sneak(bot, intensity, duration) {
    if (intensity > 0.5) {
      bot.setControlState('jump', true);
      setTimeout(() => bot.setControlState('jump', false), 200);
    } else {
      bot.setControlState('sneak', true);
      setTimeout(() => bot.setControlState('sneak', false), duration * 100);
    }
  },

  chat(bot, intensity, duration, message) {
    if (message) {
      bot.chat(message);
    }
  },

  // ── Creative Mode Actions ──────────────────────────────────────────

  place_block(bot, intensity, duration, message) {
    // Place block at look target using creative inventory
    // In creative mode, use chat command for precision
    if (message) {
      bot.chat(message);
    } else {
      // Place block where looking
      const block = bot.blockAtCursor(5);
      if (block) {
        const face = bot.blockAtCursorFace(5);
        if (face) {
          bot.placeBlock(block, face.position).catch(() => {});
        }
      }
    }
  },

  select_slot(bot, intensity, duration) {
    // Select hotbar slot based on intensity (0-8)
    const slot = Math.min(8, Math.floor(intensity * 9));
    bot.setQuickBarSlot(slot);
  },

  fly_vertical(bot, intensity, duration) {
    // Creative mode flight: positive intensity = up, negative = down
    if (intensity > 0.5) {
      bot.setControlState('jump', true);
      setTimeout(() => bot.setControlState('jump', false), duration * 100);
    } else if (intensity < -0.5) {
      bot.setControlState('sneak', true);
      setTimeout(() => bot.setControlState('sneak', false), duration * 100);
    }
  },

  start_flying(bot) {
    // Enable creative flight — double-tap jump
    if (bot.creative) {
      bot.creative.startFlying();
    } else {
      // Fallback: double-tap jump to toggle flight
      bot.setControlState('jump', true);
      setTimeout(() => {
        bot.setControlState('jump', false);
        setTimeout(() => {
          bot.setControlState('jump', true);
          setTimeout(() => bot.setControlState('jump', false), 50);
        }, 50);
      }, 50);
    }
  },

  stop_flying(bot) {
    if (bot.creative) {
      bot.creative.stopFlying();
    }
  },

  fly_to(bot, intensity, duration, message) {
    // Fly to coordinates: message = "x y z"
    if (message && bot.creative) {
      const [x, y, z] = message.split(' ').map(Number);
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        bot.creative.startFlying();
        bot.creative.flyTo(new Vec3(x, y, z)).catch((err) => {
          console.log(`fly_to error: ${err.message}`);
        });
      }
    }
  },

  // Physical creative block placement: fly to spot, look at target, equip block, place with arm swing
  creative_place(bot, intensity, duration, message) {
    if (!message) return;
    // message format: "x y z block_type"
    const parts = message.split(' ');
    if (parts.length < 4) return;
    const [x, y, z] = parts.slice(0, 3).map(Number);
    const blockName = parts[3];
    if (isNaN(x) || isNaN(y) || isNaN(z)) return;

    const targetPos = new Vec3(x, y, z);

    (async () => {
      try {
        // 1. Fly near the target (1 block above and 2 blocks away)
        const flyTarget = new Vec3(x + 1.5, y + 1.5, z + 1.5);
        bot.creative.startFlying();
        await bot.creative.flyTo(flyTarget);

        // 2. Look at the target block position
        await bot.lookAt(targetPos.offset(0.5, 0.5, 0.5));

        // 3. Equip the block in hand (slot 36 = first hotbar slot)
        const Item = require('prismarine-item')(bot.registry || bot.version);
        const itemId = bot.registry?.itemsByName[blockName]?.id;
        if (itemId != null) {
          const item = new Item(itemId, 1);
          await bot.creative.setInventorySlot(36, item);
        }

        // 4. Swing arm visually
        bot.swingArm('right');

        // 5. Place via setblock (reliable) — the visual swing makes it look like the bot placed it
        bot.chat(`/setblock ${x} ${y} ${z} minecraft:${blockName}`);

      } catch (err) {
        // Fallback: just setblock
        bot.chat(`/setblock ${x} ${y} ${z} minecraft:${blockName}`);
      }
    })();
  },

  // Look at a specific position
  look_at(bot, intensity, duration, message) {
    if (!message) return;
    const [x, y, z] = message.split(' ').map(Number);
    if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
      bot.lookAt(new Vec3(x, y, z)).catch(() => {});
    }
  },

  // Swing arm (visual feedback)
  swing_arm(bot) {
    bot.swingArm('right');
  },

  // ── Survival Mode Actions ──────────────────────────────────────────

  mine_block(bot, intensity, duration, message) {
    // Mine block at current look target or specified coords
    (async () => {
      try {
        let target;
        if (message) {
          const [x, y, z] = message.split(' ').map(Number);
          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            target = bot.blockAt(new Vec3(x, y, z));
          }
        } else {
          target = bot.blockAtCursor(5);
        }
        if (target && target.name !== 'air') {
          await bot.dig(target);
        }
      } catch (err) {
        // Block may not be diggable
      }
    })();
  },

  craft(bot, intensity, duration, message) {
    // Craft item by name: message = "item_name [count]"
    (async () => {
      try {
        if (!message) return;
        const parts = message.split(' ');
        const itemName = parts[0];
        const count = parseInt(parts[1]) || 1;
        const mcData = require('minecraft-data')(bot.version);
        const item = mcData.itemsByName[itemName];
        if (!item) return;

        const recipes = bot.recipesFor(item.id, null, 1, null);
        if (recipes.length > 0) {
          await bot.craft(recipes[0], count, null);
        }
      } catch (err) {
        // Crafting may fail
      }
    })();
  },

  equip(bot, intensity, duration, message) {
    // Equip item: message = "item_name [destination]"
    // destination: hand, head, torso, legs, feet, off-hand
    (async () => {
      try {
        if (!message) return;
        const parts = message.split(' ');
        const itemName = parts[0];
        const dest = parts[1] || 'hand';
        const item = bot.inventory.items().find(i =>
          i.name === itemName || i.name.includes(itemName)
        );
        if (item) {
          await bot.equip(item, dest);
        }
      } catch (err) {
        // Equip may fail
      }
    })();
  },

  eat(bot, intensity, duration, message) {
    // Eat food item from inventory
    (async () => {
      try {
        const mcData = require('minecraft-data')(bot.version);
        const foodItems = bot.inventory.items().filter(i => {
          const itemData = mcData.itemsByName[i.name];
          return itemData && (itemData.foodPoints > 0 || i.name.includes('apple') ||
            i.name.includes('bread') || i.name.includes('cooked') ||
            i.name.includes('steak') || i.name.includes('porkchop'));
        });
        if (foodItems.length > 0) {
          await bot.equip(foodItems[0], 'hand');
          await bot.consume();
        }
      } catch (err) {
        // Eating may fail
      }
    })();
  },

  deposit_chest(bot, intensity, duration, message) {
    // Deposit items into nearest chest
    (async () => {
      try {
        const chestBlock = bot.findBlock({
          matching: (block) => block.name === 'chest' || block.name === 'trapped_chest',
          maxDistance: 6,
        });
        if (!chestBlock) return;

        const chest = await bot.openContainer(chestBlock);
        const items = bot.inventory.items();
        for (const item of items) {
          try {
            await chest.deposit(item.type, null, item.count);
          } catch (e) {
            // Chest may be full
            break;
          }
        }
        chest.close();
      } catch (err) {
        // Chest interaction may fail
      }
    })();
  },

  pathfind_to(bot, intensity, duration, message) {
    // Pathfind to coordinates or entity: message = "x y z" or "player:Name"
    (async () => {
      try {
        if (!message) return;

        // Ensure pathfinder is loaded
        if (!bot.pathfinder) {
          bot.loadPlugin(pathfinder);
        }
        const mcData = require('minecraft-data')(bot.version);
        const movements = new Movements(bot, mcData);
        movements.canDig = false;
        movements.allow1by1towers = false;
        bot.pathfinder.setMovements(movements);

        if (message.startsWith('player:')) {
          const playerName = message.slice(7);
          const player = bot.players[playerName];
          if (player && player.entity) {
            const goal = new goals.GoalNear(
              player.entity.position.x,
              player.entity.position.y,
              player.entity.position.z,
              3
            );
            bot.pathfinder.setGoal(goal);
          }
        } else {
          const [x, y, z] = message.split(' ').map(Number);
          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            const goal = new goals.GoalNear(x, y, z, 1);
            bot.pathfinder.setGoal(goal);
          }
        }
      } catch (err) {
        console.log(`pathfind_to error: ${err.message}`);
      }
    })();
  },

  stop_pathfind(bot) {
    if (bot.pathfinder) {
      bot.pathfinder.setGoal(null);
    }
  },

  place_block_survival(bot, intensity, duration, message) {
    // Place block from inventory at target position
    (async () => {
      try {
        if (!message) return;
        const parts = message.split(' ');
        const [x, y, z] = parts.slice(0, 3).map(Number);
        if (isNaN(x) || isNaN(y) || isNaN(z)) return;

        // Find a reference block adjacent to target
        const targetPos = new Vec3(x, y, z);
        const below = bot.blockAt(targetPos.offset(0, -1, 0));
        if (below && below.name !== 'air') {
          const itemName = parts[3] || null;
          if (itemName) {
            const item = bot.inventory.items().find(i => i.name.includes(itemName));
            if (item) await bot.equip(item, 'hand');
          }
          await bot.placeBlock(below, new Vec3(0, 1, 0));
        }
      } catch (err) {
        // Placement may fail
      }
    })();
  },

  interact(bot, intensity, duration, message) {
    // Interact with nearest entity or block
    (async () => {
      try {
        if (message === 'block') {
          const block = bot.blockAtCursor(5);
          if (block) {
            await bot.activateBlock(block);
          }
        } else {
          const nearestEntity = bot.nearestEntity((e) => {
            if (!e.position) return false;
            return e.position.distanceTo(bot.entity.position) < 4.0;
          });
          if (nearestEntity) {
            await bot.useOn(nearestEntity);
          }
        }
      } catch (err) {
        // Interaction may fail
      }
    })();
  },

  get_inventory(bot, intensity, duration, message) {
    // Returns inventory as JSON via events (consumed by controller)
    const items = bot.inventory.items().map(i => ({
      name: i.name,
      count: i.count,
      slot: i.slot,
    }));
    // Emit custom event that bot_controller picks up
    bot.emit('inventoryReport', items);
  },
};

function executeAction(bot, actionData) {
  const { action, intensity = 1.0, duration = 1, message } = actionData;

  const handler = ACTION_HANDLERS[action];
  if (handler) {
    handler(bot, intensity, duration, message);
    return true;
  }
  return false;
}

function executeCompoundAction(bot, actionData) {
  const { commands = [] } = actionData;
  let executed = 0;

  for (const cmd of commands) {
    if (executeAction(bot, cmd)) {
      executed++;
    }
  }

  return executed;
}

function stopAllActions(bot) {
  bot.setControlState('forward', false);
  bot.setControlState('back', false);
  bot.setControlState('left', false);
  bot.setControlState('right', false);
  bot.setControlState('jump', false);
  bot.setControlState('sneak', false);
  bot.setControlState('sprint', false);
}

module.exports = {
  executeAction,
  executeCompoundAction,
  stopAllActions,
};
