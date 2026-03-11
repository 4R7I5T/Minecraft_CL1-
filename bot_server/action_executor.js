/**
 * Action type -> mineflayer API calls.
 * Translates brain action commands into bot movement/combat actions.
 */

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
    const nearestEntity = bot.nearestEntity((e) => {
      if (!e.position) return false;
      const dist = e.position.distanceTo(bot.entity.position);
      return dist < 4.0;
    });
    if (nearestEntity) {
      bot.attack(nearestEntity);
    } else {
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
