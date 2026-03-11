/**
 * Multi-bot lifecycle management.
 * Creates, manages, and destroys mineflayer bot instances.
 */

const mineflayer = require('mineflayer');
const BotController = require('./bot_controller');

class BotManager {
  constructor(options = {}) {
    this.mcHost = options.mcHost || '127.0.0.1';
    this.mcPort = options.mcPort || 25565;
    this.mcVersion = options.mcVersion || '1.20.4';
    this.bots = new Map(); // botId -> BotController
  }

  spawnBot(botId, options = {}) {
    if (this.bots.has(botId)) {
      console.log(`Bot ${botId} already exists`);
      return this.bots.get(botId);
    }

    const bot = mineflayer.createBot({
      host: options.host || this.mcHost,
      port: options.port || this.mcPort,
      username: options.username || `CL1_${botId}`,
      version: options.version || this.mcVersion,
      auth: 'offline',
    });

    const controller = new BotController(bot, botId);
    this.bots.set(botId, controller);
    console.log(`Bot ${botId} spawning...`);

    return controller;
  }

  despawnBot(botId) {
    const controller = this.bots.get(botId);
    if (controller) {
      controller.disconnect();
      this.bots.delete(botId);
      console.log(`Bot ${botId} despawned`);
      return true;
    }
    return false;
  }

  getBot(botId) {
    return this.bots.get(botId) || null;
  }

  getAllBotIds() {
    return Array.from(this.bots.keys());
  }

  getAllStates() {
    const states = {};
    for (const [id, controller] of this.bots) {
      const state = controller.getState();
      if (state) {
        states[id] = state;
      }
    }
    return states;
  }

  broadcastAction(actionData) {
    let count = 0;
    for (const controller of this.bots.values()) {
      if (controller.handleAction(actionData)) {
        count++;
      }
    }
    return count;
  }

  stopAll() {
    for (const controller of this.bots.values()) {
      controller.stop();
    }
  }

  disconnectAll() {
    for (const [id, controller] of this.bots) {
      controller.disconnect();
    }
    this.bots.clear();
    console.log('All bots disconnected');
  }

  get botCount() {
    return this.bots.size;
  }
}

module.exports = BotManager;
