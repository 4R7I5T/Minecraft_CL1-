/**
 * Single bot actions and state management.
 * Wraps a mineflayer bot with action execution and state observation.
 * Supports chat relay, inventory tracking, and pathfinder for survival mode.
 */

const { executeAction, executeCompoundAction, stopAllActions } = require('./action_executor');
const { getFullState } = require('./entity_observer');

class BotController {
  constructor(bot, botId) {
    this.bot = bot;
    this.botId = botId;
    this.isReady = false;
    this.events = [];
    this._maxEvents = 100;
    this.chatMessages = [];      // Incoming player chat messages
    this._maxChatMessages = 50;
    this.inventory = [];         // Cached inventory state
    this._chatCallbacks = [];    // External chat listeners

    this._setupListeners();
  }

  _setupListeners() {
    this.bot.on('spawn', () => {
      this.isReady = true;
      this._loadPathfinder();
      console.log(`[${this.botId}] Bot spawned`);
    });

    this.bot.on('death', () => {
      this._pushEvent({ type: 'death', tick: Date.now() });
      console.log(`[${this.botId}] Bot died`);
    });

    this.bot.on('health', () => {
      // Auto-eat when hunger is low
      if (this.bot.food <= 6) {
        executeAction(this.bot, { action: 'eat' });
      }
    });

    this.bot.on('entityHurt', (entity) => {
      if (entity === this.bot.entity) {
        this._pushEvent({
          type: 'hurt',
          health: this.bot.health,
          tick: Date.now(),
        });
      }
    });

    this.bot.on('entityDead', (entity) => {
      if (entity !== this.bot.entity) {
        this._pushEvent({
          type: 'entity_killed',
          entityType: entity.name || entity.type,
          tick: Date.now(),
        });
      }
    });

    // Chat relay — captures player messages
    // Responds to all chat (entity proximity check is optional)
    this.bot.on('chat', (username, message) => {
      if (username === this.bot.username) return; // ignore own messages
      if (!message || message.startsWith('/')) return; // ignore commands

      // Try to get distance if player entity is visible
      let dist = -1;
      const player = this.bot.players[username];
      if (player && player.entity && this.bot.entity) {
        dist = player.entity.position.distanceTo(this.bot.entity.position);
      }

      const chatMsg = {
        player: username,
        message,
        distance: dist,
        tick: Date.now(),
      };
      this.chatMessages.push(chatMsg);
      if (this.chatMessages.length > this._maxChatMessages) {
        this.chatMessages.shift();
      }
      this._pushEvent({ type: 'chat', ...chatMsg });

      console.log(`[${this.botId}] Chat from ${username} (dist=${dist.toFixed(0)}): ${message}`);

      // Notify external callbacks
      for (const cb of this._chatCallbacks) {
        try { cb(chatMsg); } catch (e) { /* ignore */ }
      }
    });

    // Inventory tracking
    this.bot.on('playerCollect', () => {
      this._updateInventory();
    });
    this.bot.on('inventoryReport', (items) => {
      this.inventory = items;
    });

    this.bot.on('error', (err) => {
      console.error(`[${this.botId}] Error:`, err.message);
    });

    this.bot.on('kicked', (reason) => {
      console.log(`[${this.botId}] Kicked:`, reason);
      this.isReady = false;
    });

    this.bot.on('end', () => {
      console.log(`[${this.botId}] Disconnected`);
      this.isReady = false;
    });

    // Respawn on death — just wait for auto-respawn, don't run commands
    this.bot.on('death', () => {
      this._pushEvent({ type: 'respawn_needed', tick: Date.now() });
    });
  }

  _loadPathfinder() {
    try {
      const { pathfinder } = require('mineflayer-pathfinder');
      if (!this.bot.pathfinder) {
        this.bot.loadPlugin(pathfinder);
      }
    } catch (err) {
      console.log(`[${this.botId}] Pathfinder not available: ${err.message}`);
    }
  }

  _updateInventory() {
    try {
      this.inventory = this.bot.inventory.items().map(i => ({
        name: i.name,
        count: i.count,
        slot: i.slot,
      }));
    } catch (e) { /* ignore */ }
  }

  onChat(callback) {
    this._chatCallbacks.push(callback);
  }

  _pushEvent(event) {
    this.events.push(event);
    if (this.events.length > this._maxEvents) {
      this.events.shift();
    }
  }

  getState() {
    if (!this.isReady) return null;
    try {
      const state = getFullState(this.bot);
      state.inventory = this.inventory;
      state.recentChat = this.chatMessages.slice(-10);
      return state;
    } catch (err) {
      console.error(`[${this.botId}] State error:`, err.message);
      return null;
    }
  }

  consumeChat() {
    const msgs = [...this.chatMessages];
    this.chatMessages = [];
    return msgs;
  }

  handleAction(actionData) {
    if (!this.isReady) return false;

    try {
      if (actionData.compound) {
        return executeCompoundAction(this.bot, actionData) > 0;
      }
      return executeAction(this.bot, actionData);
    } catch (err) {
      console.error(`[${this.botId}] Action error:`, err.message);
      return false;
    }
  }

  stop() {
    if (this.isReady) {
      stopAllActions(this.bot);
    }
  }

  consumeEvents() {
    const evts = [...this.events];
    this.events = [];
    return evts;
  }

  disconnect() {
    this.stop();
    this.bot.quit();
    this.isReady = false;
  }
}

module.exports = BotController;
