/**
 * Single bot actions and state management.
 * Wraps a mineflayer bot with action execution and state observation.
 */

const { executeAction, executeCompoundAction, stopAllActions } = require('./action_executor');
const { getFullState } = require('./entity_observer');

class BotController {
  constructor(bot, botId) {
    this.bot = bot;
    this.botId = botId;
    this.isReady = false;
    this.events = [];
    this._maxEvents = 50;

    this._setupListeners();
  }

  _setupListeners() {
    this.bot.on('spawn', () => {
      this.isReady = true;
      console.log(`[${this.botId}] Bot spawned`);
    });

    this.bot.on('death', () => {
      this._pushEvent({ type: 'death', tick: Date.now() });
      console.log(`[${this.botId}] Bot died`);
    });

    this.bot.on('health', () => {
      // Health change event - tracked via state
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
      return getFullState(this.bot);
    } catch (err) {
      console.error(`[${this.botId}] State error:`, err.message);
      return null;
    }
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
