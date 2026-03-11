/**
 * Express + WebSocket server for Minecraft CL1/Izhikevich AI.
 *
 * REST API:
 *   POST /action         - Send action to bot
 *   GET  /state/:botId   - Get bot state
 *   POST /bots/spawn     - Spawn a new bot
 *   POST /bots/despawn   - Remove a bot
 *   GET  /bots           - List all bots
 *
 * WebSocket:
 *   Broadcasts state updates at ~10Hz
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const BotManager = require('./bot_manager');

const PORT = parseInt(process.env.PORT || '3002', 10);
const MC_HOST = process.env.MC_HOST || '127.0.0.1';
const MC_PORT = parseInt(process.env.MC_PORT || '64418', 10);
const MC_VERSION = process.env.MC_VERSION || '1.20.4';
const STATE_BROADCAST_HZ = 10;

const app = express();
app.use(express.json());

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const manager = new BotManager({
  mcHost: MC_HOST,
  mcPort: MC_PORT,
  mcVersion: MC_VERSION,
});

// --- REST API ---

app.post('/action', (req, res) => {
  const { botId = 'default', ...actionData } = req.body;
  const bot = manager.getBot(botId);
  if (!bot) {
    return res.status(404).json({ error: `Bot ${botId} not found` });
  }
  const ok = bot.handleAction(actionData);
  res.json({ success: ok });
});

app.get('/state/:botId', (req, res) => {
  const bot = manager.getBot(req.params.botId);
  if (!bot) {
    return res.status(404).json({ error: 'Bot not found' });
  }
  const state = bot.getState();
  if (!state) {
    return res.status(503).json({ error: 'Bot not ready' });
  }
  res.json(state);
});

app.post('/bots/spawn', (req, res) => {
  const { botId, ...options } = req.body;
  if (!botId) {
    return res.status(400).json({ error: 'botId required' });
  }
  manager.spawnBot(botId, options);
  res.json({ success: true, botId });
});

app.post('/bots/despawn', (req, res) => {
  const { botId } = req.body;
  if (!botId) {
    return res.status(400).json({ error: 'botId required' });
  }
  const ok = manager.despawnBot(botId);
  res.json({ success: ok });
});

app.get('/bots', (req, res) => {
  res.json({ bots: manager.getAllBotIds(), count: manager.botCount });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', bots: manager.botCount });
});

// --- WebSocket ---

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      if (data.type === 'action') {
        const botId = data.botId || 'default';
        const bot = manager.getBot(botId);
        if (bot) {
          bot.handleAction(data);
        }
      }
    } catch (err) {
      console.error('WS message error:', err.message);
    }
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

// Broadcast state at ~10Hz
const broadcastInterval = setInterval(() => {
  if (wss.clients.size === 0) return;

  const states = manager.getAllStates();
  if (Object.keys(states).length === 0) return;

  const message = JSON.stringify({
    type: 'state',
    data: states,
    timestamp: Date.now(),
  });

  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  }
}, 1000 / STATE_BROADCAST_HZ);

// --- Lifecycle ---

server.listen(PORT, () => {
  console.log(`Minecraft CL1 bot server listening on port ${PORT}`);
  console.log(`Minecraft server: ${MC_HOST}:${MC_PORT} (${MC_VERSION})`);

  // Spawn default bot
  manager.spawnBot('default');
});

process.on('SIGINT', () => {
  console.log('\nShutting down...');
  clearInterval(broadcastInterval);
  manager.disconnectAll();
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  clearInterval(broadcastInterval);
  manager.disconnectAll();
  server.close(() => process.exit(0));
});
