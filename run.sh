#!/bin/bash
# Launch script for Minecraft CL1/Izhikevich AI system
# Starts the Node.js bot server and Python brain orchestrator

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOT_SERVER_DIR="$SCRIPT_DIR/bot_server"

# Configuration (override via environment)
MC_HOST="${MC_HOST:-127.0.0.1}"
MC_PORT="${MC_PORT:-25565}"
BOT_PORT="${BOT_PORT:-3002}"

echo "=== Minecraft CL1/Izhikevich AI System ==="
echo "Minecraft server: $MC_HOST:$MC_PORT"
echo "Bot server port:  $BOT_PORT"

# Check Node.js dependencies
if [ ! -d "$BOT_SERVER_DIR/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd "$BOT_SERVER_DIR" && npm install
    cd "$SCRIPT_DIR"
fi

# Start bot server in background
echo "Starting bot server..."
MC_HOST="$MC_HOST" MC_PORT="$MC_PORT" PORT="$BOT_PORT" \
    node "$BOT_SERVER_DIR/server.js" &
BOT_PID=$!

# Wait for bot server to be ready
echo "Waiting for bot server..."
sleep 3

# Start Python brain orchestrator
echo "Starting brain orchestrator..."
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python -m minecraft_ai.orchestrator.game_loop

# Cleanup
echo "Shutting down..."
kill $BOT_PID 2>/dev/null || true
wait $BOT_PID 2>/dev/null || true
echo "Done."
