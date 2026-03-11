"""Entry point for running the game loop as a module."""

import logging
from .game_loop import GameLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    loop = GameLoop()
    loop.start()
