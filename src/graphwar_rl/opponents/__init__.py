"""Baseline and self-play opponents."""

from graphwar_rl.opponents.heuristic_agent import LinearAimAgent
from graphwar_rl.opponents.random_agent import RandomAgent

__all__ = ["LinearAimAgent", "RandomAgent"]
