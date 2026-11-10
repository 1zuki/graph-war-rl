"""Opponent protocol."""

from __future__ import annotations

from typing import Protocol

from graphwar_rl.types import EquationAction, GameState


class Agent(Protocol):
    """Any policy that can select a Graphwar equation."""

    def act(self, state: GameState) -> EquationAction:
        """Choose one equation action."""

        ...
