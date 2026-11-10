"""Simulator protocol shared by environments and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from graphwar_rl.types import EquationAction, GameMode, GameState, ShotResult, Team


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """One deterministic Graphwar scenario."""

    mode: GameMode = GameMode.NORMAL
    players_per_team: int = 1
    soldiers_per_player: int = 2
    obstacle_count: int = 3
    max_turns: int = 40
    observation_width: int = 96
    observation_height: int = 56
    starting_team: Team = Team.ONE

    def validate(self) -> None:
        """Validate supported scenario limits."""

        if self.mode is not GameMode.NORMAL:
            raise ValueError(f"Unsupported Graphwar mode: {self.mode.name}")
        if not 1 <= self.players_per_team <= 5:
            raise ValueError("players_per_team must be between 1 and 5")
        if not 1 <= self.soldiers_per_player <= 4:
            raise ValueError("soldiers_per_player must be between 1 and 4")
        if not 0 <= self.obstacle_count <= 30:
            raise ValueError("obstacle_count must be between 0 and 30")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.observation_width <= 0 or self.observation_height <= 0:
            raise ValueError("observation dimensions must be positive")


class Simulator(Protocol):
    """A deterministic Graphwar mechanics backend."""

    def reset(self, seed: int, scenario: ScenarioConfig) -> GameState:
        """Reset the simulator."""

        ...

    def step(self, action: EquationAction) -> tuple[GameState, ShotResult]:
        """Simulate one equation shot."""

        ...

    def close(self) -> None:
        """Release simulator resources."""

        ...

    def snapshot(self) -> dict[str, Any]:
        """Serialize the live episode for exact checkpoint resume."""

        ...

    def restore(self, snapshot: dict[str, Any]) -> GameState:
        """Restore a serialized live episode."""

        ...
