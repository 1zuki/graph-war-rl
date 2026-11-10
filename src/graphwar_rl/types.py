"""Shared state, action, and transition types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from torch import Tensor


class GameMode(IntEnum):
    """Graphwar equation modes."""

    NORMAL = 0
    FIRST_ORDER_ODE = 1
    SECOND_ORDER_ODE = 2


class Team(IntEnum):
    """The two Graphwar teams."""

    ONE = 1
    TWO = 2


@dataclass(frozen=True, slots=True)
class Vec2:
    """A two-dimensional pixel position."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class SoldierState:
    """One soldier in an observation."""

    soldier_id: int
    player_id: int
    team: Team
    position: Vec2
    alive: bool
    is_current_shooter: bool


@dataclass(frozen=True, slots=True)
class GameState:
    """A complete immutable environment observation."""

    terrain: Tensor
    soldiers: tuple[SoldierState, ...]
    current_player_id: int
    current_soldier_id: int
    mode: GameMode
    turn_index: int
    max_turns: int
    scenario_seed: int
    terminated: bool = False
    truncated: bool = False
    winning_team: Team | None = None
    rules_version: str = "normal-v1"


@dataclass(frozen=True, slots=True)
class EquationAction:
    """One grammar-constrained equation shot."""

    prefix_tokens: tuple[int, ...]
    constants: tuple[float, ...]
    angle: float | None = None


@dataclass(frozen=True, slots=True)
class ShotResult:
    """The simulator result of one equation."""

    valid: bool
    trajectory: Tensor | None
    enemy_hits: tuple[int, ...]
    ally_hits: tuple[int, ...]
    closest_enemy_distance: float | None
    terminated: bool
    truncated: bool
    winning_team: Team | None
    terminal_reason: str | None = None
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StepInfo:
    """Decomposed rewards and simulator diagnostics."""

    equation: str
    equation_nodes: int
    combat_reward: float
    beauty_reward: float
    multihit_reward: float
    shot: ShotResult


@dataclass(frozen=True, slots=True)
class PolicyStep:
    """One sampled policy decision."""

    action: EquationAction
    token_ids: Tensor
    constant_values: Tensor
    grammar_masks: Tensor
    log_probability: Tensor
    entropy: Tensor
    value: Tensor
