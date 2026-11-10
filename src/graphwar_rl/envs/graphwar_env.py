"""Graphwar reinforcement-learning environment."""

from __future__ import annotations

from typing import Any

from graphwar_rl.config import RewardConfig
from graphwar_rl.equations.ast import count_nodes, render_equation, simplify_ast
from graphwar_rl.equations.beauty import extract_beauty_features, score_beauty
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.simulator.base import ScenarioConfig, Simulator
from graphwar_rl.types import (
    EquationAction,
    GameState,
    ShotResult,
    StepInfo,
    Team,
)


class GraphwarEnv:
    """Connect an agent to a deterministic headless simulator."""

    def __init__(
        self,
        simulator: Simulator,
        reward_config: RewardConfig,
        grammar: EquationGrammar,
    ) -> None:
        self.simulator = simulator
        self.reward_config = reward_config
        self.grammar = grammar
        self.state: GameState | None = None

    def reset(
        self,
        seed: int,
        scenario: ScenarioConfig | None = None,
    ) -> GameState:
        """Start a deterministic episode."""

        self.state = self.simulator.reset(seed, scenario or ScenarioConfig())
        return self.state

    def step(
        self,
        action: EquationAction,
    ) -> tuple[GameState, float, bool, StepInfo]:
        """Execute one equation and return one transition."""

        if self.state is None:
            raise RuntimeError("reset must be called before step")
        previous_state = self.state
        next_state, result = self.simulator.step(action)
        combat, beauty, multihit = self.compute_reward(
            previous_state,
            action,
            result,
        )
        equation = "<invalid>"
        equation_nodes = 0
        if result.valid:
            node = self.grammar.decode(action, previous_state.mode)
            equation = render_equation(node)
            equation_nodes = count_nodes(node)
        self.state = next_state
        info = StepInfo(
            equation=equation,
            equation_nodes=equation_nodes,
            combat_reward=combat,
            beauty_reward=beauty,
            multihit_reward=multihit,
            shot=result,
        )
        return next_state, combat + beauty + multihit, (
            result.terminated or result.truncated
        ), info

    def compute_reward(
        self,
        state: GameState,
        action: EquationAction,
        result: ShotResult,
    ) -> tuple[float, float, float]:
        """Compute combat, beauty, and multihit rewards."""

        if not result.valid:
            return self.reward_config.invalid_action, 0.0, 0.0

        combat = (
            len(result.enemy_hits) * self.reward_config.enemy_hit
            + len(result.ally_hits) * self.reward_config.ally_hit
        )
        shooter_team = next(
            soldier.team
            for soldier in state.soldiers
            if soldier.soldier_id == state.current_soldier_id
        )
        if result.terminated:
            combat += (
                self.reward_config.win
                if result.winning_team is shooter_team
                else self.reward_config.loss
            )

        beauty = 0.0
        if result.trajectory is not None:
            original = self.grammar.decode(action, state.mode)
            simplified = simplify_ast(original)
            features = extract_beauty_features(
                original,
                simplified,
                result.trajectory,
            )
            beauty = self.reward_config.beauty * score_beauty(features)

        multihit = min(
            self.reward_config.max_flex_bonus,
            max(0, len(result.enemy_hits) - 1) * self.reward_config.extra_enemy,
        )
        return combat, beauty, multihit

    def current_team(self) -> Team:
        """Return the team whose agent must act."""

        if self.state is None:
            raise RuntimeError("reset must be called before current_team")
        return next(
            soldier.team
            for soldier in self.state.soldiers
            if soldier.soldier_id == self.state.current_soldier_id
        )

    def close(self) -> None:
        """Release simulator resources."""

        self.simulator.close()
        self.state = None

    def snapshot(self) -> dict[str, Any]:
        """Serialize the live episode for training resume."""

        if self.state is None:
            raise RuntimeError("Cannot snapshot before reset")
        return self.simulator.snapshot()

    def restore(self, snapshot: dict[str, Any]) -> GameState:
        """Restore a serialized live episode."""

        self.state = self.simulator.restore(snapshot)
        return self.state
