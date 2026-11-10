"""Deterministic geometry-based baselines."""

from __future__ import annotations

from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.types import EquationAction, GameState, Team


class LinearAimAgent:
    """Aim a straight equation at the nearest visible enemy."""

    plane_width = 770.0

    def __init__(self, grammar: EquationGrammar, constant_limit: float = 12.0) -> None:
        self.grammar = grammar
        self.constant_limit = constant_limit

    def act(self, state: GameState) -> EquationAction:
        """Construct a line whose local slope points at the nearest enemy."""

        shooter = next(
            soldier
            for soldier in state.soldiers
            if soldier.soldier_id == state.current_soldier_id
        )
        enemies = [
            soldier
            for soldier in state.soldiers
            if soldier.alive and soldier.team is not shooter.team
        ]
        if not enemies:
            return self.grammar.build_action(
                [TokenKind.X],
                [],
                angle=None,
                mode=state.mode,
            )
        target = min(
            enemies,
            key=lambda soldier: (
                soldier.position.x - shooter.position.x
            ) ** 2
            + (soldier.position.y - shooter.position.y) ** 2,
        )
        if shooter.team is Team.ONE:
            local_shooter_x = shooter.position.x
            local_target_x = target.position.x
        else:
            local_shooter_x = self.plane_width - shooter.position.x
            local_target_x = self.plane_width - target.position.x
        horizontal_distance = local_target_x - local_shooter_x
        slope = 0.0 if abs(horizontal_distance) < 1e-9 else (
            -(target.position.y - shooter.position.y) / horizontal_distance
        )
        slope = max(-self.constant_limit, min(self.constant_limit, slope))
        return self.grammar.build_action(
            [TokenKind.MULTIPLY, TokenKind.CONSTANT, TokenKind.X],
            [slope],
            angle=None,
            mode=state.mode,
        )
