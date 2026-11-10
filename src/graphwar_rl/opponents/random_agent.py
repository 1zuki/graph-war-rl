"""Random grammar-valid baseline agent."""

from __future__ import annotations

import random

from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.types import EquationAction, GameState


class RandomAgent:
    """Sample complete grammar-valid equations."""

    def __init__(
        self,
        grammar: EquationGrammar,
        seed: int,
        constant_limit: float = 12.0,
    ) -> None:
        self.grammar = grammar
        self.random = random.Random(seed)
        self.constant_limit = constant_limit

    def act(self, state: GameState) -> EquationAction:
        """Sample one grammar-valid random equation."""

        tokens: list[TokenKind] = []
        while not self.grammar.is_complete(tokens):
            valid = self.grammar.valid_next_tokens(tokens, state.mode)
            leaves = [
                token
                for token in valid
                if token in {TokenKind.X, TokenKind.CONSTANT}
            ]
            choices = leaves if leaves and self.random.random() < 0.45 else list(valid)
            tokens.append(self.random.choice(choices))
        constant_count = tokens.count(TokenKind.CONSTANT)
        constants = tuple(
            max(
                -self.constant_limit,
                min(self.constant_limit, self.random.gauss(0.0, 3.0)),
            )
            for _ in range(constant_count)
        )
        return self.grammar.build_action(
            tokens,
            constants,
            angle=None,
            mode=state.mode,
        )
