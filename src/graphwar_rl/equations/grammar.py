"""Grammar constraints for autoregressive equation generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from graphwar_rl.equations.ast import TOKEN_ARITY, EquationNode, TokenKind, tokens_to_ast
from graphwar_rl.types import EquationAction, GameMode


@dataclass(frozen=True, slots=True)
class GrammarState:
    """Unfinished operands during prefix generation."""

    values_needed: int
    token_count: int
    complete: bool


class EquationGrammar:
    """A strict prefix grammar for supported Graphwar equations."""

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    def inspect(self, prefix: Sequence[TokenKind]) -> GrammarState:
        """Calculate the current grammar state."""

        values_needed = 1
        for token_index, raw_token in enumerate(prefix):
            if values_needed == 0:
                raise ValueError(f"Trailing token at position {token_index}")
            token = TokenKind(raw_token)
            values_needed = values_needed - 1 + TOKEN_ARITY[token]
        if len(prefix) > self.max_tokens:
            raise ValueError("Equation exceeds the token limit")
        return GrammarState(
            values_needed=values_needed,
            token_count=len(prefix),
            complete=values_needed == 0,
        )

    def valid_next_tokens(
        self,
        prefix: Sequence[TokenKind],
        mode: GameMode,
    ) -> tuple[TokenKind, ...]:
        """List tokens allowed at the next decoder position."""

        if mode is not GameMode.NORMAL:
            raise ValueError(f"Unsupported Graphwar mode: {mode.name}")
        state = self.inspect(prefix)
        if state.complete:
            return ()

        candidates = (
            TokenKind.ADD,
            TokenKind.NEGATE,
            TokenKind.MULTIPLY,
            TokenKind.DIVIDE,
            TokenKind.POWER,
            TokenKind.SQRT,
            TokenKind.LOG10,
            TokenKind.ABS,
            TokenKind.SIN,
            TokenKind.COS,
            TokenKind.TAN,
            TokenKind.LN,
            TokenKind.X,
            TokenKind.CONSTANT,
        )
        valid: list[TokenKind] = []
        for token in candidates:
            remaining_values = state.values_needed - 1 + TOKEN_ARITY[token]
            minimum_total = state.token_count + 1 + remaining_values
            if minimum_total <= self.max_tokens:
                valid.append(token)
        return tuple(valid)

    def is_complete(self, prefix: Sequence[TokenKind]) -> bool:
        """Check whether an equation tree is complete."""

        return self.inspect(prefix).complete

    def validate(self, prefix: Sequence[TokenKind], mode: GameMode) -> None:
        """Reject malformed or unsupported token sequences."""

        if mode is not GameMode.NORMAL:
            raise ValueError(f"Unsupported Graphwar mode: {mode.name}")
        tokens = tuple(TokenKind(token) for token in prefix)
        if any(token in {TokenKind.Y, TokenKind.Y_PRIME} for token in tokens):
            raise ValueError("Normal mode only permits x and constants")
        state = self.inspect(tokens)
        if not state.complete:
            raise ValueError(
                f"Incomplete equation: {state.values_needed} values are still required"
            )

    def build_action(
        self,
        prefix: Sequence[TokenKind],
        constants: Sequence[float],
        angle: float | None,
        mode: GameMode,
    ) -> EquationAction:
        """Construct a validated simulator action."""

        tokens = tuple(TokenKind(token) for token in prefix)
        values = tuple(float(value) for value in constants)
        self.validate(tokens, mode)
        if mode is GameMode.NORMAL and angle is not None:
            raise ValueError("Normal-mode actions do not use an explicit angle")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Equation constants must be finite")
        tokens_to_ast(tokens, values)
        return EquationAction(
            prefix_tokens=tuple(int(token) for token in tokens),
            constants=values,
            angle=angle,
        )

    def decode(self, action: EquationAction, mode: GameMode = GameMode.NORMAL) -> EquationNode:
        """Decode a validated action into its equation tree."""

        tokens = tuple(TokenKind(token) for token in action.prefix_tokens)
        self.validate(tokens, mode)
        return tokens_to_ast(tokens, action.constants)
