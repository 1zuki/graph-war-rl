from __future__ import annotations

from pathlib import Path

import pytest
import torch

from graphwar_rl.config import load_config
from graphwar_rl.equations.ast import (
    EquationNode,
    TokenKind,
    ast_to_tokens,
    count_nodes,
    render_equation,
    simplify_ast,
    tokens_to_ast,
)
from graphwar_rl.equations.beauty import extract_beauty_features, score_beauty
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.types import GameMode


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_complete_and_consistent() -> None:
    config = load_config(ROOT / "configs/default.toml")

    assert config.model.entity_feature_dim == 7
    assert config.simulator.backend == "python"
    assert config.training.total_steps == 20_000
    assert set(config.evaluation.seeds).isdisjoint(
        range(config.training.seed, config.training.seed + 1_000)
    )


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[training]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(config_path)


def test_ast_round_trip_render_and_simplification() -> None:
    original = EquationNode(
        TokenKind.ADD,
        children=(
            EquationNode(
                TokenKind.MULTIPLY,
                children=(
                    EquationNode(TokenKind.CONSTANT, value=1.0),
                    EquationNode(TokenKind.X),
                ),
            ),
            EquationNode(TokenKind.CONSTANT, value=0.0),
        ),
    )

    tokens, constants = ast_to_tokens(original)
    decoded = tokens_to_ast(tokens, constants)
    simplified = simplify_ast(decoded)

    assert decoded == original
    assert render_equation(decoded) == "((1*x)+0)"
    assert simplified == EquationNode(TokenKind.X)
    assert count_nodes(original) == 5


def test_ast_rejects_trailing_tokens_and_constants() -> None:
    with pytest.raises(ValueError, match="trailing tokens"):
        tokens_to_ast([TokenKind.X, TokenKind.X], [])
    with pytest.raises(ValueError, match="unused constants"):
        tokens_to_ast([TokenKind.X], [1.0])


def test_grammar_masks_only_sequences_that_can_finish() -> None:
    grammar = EquationGrammar(max_tokens=3)

    assert grammar.valid_next_tokens([], GameMode.NORMAL)
    assert TokenKind.ADD in grammar.valid_next_tokens([], GameMode.NORMAL)
    assert grammar.valid_next_tokens(
        [TokenKind.ADD, TokenKind.X],
        GameMode.NORMAL,
    ) == (TokenKind.X, TokenKind.CONSTANT)
    action = grammar.build_action(
        [TokenKind.ADD, TokenKind.X, TokenKind.CONSTANT],
        [2.0],
        angle=None,
        mode=GameMode.NORMAL,
    )
    assert grammar.decode(action) == EquationNode(
        TokenKind.ADD,
        children=(
            EquationNode(TokenKind.X),
            EquationNode(TokenKind.CONSTANT, value=2.0),
        ),
    )


def test_grammar_fails_closed_for_unsupported_modes() -> None:
    grammar = EquationGrammar(max_tokens=4)

    with pytest.raises(ValueError, match="Unsupported Graphwar mode"):
        grammar.valid_next_tokens([], GameMode.FIRST_ORDER_ODE)


def test_beauty_score_is_finite_and_bounded() -> None:
    equation = EquationNode(
        TokenKind.SIN,
        children=(EquationNode(TokenKind.X),),
    )
    trajectory = torch.tensor(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 0.9], [3.0, 1.0]],
    )

    features = extract_beauty_features(equation, equation, trajectory)
    score = score_beauty(features)

    assert 0.0 <= score <= 1.0
