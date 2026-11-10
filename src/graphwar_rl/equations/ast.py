"""Abstract syntax trees for Graphwar equations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class TokenKind(IntEnum):
    """Stable token identifiers used by policies and checkpoints."""

    ADD = 0
    NEGATE = 1
    MULTIPLY = 2
    DIVIDE = 3
    POWER = 4
    SQRT = 5
    LOG10 = 6
    ABS = 7
    SIN = 8
    COS = 9
    TAN = 10
    LN = 11
    X = 12
    Y = 13
    Y_PRIME = 14
    CONSTANT = 15


TOKEN_ARITY: dict[TokenKind, int] = {
    TokenKind.ADD: 2,
    TokenKind.NEGATE: 1,
    TokenKind.MULTIPLY: 2,
    TokenKind.DIVIDE: 2,
    TokenKind.POWER: 2,
    TokenKind.SQRT: 1,
    TokenKind.LOG10: 1,
    TokenKind.ABS: 1,
    TokenKind.SIN: 1,
    TokenKind.COS: 1,
    TokenKind.TAN: 1,
    TokenKind.LN: 1,
    TokenKind.X: 0,
    TokenKind.Y: 0,
    TokenKind.Y_PRIME: 0,
    TokenKind.CONSTANT: 0,
}


@dataclass(frozen=True, slots=True)
class EquationNode:
    """One node in an equation tree."""

    token: TokenKind
    value: float | None = None
    children: tuple[EquationNode, ...] = ()


def tokens_to_ast(
    tokens: Sequence[TokenKind],
    constants: Sequence[float],
) -> EquationNode:
    """Decode prefix tokens into an equation tree."""

    token_index = 0
    constant_index = 0

    def decode_node() -> EquationNode:
        nonlocal token_index, constant_index
        if token_index >= len(tokens):
            raise ValueError("Token sequence ended before the equation was complete")
        token = TokenKind(tokens[token_index])
        token_index += 1
        arity = TOKEN_ARITY[token]
        value: float | None = None
        if token is TokenKind.CONSTANT:
            if constant_index >= len(constants):
                raise ValueError("Missing value for CONSTANT token")
            value = float(constants[constant_index])
            constant_index += 1
            if not math.isfinite(value):
                raise ValueError("Equation constants must be finite")
        children = tuple(decode_node() for _ in range(arity))
        return EquationNode(token=token, value=value, children=children)

    if not tokens:
        raise ValueError("Equation token sequence must not be empty")
    root = decode_node()
    if token_index != len(tokens):
        raise ValueError("Equation contains trailing tokens")
    if constant_index != len(constants):
        raise ValueError("Equation contains unused constants")
    return root


def ast_to_tokens(node: EquationNode) -> tuple[tuple[TokenKind, ...], tuple[float, ...]]:
    """Encode an equation tree as prefix tokens and constants."""

    expected_children = TOKEN_ARITY[node.token]
    if len(node.children) != expected_children:
        raise ValueError(
            f"{node.token.name} expects {expected_children} children, "
            f"got {len(node.children)}"
        )
    if node.token is TokenKind.CONSTANT:
        if node.value is None or not math.isfinite(node.value):
            raise ValueError("CONSTANT nodes require a finite value")
    elif node.value is not None:
        raise ValueError(f"{node.token.name} nodes cannot carry a value")

    tokens: list[TokenKind] = [node.token]
    constants: list[float] = []
    if node.token is TokenKind.CONSTANT:
        constants.append(float(node.value))
    for child in node.children:
        child_tokens, child_constants = ast_to_tokens(child)
        tokens.extend(child_tokens)
        constants.extend(child_constants)
    return tuple(tokens), tuple(constants)


def render_equation(node: EquationNode) -> str:
    """Render an equation using Graphwar-compatible syntax."""

    if node.token is TokenKind.CONSTANT:
        assert node.value is not None
        return format(node.value, ".12g")
    if node.token is TokenKind.X:
        return "x"
    if node.token is TokenKind.Y:
        return "y"
    if node.token is TokenKind.Y_PRIME:
        return "y'"

    rendered = tuple(render_equation(child) for child in node.children)
    binary_symbols = {
        TokenKind.ADD: "+",
        TokenKind.MULTIPLY: "*",
        TokenKind.DIVIDE: "/",
        TokenKind.POWER: "^",
    }
    function_names = {
        TokenKind.SQRT: "sqrt",
        TokenKind.LOG10: "log",
        TokenKind.ABS: "abs",
        TokenKind.SIN: "sin",
        TokenKind.COS: "cos",
        TokenKind.TAN: "tan",
        TokenKind.LN: "ln",
    }
    if node.token is TokenKind.NEGATE:
        return f"-({rendered[0]})"
    if node.token in binary_symbols:
        return f"({rendered[0]}{binary_symbols[node.token]}{rendered[1]})"
    if node.token in function_names:
        return f"{function_names[node.token]}({rendered[0]})"
    raise ValueError(f"Unsupported token: {node.token}")


def evaluate_ast(
    node: EquationNode,
    x_value: float,
    y_value: float = 0.0,
    y_prime_value: float = 0.0,
) -> float:
    """Evaluate an equation with Java-compatible floating-point semantics."""

    try:
        if node.token is TokenKind.CONSTANT:
            assert node.value is not None
            return node.value
        if node.token is TokenKind.X:
            return x_value
        if node.token is TokenKind.Y:
            return y_value
        if node.token is TokenKind.Y_PRIME:
            return y_prime_value

        values = tuple(
            evaluate_ast(child, x_value, y_value, y_prime_value)
            for child in node.children
        )
        if node.token is TokenKind.ADD:
            return values[0] + values[1]
        if node.token is TokenKind.NEGATE:
            return -values[0]
        if node.token is TokenKind.MULTIPLY:
            return values[0] * values[1]
        if node.token is TokenKind.DIVIDE:
            return values[0] / values[1]
        if node.token is TokenKind.POWER:
            return math.pow(values[0], values[1])
        if node.token is TokenKind.SQRT:
            return math.sqrt(values[0])
        if node.token is TokenKind.LOG10:
            return math.log10(values[0])
        if node.token is TokenKind.ABS:
            return abs(values[0])
        if node.token is TokenKind.SIN:
            return math.sin(values[0])
        if node.token is TokenKind.COS:
            return math.cos(values[0])
        if node.token is TokenKind.TAN:
            return math.tan(values[0])
        if node.token is TokenKind.LN:
            return math.log(values[0])
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan
    raise ValueError(f"Unsupported token: {node.token}")


def simplify_ast(node: EquationNode) -> EquationNode:
    """Fold finite constants and remove safe algebraic identities."""

    children = tuple(simplify_ast(child) for child in node.children)
    simplified = EquationNode(token=node.token, value=node.value, children=children)

    if children and all(child.token is TokenKind.CONSTANT for child in children):
        folded = evaluate_ast(simplified, 0.0)
        if math.isfinite(folded):
            return EquationNode(TokenKind.CONSTANT, value=folded)

    zero = lambda child: child.token is TokenKind.CONSTANT and child.value == 0.0
    one = lambda child: child.token is TokenKind.CONSTANT and child.value == 1.0
    if node.token is TokenKind.ADD:
        if zero(children[0]):
            return children[1]
        if zero(children[1]):
            return children[0]
    if node.token is TokenKind.MULTIPLY:
        if zero(children[0]) or zero(children[1]):
            return EquationNode(TokenKind.CONSTANT, value=0.0)
        if one(children[0]):
            return children[1]
        if one(children[1]):
            return children[0]
    if node.token is TokenKind.DIVIDE and one(children[1]):
        return children[0]
    if node.token is TokenKind.POWER and one(children[1]):
        return children[0]
    if node.token is TokenKind.NEGATE and children[0].token is TokenKind.NEGATE:
        return children[0].children[0]
    return simplified


def count_nodes(node: EquationNode) -> int:
    """Count nodes in an equation tree."""

    return 1 + sum(count_nodes(child) for child in node.children)


def tree_depth(node: EquationNode) -> int:
    """Measure the maximum equation-tree depth."""

    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)
