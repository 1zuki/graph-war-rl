"""Equation and trajectory style features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from graphwar_rl.equations.ast import (
    TOKEN_ARITY,
    EquationNode,
    TokenKind,
    count_nodes,
    tree_depth,
)


@dataclass(frozen=True, slots=True)
class BeautyFeatures:
    """Measurable equation-style features."""

    node_count: int
    tree_depth: int
    distinct_operator_count: int
    rounded_constant_ratio: float
    simplification_ratio: float
    trajectory_smoothness: float
    curvature_variety: float


def _walk(node: EquationNode) -> tuple[EquationNode, ...]:
    nodes = [node]
    for child in node.children:
        nodes.extend(_walk(child))
    return tuple(nodes)


def extract_beauty_features(
    original: EquationNode,
    simplified: EquationNode,
    trajectory: Tensor,
) -> BeautyFeatures:
    """Measure equation structure and rendered trajectory shape."""

    original_nodes = _walk(original)
    constants = [
        node.value
        for node in original_nodes
        if node.token is TokenKind.CONSTANT and node.value is not None
    ]
    rounded_count = sum(
        abs(value - round(value)) <= 1e-6
        for value in constants
    )
    rounded_ratio = rounded_count / len(constants) if constants else 1.0
    operator_count = len(
        {
            node.token
            for node in original_nodes
            if TOKEN_ARITY[node.token] > 0
        }
    )
    original_count = count_nodes(original)
    simplified_count = count_nodes(simplified)
    simplification_ratio = max(0.0, 1.0 - simplified_count / original_count)

    smoothness = 0.0
    curvature_variety = 0.0
    if trajectory.ndim == 2 and trajectory.shape[0] >= 3 and trajectory.shape[1] == 2:
        finite_trajectory = trajectory.to(dtype=torch.float32)
        if torch.isfinite(finite_trajectory).all():
            first_difference = finite_trajectory[1:] - finite_trajectory[:-1]
            second_difference = first_difference[1:] - first_difference[:-1]
            roughness = second_difference.norm(dim=1).mean().item()
            smoothness = math.exp(-roughness / 3.0)

            headings = torch.atan2(first_difference[:, 1], first_difference[:, 0])
            turns = torch.atan2(
                torch.sin(headings[1:] - headings[:-1]),
                torch.cos(headings[1:] - headings[:-1]),
            ).abs()
            if turns.numel() > 0:
                curvature_variety = min(1.0, turns.std(unbiased=False).item() / 0.25)

    return BeautyFeatures(
        node_count=original_count,
        tree_depth=tree_depth(original),
        distinct_operator_count=operator_count,
        rounded_constant_ratio=rounded_ratio,
        simplification_ratio=simplification_ratio,
        trajectory_smoothness=smoothness,
        curvature_variety=curvature_variety,
    )


def score_beauty(features: BeautyFeatures) -> float:
    """Convert beauty features into a bounded style score."""

    compactness = math.exp(-abs(features.node_count - 7) / 8.0)
    depth_score = math.exp(-max(0, features.tree_depth - 5) / 3.0)
    operator_variety = min(1.0, features.distinct_operator_count / 4.0)
    simplification_score = 1.0 - min(1.0, features.simplification_ratio)
    score = (
        0.20 * compactness
        + 0.10 * depth_score
        + 0.15 * operator_variety
        + 0.10 * features.rounded_constant_ratio
        + 0.10 * simplification_score
        + 0.25 * features.trajectory_smoothness
        + 0.10 * features.curvature_variety
    )
    return max(0.0, min(1.0, score))
