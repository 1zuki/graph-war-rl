"""Conversion from immutable game states to neural-network tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, device

from graphwar_rl.types import GameMode, GameState, Team


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """A padded batch of Graphwar observations."""

    terrain: Tensor
    entities: Tensor
    entity_mask: Tensor
    turn_features: Tensor
    modes: tuple[GameMode, ...]


def states_to_batch(
    states: Sequence[GameState],
    max_entities: int,
    target_device: device | str,
) -> ObservationBatch:
    """Convert states to normalized, fixed-shape tensors."""

    if not states:
        raise ValueError("states must not be empty")
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")

    terrain = torch.stack([state.terrain for state in states]).to(
        target_device,
        dtype=torch.float32,
    )
    entities = torch.zeros(
        (len(states), max_entities, 7),
        dtype=torch.float32,
        device=target_device,
    )
    entity_mask = torch.zeros(
        (len(states), max_entities),
        dtype=torch.bool,
        device=target_device,
    )
    turn_features = torch.zeros(
        (len(states), 4),
        dtype=torch.float32,
        device=target_device,
    )

    for batch_index, state in enumerate(states):
        if len(state.soldiers) > max_entities:
            raise ValueError(
                f"State contains {len(state.soldiers)} soldiers, "
                f"but max_entities={max_entities}"
            )
        shooter = next(
            soldier
            for soldier in state.soldiers
            if soldier.soldier_id == state.current_soldier_id
        )
        for entity_index, soldier in enumerate(state.soldiers):
            entities[batch_index, entity_index] = torch.tensor(
                [
                    soldier.position.x / 385.0 - 1.0,
                    soldier.position.y / 225.0 - 1.0,
                    float(soldier.team is Team.ONE),
                    float(soldier.team is Team.TWO),
                    float(soldier.alive),
                    float(soldier.is_current_shooter),
                    float(soldier.team is shooter.team),
                ],
                dtype=torch.float32,
                device=target_device,
            )
            entity_mask[batch_index, entity_index] = True
        alive_fraction = sum(soldier.alive for soldier in state.soldiers) / len(
            state.soldiers
        )
        turn_features[batch_index] = torch.tensor(
            [
                state.turn_index / max(1, state.max_turns),
                float(shooter.team is Team.ONE),
                float(shooter.team is Team.TWO),
                alive_fraction,
            ],
            dtype=torch.float32,
            device=target_device,
        )

    return ObservationBatch(
        terrain=terrain,
        entities=entities,
        entity_mask=entity_mask,
        turn_features=turn_features,
        modes=tuple(state.mode for state in states),
    )
