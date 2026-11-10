"""Rollout storage shared by policy-gradient algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """One sampled environment transition."""

    terrain: Tensor
    entities: Tensor
    entity_mask: Tensor
    turn_features: Tensor
    token_ids: Tensor
    constant_values: Tensor
    grammar_masks: Tensor
    log_probability: Tensor
    value: Tensor
    reward: float
    terminated: bool


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One padded optimization minibatch."""

    terrain: Tensor
    entities: Tensor
    entity_mask: Tensor
    turn_features: Tensor
    token_ids: Tensor
    constant_values: Tensor
    grammar_masks: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


class RolloutBuffer:
    """Collect trajectories before an optimizer update."""

    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []
        self.returns: Tensor | None = None
        self.advantages: Tensor | None = None

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: RolloutStep) -> None:
        """Append one detached transition."""

        if self.returns is not None or self.advantages is not None:
            raise RuntimeError("clear the buffer before adding a new rollout")
        self.steps.append(
            RolloutStep(
                terrain=step.terrain.detach().cpu(),
                entities=step.entities.detach().cpu(),
                entity_mask=step.entity_mask.detach().cpu(),
                turn_features=step.turn_features.detach().cpu(),
                token_ids=step.token_ids.detach().cpu(),
                constant_values=step.constant_values.detach().cpu(),
                grammar_masks=step.grammar_masks.detach().cpu(),
                log_probability=step.log_probability.detach().cpu(),
                value=step.value.detach().cpu(),
                reward=float(step.reward),
                terminated=bool(step.terminated),
            )
        )

    def clear(self) -> None:
        """Remove all collected transitions."""

        self.steps.clear()
        self.returns = None
        self.advantages = None

    def compute_returns_and_advantages(
        self,
        last_value: Tensor,
        discount: float,
        gae_lambda: float,
    ) -> None:
        """Compute bootstrapped returns and normalized GAE advantages."""

        if not self.steps:
            raise RuntimeError("Cannot compute returns for an empty buffer")
        advantages = torch.zeros(len(self.steps), dtype=torch.float32)
        gae = 0.0
        next_value = float(last_value.detach().cpu().item())
        for step_index in range(len(self.steps) - 1, -1, -1):
            step = self.steps[step_index]
            nonterminal = 0.0 if step.terminated else 1.0
            step_value = float(step.value.item())
            delta = step.reward + discount * next_value * nonterminal - step_value
            gae = delta + discount * gae_lambda * nonterminal * gae
            advantages[step_index] = gae
            next_value = step_value
        returns = advantages + torch.tensor(
            [float(step.value.item()) for step in self.steps],
            dtype=torch.float32,
        )
        if len(self.steps) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        self.advantages = advantages
        self.returns = returns

    def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[RolloutBatch]:
        """Yield padded optimization minibatches."""

        if self.returns is None or self.advantages is None:
            raise RuntimeError("Returns and advantages have not been computed")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = (
            torch.randperm(len(self.steps))
            if shuffle
            else torch.arange(len(self.steps))
        )
        for start_index in range(0, len(self.steps), batch_size):
            batch_indices = indices[start_index : start_index + batch_size]
            selected = [self.steps[int(index)] for index in batch_indices]
            max_tokens = max(step.token_ids.shape[0] for step in selected)
            vocabulary_size = selected[0].grammar_masks.shape[1]
            token_ids = torch.full(
                (len(selected), max_tokens),
                -1,
                dtype=torch.long,
            )
            constant_values = torch.zeros(
                (len(selected), max_tokens),
                dtype=torch.float32,
            )
            grammar_masks = torch.zeros(
                (len(selected), max_tokens, vocabulary_size),
                dtype=torch.bool,
            )
            for row_index, step in enumerate(selected):
                token_count = step.token_ids.shape[0]
                token_ids[row_index, :token_count] = step.token_ids
                constant_values[row_index, :token_count] = step.constant_values
                grammar_masks[row_index, :token_count] = step.grammar_masks
            yield RolloutBatch(
                terrain=torch.stack([step.terrain for step in selected]),
                entities=torch.stack([step.entities for step in selected]),
                entity_mask=torch.stack([step.entity_mask for step in selected]),
                turn_features=torch.stack([step.turn_features for step in selected]),
                token_ids=token_ids,
                constant_values=constant_values,
                grammar_masks=grammar_masks,
                old_log_probabilities=torch.stack(
                    [step.log_probability for step in selected]
                ),
                old_values=torch.stack([step.value for step in selected]),
                returns=self.returns[batch_indices],
                advantages=self.advantages[batch_indices],
            )
