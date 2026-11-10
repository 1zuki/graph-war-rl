"""Proximal Policy Optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from graphwar_rl.algorithms.buffer import RolloutBuffer
from graphwar_rl.config import PPOConfig
from graphwar_rl.models.policy import ActorCriticPolicy


@dataclass(frozen=True, slots=True)
class PPOStats:
    """One PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


def compute_ppo_losses(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    new_values: Tensor,
    returns: Tensor,
    advantages: Tensor,
    entropy: Tensor,
    config: PPOConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute clipped policy, value, and entropy terms."""

    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    ) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - new_values).square().mean()
    entropy_mean = entropy.mean()
    return policy_loss, value_loss, entropy_mean


def update_ppo(
    policy: ActorCriticPolicy,
    optimizer: Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
) -> PPOStats:
    """Optimize the actor-critic policy from one rollout."""

    target_device = next(policy.parameters()).device
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
    }
    update_count = 0
    for _ in range(config.update_epochs):
        for batch in buffer.batches(config.batch_size):
            terrain = batch.terrain.to(target_device)
            entities = batch.entities.to(target_device)
            entity_mask = batch.entity_mask.to(target_device)
            turn_features = batch.turn_features.to(target_device)
            token_ids = batch.token_ids.to(target_device)
            constant_values = batch.constant_values.to(target_device)
            grammar_masks = batch.grammar_masks.to(target_device)
            old_log_probabilities = batch.old_log_probabilities.to(target_device)
            returns = batch.returns.to(target_device)
            advantages = batch.advantages.to(target_device)

            state_embedding = policy.encode_state(
                terrain,
                entities,
                entity_mask,
                turn_features,
            )
            new_log_probabilities, entropy, new_values = policy.evaluate_actions(
                state_embedding,
                token_ids,
                grammar_masks,
                constant_values,
            )
            policy_loss, value_loss, entropy_mean = compute_ppo_losses(
                new_log_probabilities,
                old_log_probabilities,
                new_values,
                returns,
                advantages,
                entropy,
                config,
            )
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is not finite")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_gradient_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is not finite")
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probabilities - old_log_probabilities
                approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                ratio = torch.exp(log_ratio)
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_ratio
                ).float().mean()
            totals["policy_loss"] += float(policy_loss.detach().item())
            totals["value_loss"] += float(value_loss.detach().item())
            totals["entropy"] += float(entropy_mean.detach().item())
            totals["approximate_kl"] += float(approximate_kl.item())
            totals["clip_fraction"] += float(clip_fraction.item())
            update_count += 1

    if update_count == 0:
        raise RuntimeError("PPO received no optimization batches")
    return PPOStats(
        policy_loss=totals["policy_loss"] / update_count,
        value_loss=totals["value_loss"] / update_count,
        entropy=totals["entropy"] / update_count,
        approximate_kl=totals["approximate_kl"] / update_count,
        clip_fraction=totals["clip_fraction"] / update_count,
    )
