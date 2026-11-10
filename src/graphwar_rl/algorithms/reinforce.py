"""Basic REINFORCE implementation for comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class ReinforceStats:
    """One REINFORCE update."""

    loss: float
    mean_return: float
    entropy: float


def compute_reinforce_loss(
    log_probabilities: Tensor,
    returns: Tensor,
    entropies: Tensor,
    entropy_coefficient: float,
) -> Tensor:
    """Compute the Monte Carlo policy-gradient objective."""

    if not (
        log_probabilities.shape == returns.shape == entropies.shape
    ):
        raise ValueError("REINFORCE tensors must have matching shapes")
    normalized_returns = returns
    if returns.numel() > 1:
        normalized_returns = (returns - returns.mean()) / (
            returns.std(unbiased=False) + 1e-8
        )
    return -(
        log_probabilities * normalized_returns.detach()
    ).mean() - entropy_coefficient * entropies.mean()


def update_reinforce(
    policy: nn.Module,
    optimizer: Optimizer,
    log_probabilities: Tensor,
    returns: Tensor,
    entropies: Tensor,
    entropy_coefficient: float,
    max_gradient_norm: float,
) -> ReinforceStats:
    """Perform one REINFORCE optimizer update."""

    optimizer.zero_grad(set_to_none=True)
    loss = compute_reinforce_loss(
        log_probabilities,
        returns,
        entropies,
        entropy_coefficient,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("REINFORCE loss is not finite")
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm)
    optimizer.step()
    return ReinforceStats(
        loss=float(loss.detach().item()),
        mean_return=float(returns.mean().detach().item()),
        entropy=float(entropies.mean().detach().item()),
    )
