"""Neural encoders for terrain and variable soldier sets."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TerrainEncoder(nn.Module):
    """Encode the obstacle map with a compact convolutional network."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, terrain: Tensor) -> Tensor:
        """Map terrain `(B, 1, H, W)` to `(B, D)`."""

        if terrain.ndim != 4 or terrain.shape[1] != 1:
            raise ValueError("terrain must have shape (B, 1, H, W)")
        return self.network(terrain)


class EntityEncoder(nn.Module):
    """Encode a padded soldier set with masked Deep Sets pooling."""

    def __init__(self, feature_dim: int, output_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.entity_network = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )
        self.output_network = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, entities: Tensor, entity_mask: Tensor) -> Tensor:
        """Map entities `(B, N, F)` to `(B, D)`."""

        if entities.ndim != 3 or entities.shape[-1] != self.feature_dim:
            raise ValueError(
                f"entities must have shape (B, N, {self.feature_dim})"
            )
        if entity_mask.shape != entities.shape[:2]:
            raise ValueError("entity_mask must have shape (B, N)")
        if not entity_mask.any(dim=1).all():
            raise ValueError("Every state must contain at least one entity")

        encoded = self.entity_network(entities)
        float_mask = entity_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        mean_pool = (encoded * float_mask).sum(dim=1) / float_mask.sum(dim=1)
        minimum_value = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~entity_mask.unsqueeze(-1), minimum_value).amax(
            dim=1
        )
        return self.output_network(torch.cat([mean_pool, max_pool], dim=-1))


class StateEncoder(nn.Module):
    """Combine terrain, soldier, and turn information."""

    def __init__(
        self,
        terrain_dim: int,
        entity_feature_dim: int,
        entity_dim: int,
        output_dim: int,
        turn_feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.terrain_encoder = TerrainEncoder(terrain_dim)
        self.entity_encoder = EntityEncoder(entity_feature_dim, entity_dim)
        self.output_network = nn.Sequential(
            nn.Linear(terrain_dim + entity_dim + turn_feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        terrain: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        turn_features: Tensor,
    ) -> Tensor:
        """Produce one embedding per environment state."""

        if turn_features.ndim != 2 or turn_features.shape[1] != 4:
            raise ValueError("turn_features must have shape (B, 4)")
        terrain_embedding = self.terrain_encoder(terrain)
        entity_embedding = self.entity_encoder(entities, entity_mask)
        return self.output_network(
            torch.cat(
                [terrain_embedding, entity_embedding, turn_features],
                dim=-1,
            )
        )
