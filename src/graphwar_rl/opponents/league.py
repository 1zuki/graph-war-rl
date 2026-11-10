"""Small deterministic self-play opponent league."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random


@dataclass(frozen=True, slots=True)
class OpponentEntry:
    """One frozen self-play opponent."""

    checkpoint_path: Path
    rating: float
    games_played: int


class SelfPlayLeague:
    """Sample and update historical policy checkpoints."""

    def __init__(self, seed: int) -> None:
        self.random = Random(seed)
        self.entries: list[OpponentEntry] = []

    def add_checkpoint(self, checkpoint_path: Path, initial_rating: float = 1000.0) -> None:
        """Register a unique frozen policy checkpoint."""

        resolved = checkpoint_path.resolve()
        if any(entry.checkpoint_path.resolve() == resolved for entry in self.entries):
            raise ValueError(f"Checkpoint is already registered: {checkpoint_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        self.entries.append(OpponentEntry(checkpoint_path, initial_rating, 0))

    def sample(self) -> OpponentEntry:
        """Choose an opponent with mild preference for less-played entries."""

        if not self.entries:
            raise RuntimeError("Cannot sample an empty opponent league")
        weights = [1.0 / math.sqrt(entry.games_played + 1.0) for entry in self.entries]
        return self.random.choices(self.entries, weights=weights, k=1)[0]

    def record_result(self, opponent: OpponentEntry, score: float) -> None:
        """Update an opponent Elo rating from the learner's score."""

        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        try:
            entry_index = self.entries.index(opponent)
        except ValueError as error:
            raise ValueError("Opponent is not registered in this league") from error
        expected_learner = 1.0 / (1.0 + 10.0 ** ((opponent.rating - 1000.0) / 400.0))
        opponent_score = 1.0 - score
        expected_opponent = 1.0 - expected_learner
        updated = replace(
            opponent,
            rating=opponent.rating + 24.0 * (opponent_score - expected_opponent),
            games_played=opponent.games_played + 1,
        )
        self.entries[entry_index] = updated
