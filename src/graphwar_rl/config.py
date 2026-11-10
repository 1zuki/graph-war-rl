"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Combat, beauty, and multihit reward weights."""

    win: float = 10.0
    loss: float = -10.0
    enemy_hit: float = 3.0
    ally_hit: float = -4.0
    invalid_action: float = -1.0
    beauty: float = 0.8
    extra_enemy: float = 0.3
    max_flex_bonus: float = 1.5


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Neural-network dimensions and action limits."""

    terrain_embedding_dim: int = 64
    entity_feature_dim: int = 7
    entity_embedding_dim: int = 64
    state_embedding_dim: int = 128
    token_embedding_dim: int = 64
    hidden_dim: int = 128
    max_equation_tokens: int = 16
    max_entities: int = 16
    constant_limit: float = 12.0


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Deterministic simulator settings."""

    backend: str = "python"
    java_jar: Path = Path("java-sim/build/graphwar-headless.jar")
    observation_width: int = 96
    observation_height: int = 56
    max_trajectory_points: int = 512


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO optimization settings."""

    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 1.0
    update_epochs: int = 4
    batch_size: int = 64


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Experiment length, outputs, and reproducibility."""

    seed: int = 42
    total_steps: int = 20_000
    rollout_steps: int = 256
    checkpoint_interval: int = 5_000
    evaluation_interval: int = 2_500
    run_dir: Path = Path("runs/default")
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Held-out evaluation settings."""

    seeds: tuple[int, ...] = (10_001, 10_002, 10_003, 10_004, 10_005)
    output_path: Path = Path("artifacts/evaluation.json")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """All project configuration."""

    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


ConfigSection = TypeVar(
    "ConfigSection",
    RewardConfig,
    ModelConfig,
    SimulatorConfig,
    PPOConfig,
    TrainingConfig,
    EvaluationConfig,
)


def _load_section(
    section_type: type[ConfigSection],
    values: dict[str, Any],
) -> ConfigSection:
    allowed = {item.name for item in fields(section_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in [{section_type.__name__}]: {', '.join(unknown)}"
        )

    converted = dict(values)
    for item in fields(section_type):
        if item.name not in converted:
            continue
        if item.name in {"java_jar", "run_dir", "output_path"}:
            converted[item.name] = Path(converted[item.name])
        elif item.name == "resume_from":
            raw_path = converted[item.name]
            converted[item.name] = None if raw_path in {None, ""} else Path(raw_path)
        elif item.name == "seeds":
            converted[item.name] = tuple(int(seed) for seed in converted[item.name])
    return section_type(**converted)


def _validate(config: ProjectConfig) -> None:
    positive_integers = {
        "model.max_equation_tokens": config.model.max_equation_tokens,
        "model.max_entities": config.model.max_entities,
        "simulator.observation_width": config.simulator.observation_width,
        "simulator.observation_height": config.simulator.observation_height,
        "simulator.max_trajectory_points": config.simulator.max_trajectory_points,
        "ppo.update_epochs": config.ppo.update_epochs,
        "ppo.batch_size": config.ppo.batch_size,
        "training.total_steps": config.training.total_steps,
        "training.rollout_steps": config.training.rollout_steps,
        "training.checkpoint_interval": config.training.checkpoint_interval,
        "training.evaluation_interval": config.training.evaluation_interval,
    }
    invalid = [name for name, value in positive_integers.items() if value <= 0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
    if config.simulator.backend not in {"python", "java"}:
        raise ValueError("simulator.backend must be 'python' or 'java'")
    if not 0.0 <= config.ppo.clip_ratio < 1.0:
        raise ValueError("ppo.clip_ratio must be in [0, 1)")
    if not 0.0 <= config.ppo.gae_lambda <= 1.0:
        raise ValueError("ppo.gae_lambda must be in [0, 1]")
    if not 0.0 <= config.ppo.discount <= 1.0:
        raise ValueError("ppo.discount must be in [0, 1]")
    if config.model.constant_limit <= 0.0:
        raise ValueError("model.constant_limit must be positive")
    if not config.evaluation.seeds:
        raise ValueError("evaluation.seeds must not be empty")
    if len(set(config.evaluation.seeds)) != len(config.evaluation.seeds):
        raise ValueError("evaluation.seeds must be unique")


def load_config(path: Path) -> ProjectConfig:
    """Load a strict project configuration from TOML."""

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    section_types: dict[str, type[Any]] = {
        "reward": RewardConfig,
        "model": ModelConfig,
        "simulator": SimulatorConfig,
        "ppo": PPOConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
    }
    unknown_sections = sorted(set(raw) - set(section_types))
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {', '.join(unknown_sections)}")

    loaded = {
        name: _load_section(section_type, raw.get(name, {}))
        for name, section_type in section_types.items()
    }
    config = ProjectConfig(**loaded)
    _validate(config)
    return config
