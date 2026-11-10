"""Held-out evaluation for combat, style, and balance."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

import torch

from graphwar_rl.config import ModelConfig, load_config
from graphwar_rl.envs.graphwar_env import GraphwarEnv
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.models.policy import PolicyAgent
from graphwar_rl.opponents.base import Agent
from graphwar_rl.opponents.heuristic_agent import LinearAimAgent
from graphwar_rl.opponents.random_agent import RandomAgent
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.training.train import (
    build_environment,
    build_policy_from_model_config,
    load_checkpoint,
)
from graphwar_rl.types import Team


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Externally meaningful episode outcomes."""

    seed: int
    agent_team: Team
    won: bool
    truncated: bool
    turns: int
    agent_actions: int
    enemy_hits: int
    ally_hits: int
    invalid_actions: int
    mean_beauty: float
    mean_equation_nodes: float
    max_enemies_hit_in_one_shot: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate results across seeds, opponents, and side swaps."""

    episode_count: int
    win_rate: float
    win_rate_standard_error: float
    truncation_rate: float
    mean_turns: float
    mean_enemy_hits: float
    mean_ally_hits: float
    invalid_action_rate: float
    mean_beauty: float
    mean_equation_nodes: float
    max_enemies_hit_in_one_shot: int


def run_episode(
    env: GraphwarEnv,
    agent: Agent,
    opponent: Agent,
    seed: int,
    swap_sides: bool,
    scenario: ScenarioConfig | None = None,
) -> EpisodeMetrics:
    """Evaluate one deterministic matchup."""

    episode_scenario = scenario or ScenarioConfig()
    state = env.reset(seed, episode_scenario)
    agent_team = Team.TWO if swap_sides else Team.ONE
    enemy_hits = 0
    ally_hits = 0
    invalid_actions = 0
    beauties: list[float] = []
    equation_nodes: list[int] = []
    max_multihit = 0
    agent_actions = 0
    attempts = 0
    maximum_attempts = episode_scenario.max_turns * 4
    safety_truncated = False

    while not state.terminated and not state.truncated:
        acting_for_agent = env.current_team() is agent_team
        acting_agent = agent if acting_for_agent else opponent
        action = acting_agent.act(state)
        state, _, _, info = env.step(action)
        attempts += 1
        if acting_for_agent:
            agent_actions += 1
            enemy_hits += len(info.shot.enemy_hits)
            ally_hits += len(info.shot.ally_hits)
            invalid_actions += int(not info.shot.valid)
            beauties.append(info.beauty_reward)
            equation_nodes.append(info.equation_nodes)
            max_multihit = max(max_multihit, len(info.shot.enemy_hits))
        if (
            attempts >= maximum_attempts
            and not state.terminated
            and not state.truncated
        ):
            safety_truncated = True
            break

    return EpisodeMetrics(
        seed=seed,
        agent_team=agent_team,
        won=state.winning_team is agent_team,
        truncated=state.truncated or safety_truncated,
        turns=state.turn_index,
        agent_actions=agent_actions,
        enemy_hits=enemy_hits,
        ally_hits=ally_hits,
        invalid_actions=invalid_actions,
        mean_beauty=mean(beauties) if beauties else 0.0,
        mean_equation_nodes=mean(equation_nodes) if equation_nodes else 0.0,
        max_enemies_hit_in_one_shot=max_multihit,
    )


def summarize_metrics(metrics: Sequence[EpisodeMetrics]) -> EvaluationSummary:
    """Aggregate episode-level evaluation metrics."""

    if not metrics:
        raise ValueError("metrics must not be empty")
    episode_count = len(metrics)
    wins = sum(metric.won for metric in metrics)
    win_rate = wins / episode_count
    total_actions = sum(metric.agent_actions for metric in metrics)
    return EvaluationSummary(
        episode_count=episode_count,
        win_rate=win_rate,
        win_rate_standard_error=math.sqrt(
            win_rate * (1.0 - win_rate) / episode_count
        ),
        truncation_rate=sum(metric.truncated for metric in metrics) / episode_count,
        mean_turns=mean(metric.turns for metric in metrics),
        mean_enemy_hits=mean(metric.enemy_hits for metric in metrics),
        mean_ally_hits=mean(metric.ally_hits for metric in metrics),
        invalid_action_rate=(
            sum(metric.invalid_actions for metric in metrics) / total_actions
            if total_actions
            else 0.0
        ),
        mean_beauty=mean(metric.mean_beauty for metric in metrics),
        mean_equation_nodes=mean(
            metric.mean_equation_nodes for metric in metrics
        ),
        max_enemies_hit_in_one_shot=max(
            metric.max_enemies_hit_in_one_shot for metric in metrics
        ),
    )


def evaluate_agent(
    env: GraphwarEnv,
    agent: Agent,
    opponents: Sequence[Agent],
    seeds: Sequence[int],
    scenario: ScenarioConfig | None = None,
) -> tuple[EvaluationSummary, tuple[EpisodeMetrics, ...]]:
    """Evaluate against multiple opponents, seeds, and side assignments."""

    if not opponents:
        raise ValueError("opponents must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    metrics = tuple(
        run_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            seed=int(seed),
            swap_sides=swap_sides,
            scenario=scenario,
        )
        for opponent in opponents
        for seed in seeds
        for swap_sides in (False, True)
    )
    return summarize_metrics(metrics), metrics


def load_agent(checkpoint_path: Path) -> PolicyAgent:
    """Load an inference-only policy checkpoint."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    model_values = metadata.get("model_config")
    if not isinstance(model_values, dict):
        raise ValueError("Checkpoint does not contain model_config metadata")
    model_config = ModelConfig(**model_values)
    policy = build_policy_from_model_config(model_config)
    load_checkpoint(
        checkpoint_path,
        policy,
        optimizer=None,
        restore_rng=False,
    )
    policy.eval()
    grammar = EquationGrammar(model_config.max_equation_tokens)
    return PolicyAgent(policy, grammar, model_config.max_entities)


def main() -> None:
    """Launch held-out evaluation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    env = build_environment(config)
    agent = load_agent(arguments.checkpoint)
    grammar = EquationGrammar(config.model.max_equation_tokens)
    opponents: tuple[Agent, ...] = (
        RandomAgent(grammar, seed=91, constant_limit=config.model.constant_limit),
        LinearAimAgent(grammar, constant_limit=config.model.constant_limit),
    )
    scenario = ScenarioConfig(
        observation_width=config.simulator.observation_width,
        observation_height=config.simulator.observation_height,
    )
    try:
        summary, episodes = evaluate_agent(
            env,
            agent,
            opponents,
            config.evaluation.seeds,
            scenario,
        )
    finally:
        env.close()

    output_path = arguments.output or config.evaluation.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "checkpoint": str(arguments.checkpoint),
        "rules_version": "normal-v1",
        "summary": asdict(summary),
        "episodes": [asdict(episode) for episode in episodes],
    }
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=int) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
