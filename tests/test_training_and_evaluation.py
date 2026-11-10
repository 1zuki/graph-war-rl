from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Adam

from graphwar_rl.config import (
    EvaluationConfig,
    ModelConfig,
    PPOConfig,
    ProjectConfig,
    SimulatorConfig,
    TrainingConfig,
)
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.evaluation.evaluate import evaluate_agent, load_agent, run_episode
from graphwar_rl.models.policy import PolicyAgent
from graphwar_rl.observations import states_to_batch
from graphwar_rl.opponents import LinearAimAgent
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.training.train import (
    build_environment,
    build_policy,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train,
)
from graphwar_rl.types import EquationAction, GameState


class AlwaysInvalidAgent:
    """Agent used to verify bounded evaluation of invalid policies."""

    def act(self, state: GameState) -> EquationAction:
        return EquationAction(
            prefix_tokens=(0, 12),
            constants=(),
        )


def tiny_config(run_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(
            terrain_embedding_dim=16,
            entity_embedding_dim=16,
            state_embedding_dim=32,
            token_embedding_dim=16,
            hidden_dim=32,
            max_equation_tokens=6,
        ),
        simulator=SimulatorConfig(
            observation_width=48,
            observation_height=28,
            max_trajectory_points=64,
        ),
        ppo=PPOConfig(update_epochs=1, batch_size=4),
        training=TrainingConfig(
            seed=12,
            total_steps=4,
            rollout_steps=4,
            checkpoint_interval=4,
            evaluation_interval=1_000,
            run_dir=run_dir,
        ),
        evaluation=EvaluationConfig(seeds=(50_001, 50_002)),
    )


def test_checkpoint_restores_next_action_and_transition(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "run")
    set_seed(123)
    env = build_environment(config)
    policy = build_policy(config)
    optimizer = Adam(policy.parameters(), lr=config.ppo.learning_rate)
    state = env.reset(
        44,
        ScenarioConfig(
            obstacle_count=2,
            max_turns=8,
            observation_width=48,
            observation_height=28,
        ),
    )
    checkpoint_path = tmp_path / "resume.pt"
    metadata = {
        "model_config": asdict(config.model),
        "next_episode_seed": 44,
        "rules_version": "normal-v1",
        "environment_snapshot": env.snapshot(),
    }
    save_checkpoint(checkpoint_path, policy, optimizer, 3, metadata)

    observation = states_to_batch([state], config.model.max_entities, "cpu")
    with torch.no_grad():
        embedding = policy.encode_state(
            observation.terrain,
            observation.entities,
            observation.entity_mask,
            observation.turn_features,
        )
        expected_step = policy.act(
            embedding,
            env.grammar,
            observation.modes,
        )[0]
    expected_state, expected_result = env.simulator.step(expected_step.action)

    set_seed(999)
    restored_env = build_environment(config)
    restored_policy = build_policy(config)
    restored_optimizer = Adam(
        restored_policy.parameters(),
        lr=config.ppo.learning_rate,
    )
    restored = load_checkpoint(
        checkpoint_path,
        restored_policy,
        restored_optimizer,
    )
    restored_state = restored_env.restore(
        restored["metadata"]["environment_snapshot"]
    )
    restored_observation = states_to_batch(
        [restored_state],
        config.model.max_entities,
        "cpu",
    )
    with torch.no_grad():
        restored_embedding = restored_policy.encode_state(
            restored_observation.terrain,
            restored_observation.entities,
            restored_observation.entity_mask,
            restored_observation.turn_features,
        )
        actual_step = restored_policy.act(
            restored_embedding,
            restored_env.grammar,
            restored_observation.modes,
        )[0]
    actual_state, actual_result = restored_env.simulator.step(actual_step.action)

    assert actual_step.action == expected_step.action
    assert actual_result.enemy_hits == expected_result.enemy_hits
    assert actual_result.ally_hits == expected_result.ally_hits
    assert actual_state.soldiers == expected_state.soldiers
    assert torch.equal(actual_state.terrain, expected_state.terrain)


def test_training_smoke_writes_resumable_checkpoint(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "smoke")

    train(config)

    checkpoint = config.training.run_dir / "checkpoints/latest.pt"
    assert checkpoint.is_file()
    assert (config.training.run_dir / "metrics.jsonl").is_file()
    agent = load_agent(checkpoint)
    assert isinstance(agent, PolicyAgent)


def test_held_out_evaluation_covers_both_sides(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "evaluation")
    env = build_environment(config)
    grammar = EquationGrammar(config.model.max_equation_tokens)
    agent = LinearAimAgent(grammar)
    scenario = ScenarioConfig(
        obstacle_count=1,
        max_turns=8,
        observation_width=48,
        observation_height=28,
    )

    summary, episodes = evaluate_agent(
        env,
        agent,
        opponents=(LinearAimAgent(grammar),),
        seeds=config.evaluation.seeds,
        scenario=scenario,
    )
    env.close()

    assert summary.episode_count == 4
    assert {episode.agent_team for episode in episodes} == {1, 2}
    assert {episode.seed for episode in episodes} == set(config.evaluation.seeds)


def test_evaluation_reports_safety_truncation_for_invalid_agent() -> None:
    config = tiny_config(Path("/tmp/graphwar-invalid-evaluation"))
    env = build_environment(config)
    grammar = EquationGrammar(config.model.max_equation_tokens)
    scenario = ScenarioConfig(
        obstacle_count=0,
        max_turns=2,
        observation_width=48,
        observation_height=28,
    )

    try:
        metrics = run_episode(
            env,
            AlwaysInvalidAgent(),
            LinearAimAgent(grammar),
            seed=71,
            swap_sides=False,
            scenario=scenario,
        )
    finally:
        env.close()

    assert metrics.truncated
    assert metrics.turns == 0
    assert metrics.agent_actions == scenario.max_turns * 4
    assert metrics.invalid_actions == metrics.agent_actions
