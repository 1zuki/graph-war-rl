"""Experiment construction, rollout collection, and checkpointing."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import device, nn
from torch.optim import Adam, Optimizer

from graphwar_rl.algorithms.buffer import RolloutBuffer, RolloutStep
from graphwar_rl.algorithms.ppo import update_ppo
from graphwar_rl.config import ModelConfig, ProjectConfig, load_config
from graphwar_rl.envs.graphwar_env import GraphwarEnv
from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.models.encoders import StateEncoder
from graphwar_rl.models.policy import (
    ActorCriticPolicy,
    EquationDecoder,
    PolicyAgent,
)
from graphwar_rl.observations import states_to_batch
from graphwar_rl.opponents.heuristic_agent import LinearAimAgent
from graphwar_rl.opponents.random_agent import RandomAgent
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.simulator.java_bridge import JavaGraphwarSimulator
from graphwar_rl.simulator.python_sim import PythonGraphwarSimulator


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch reproducibly."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> device:
    """Select CUDA, MPS, or CPU."""

    if torch.cuda.is_available():
        return device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return device("mps")
    return device("cpu")


def build_environment(config: ProjectConfig) -> GraphwarEnv:
    """Construct the selected simulator and RL environment."""

    grammar = EquationGrammar(config.model.max_equation_tokens)
    if config.simulator.backend == "python":
        simulator = PythonGraphwarSimulator(
            max_equation_tokens=config.model.max_equation_tokens,
            max_trajectory_points=config.simulator.max_trajectory_points,
        )
    else:
        simulator = JavaGraphwarSimulator(
            jar_path=config.simulator.java_jar,
            max_equation_tokens=config.model.max_equation_tokens,
            max_trajectory_points=config.simulator.max_trajectory_points,
        )
        simulator.start()
    return GraphwarEnv(simulator, config.reward, grammar)


def build_policy_from_model_config(model_config: ModelConfig) -> ActorCriticPolicy:
    """Construct a policy from its checkpoint-stable model config."""

    state_encoder = StateEncoder(
        terrain_dim=model_config.terrain_embedding_dim,
        entity_feature_dim=model_config.entity_feature_dim,
        entity_dim=model_config.entity_embedding_dim,
        output_dim=model_config.state_embedding_dim,
    )
    equation_decoder = EquationDecoder(
        vocabulary_size=len(TokenKind),
        state_dim=model_config.state_embedding_dim,
        token_embedding_dim=model_config.token_embedding_dim,
        hidden_dim=model_config.hidden_dim,
    )
    return ActorCriticPolicy(
        state_encoder=state_encoder,
        equation_decoder=equation_decoder,
        state_dim=model_config.state_embedding_dim,
        constant_limit=model_config.constant_limit,
    )


def build_policy(config: ProjectConfig) -> ActorCriticPolicy:
    """Construct the state encoder and equation policy."""

    return build_policy_from_model_config(config.model)


def _training_scenario(config: ProjectConfig, global_step: int) -> ScenarioConfig:
    progress = global_step / max(1, config.training.total_steps)
    if progress < 0.20:
        soldiers_per_player = 1
        obstacle_count = 0
        max_turns = 20
    elif progress < 0.50:
        soldiers_per_player = 2
        obstacle_count = 0
        max_turns = 30
    elif progress < 0.75:
        soldiers_per_player = 2
        obstacle_count = 1
        max_turns = 40
    else:
        soldiers_per_player = 2
        obstacle_count = 3
        max_turns = 40
    return ScenarioConfig(
        soldiers_per_player=soldiers_per_player,
        obstacle_count=obstacle_count,
        max_turns=max_turns,
        observation_width=config.simulator.observation_width,
        observation_height=config.simulator.observation_height,
    )


def _run_periodic_evaluation(
    config: ProjectConfig,
    policy: ActorCriticPolicy,
    global_step: int,
) -> dict[str, Any]:
    from graphwar_rl.evaluation.evaluate import evaluate_agent

    evaluation_env = build_environment(config)
    grammar = EquationGrammar(config.model.max_equation_tokens)
    agent = PolicyAgent(
        policy,
        grammar,
        config.model.max_entities,
        deterministic=True,
    )
    opponents = (
        RandomAgent(
            grammar,
            seed=91,
            constant_limit=config.model.constant_limit,
        ),
        LinearAimAgent(
            grammar,
            constant_limit=config.model.constant_limit,
        ),
    )
    evaluation_scenario = ScenarioConfig(
        observation_width=config.simulator.observation_width,
        observation_height=config.simulator.observation_height,
    )
    try:
        summary, episodes = evaluate_agent(
            evaluation_env,
            agent,
            opponents,
            config.evaluation.seeds,
            evaluation_scenario,
        )
    finally:
        evaluation_env.close()
    return {
        "step": global_step,
        "rules_version": "normal-v1",
        "summary": asdict(summary),
        "episodes": [asdict(episode) for episode in episodes],
    }


def collect_rollout(
    env: GraphwarEnv,
    policy: ActorCriticPolicy,
    buffer: RolloutBuffer,
    rollout_steps: int,
    target_device: device,
    max_entities: int,
    episode_seed: int,
    scenario: ScenarioConfig,
) -> dict[str, float]:
    """Collect environment experience before an update."""

    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    state = env.state
    if state is None or state.terminated or state.truncated:
        state = env.reset(episode_seed, scenario)

    reward_total = 0.0
    episode_count = 0
    enemy_hits = 0
    invalid_actions = 0
    current_seed = episode_seed

    policy.eval()
    for _ in range(rollout_steps):
        observation = states_to_batch([state], max_entities, target_device)
        with torch.no_grad():
            state_embedding = policy.encode_state(
                observation.terrain,
                observation.entities,
                observation.entity_mask,
                observation.turn_features,
            )
            policy_step = policy.act(
                state_embedding,
                env.grammar,
                observation.modes,
            )[0]
        next_state, reward, done, info = env.step(policy_step.action)
        buffer.add(
            RolloutStep(
                terrain=observation.terrain[0],
                entities=observation.entities[0],
                entity_mask=observation.entity_mask[0],
                turn_features=observation.turn_features[0],
                token_ids=policy_step.token_ids,
                constant_values=policy_step.constant_values,
                grammar_masks=policy_step.grammar_masks,
                log_probability=policy_step.log_probability,
                value=policy_step.value,
                reward=reward,
                terminated=done,
            )
        )
        reward_total += reward
        enemy_hits += len(info.shot.enemy_hits)
        invalid_actions += int(not info.shot.valid)

        if done:
            episode_count += 1
            current_seed += 1
            state = env.reset(current_seed, scenario)
        else:
            state = next_state

    return {
        "mean_reward": reward_total / rollout_steps,
        "episodes_completed": float(episode_count),
        "enemy_hits": float(enemy_hits),
        "invalid_actions": float(invalid_actions),
        "next_episode_seed": float(current_seed),
    }


def save_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> None:
    """Atomically save complete resumable training state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "step": int(step),
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": metadata,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore policy, optional optimizer, and RNG state."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "step",
        "policy_state",
        "optimizer_state",
        "metadata",
        "python_rng_state",
        "torch_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {', '.join(missing)}")
    if payload["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format: {payload['format_version']}"
        )
    policy.load_state_dict(payload["policy_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
    return {
        "step": int(payload["step"]),
        "metadata": payload["metadata"],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def train(config: ProjectConfig) -> None:
    """Run the complete PPO optimization loop."""

    set_seed(config.training.seed)
    target_device = select_device()
    run_dir = config.training.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(_json_safe(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = build_environment(config)
    policy = build_policy(config).to(target_device)
    optimizer = Adam(policy.parameters(), lr=config.ppo.learning_rate)
    global_step = 0
    next_episode_seed = config.training.seed

    if config.training.resume_from is not None:
        restored = load_checkpoint(
            config.training.resume_from,
            policy,
            optimizer,
        )
        global_step = restored["step"]
        checkpoint_model_config = restored["metadata"].get("model_config")
        if checkpoint_model_config != _json_safe(asdict(config.model)):
            raise ValueError("Checkpoint model configuration does not match")
        next_episode_seed = int(
            restored["metadata"].get("next_episode_seed", next_episode_seed)
        )
        environment_snapshot = restored["metadata"].get("environment_snapshot")
        if environment_snapshot is None:
            raise ValueError("Checkpoint is missing the live environment snapshot")
        env.restore(environment_snapshot)

    active_scenario: ScenarioConfig | None = None
    if config.training.resume_from is not None:
        active_scenario = env.snapshot()["scenario"]
    buffer = RolloutBuffer()
    next_checkpoint = (
        (global_step // config.training.checkpoint_interval) + 1
    ) * config.training.checkpoint_interval
    metrics_path = run_dir / "metrics.jsonl"
    next_evaluation = (
        (global_step // config.training.evaluation_interval) + 1
    ) * config.training.evaluation_interval

    try:
        while global_step < config.training.total_steps:
            scenario = _training_scenario(config, global_step)
            if active_scenario != scenario:
                if active_scenario is not None:
                    next_episode_seed += 1
                env.reset(next_episode_seed, scenario)
                active_scenario = scenario
            buffer.clear()
            steps_this_rollout = min(
                config.training.rollout_steps,
                config.training.total_steps - global_step,
            )
            rollout_metrics = collect_rollout(
                env=env,
                policy=policy,
                buffer=buffer,
                rollout_steps=steps_this_rollout,
                target_device=target_device,
                max_entities=config.model.max_entities,
                episode_seed=next_episode_seed,
                scenario=scenario,
            )
            next_episode_seed = int(rollout_metrics["next_episode_seed"])

            assert env.state is not None
            observation = states_to_batch(
                [env.state],
                config.model.max_entities,
                target_device,
            )
            with torch.no_grad():
                final_embedding = policy.encode_state(
                    observation.terrain,
                    observation.entities,
                    observation.entity_mask,
                    observation.turn_features,
                )
                last_value = (
                    torch.zeros((), device=target_device)
                    if env.state.terminated or env.state.truncated
                    else policy.value(final_embedding)[0]
                )
            buffer.compute_returns_and_advantages(
                last_value=last_value,
                discount=config.ppo.discount,
                gae_lambda=config.ppo.gae_lambda,
            )
            policy.train()
            ppo_stats = update_ppo(policy, optimizer, buffer, config.ppo)
            global_step += len(buffer)

            record = {
                "step": global_step,
                "device": str(target_device),
                **rollout_metrics,
                **asdict(ppo_stats),
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

            metadata = {
                "model_config": _json_safe(asdict(config.model)),
                "next_episode_seed": next_episode_seed,
                "rules_version": "normal-v1",
                "environment_snapshot": env.snapshot(),
            }
            if global_step >= next_checkpoint:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step_{global_step}.pt",
                    policy,
                    optimizer,
                    global_step,
                    metadata,
                )
                next_checkpoint += config.training.checkpoint_interval
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                policy,
                optimizer,
                global_step,
                metadata,
            )
            if global_step >= next_evaluation:
                evaluation_record = _run_periodic_evaluation(
                    config,
                    policy,
                    global_step,
                )
                evaluation_path = (
                    run_dir / "evaluations" / f"step_{global_step}.json"
                )
                evaluation_path.parent.mkdir(parents=True, exist_ok=True)
                evaluation_path.write_text(
                    json.dumps(
                        evaluation_record,
                        indent=2,
                        sort_keys=True,
                        default=int,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                next_evaluation += config.training.evaluation_interval
    finally:
        env.close()


def main() -> None:
    """Launch training from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    training_config = config.training
    if arguments.resume is not None:
        training_config = replace(training_config, resume_from=arguments.resume)
    if arguments.total_steps is not None:
        training_config = replace(
            training_config,
            total_steps=arguments.total_steps,
        )
    train(replace(config, training=training_config))


if __name__ == "__main__":
    main()
