from __future__ import annotations

import torch
from torch.optim import Adam

from graphwar_rl.algorithms.buffer import RolloutBuffer, RolloutStep
from graphwar_rl.algorithms.ppo import update_ppo
from graphwar_rl.algorithms.reinforce import compute_reinforce_loss
from graphwar_rl.config import ModelConfig, PPOConfig
from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.models.encoders import StateEncoder
from graphwar_rl.models.policy import ActorCriticPolicy, EquationDecoder
from graphwar_rl.observations import states_to_batch
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.simulator.python_sim import PythonGraphwarSimulator


def make_policy() -> tuple[ActorCriticPolicy, ModelConfig]:
    model_config = ModelConfig(
        terrain_embedding_dim=16,
        entity_embedding_dim=16,
        state_embedding_dim=32,
        token_embedding_dim=16,
        hidden_dim=32,
        max_equation_tokens=6,
    )
    encoder = StateEncoder(
        terrain_dim=model_config.terrain_embedding_dim,
        entity_feature_dim=model_config.entity_feature_dim,
        entity_dim=model_config.entity_embedding_dim,
        output_dim=model_config.state_embedding_dim,
    )
    decoder = EquationDecoder(
        vocabulary_size=len(TokenKind),
        state_dim=model_config.state_embedding_dim,
        token_embedding_dim=model_config.token_embedding_dim,
        hidden_dim=model_config.hidden_dim,
    )
    return (
        ActorCriticPolicy(
            encoder,
            decoder,
            model_config.state_embedding_dim,
            model_config.constant_limit,
        ),
        model_config,
    )


def test_policy_samples_valid_action_and_replays_log_probability() -> None:
    torch.manual_seed(4)
    policy, model_config = make_policy()
    grammar = EquationGrammar(model_config.max_equation_tokens)
    simulator = PythonGraphwarSimulator(model_config.max_equation_tokens)
    state = simulator.reset(
        5,
        ScenarioConfig(obstacle_count=0, observation_width=48, observation_height=28),
    )
    observation = states_to_batch(
        [state],
        model_config.max_entities,
        "cpu",
    )
    embedding = policy.encode_state(
        observation.terrain,
        observation.entities,
        observation.entity_mask,
        observation.turn_features,
    )

    step = policy.act(embedding, grammar, observation.modes)[0]
    replayed, entropy, values = policy.evaluate_actions(
        embedding,
        step.token_ids.unsqueeze(0),
        step.grammar_masks.unsqueeze(0),
        step.constant_values.unsqueeze(0),
    )

    grammar.decode(step.action, state.mode)
    assert torch.allclose(step.log_probability, replayed[0], atol=1e-5)
    assert torch.isfinite(entropy).all()
    assert values.shape == (1,)


def test_rollout_buffer_computes_bootstrapped_returns() -> None:
    buffer = RolloutBuffer()
    common = {
        "terrain": torch.zeros(1, 4, 4),
        "entities": torch.zeros(2, 7),
        "entity_mask": torch.ones(2, dtype=torch.bool),
        "turn_features": torch.zeros(4),
        "token_ids": torch.tensor([int(TokenKind.X)]),
        "constant_values": torch.zeros(1),
        "grammar_masks": torch.ones(1, len(TokenKind), dtype=torch.bool),
        "log_probability": torch.tensor(-0.5),
    }
    buffer.add(
        RolloutStep(
            **common,
            value=torch.tensor(0.5),
            reward=1.0,
            terminated=False,
        )
    )
    buffer.add(
        RolloutStep(
            **common,
            value=torch.tensor(0.25),
            reward=2.0,
            terminated=True,
        )
    )

    buffer.compute_returns_and_advantages(
        last_value=torch.tensor(99.0),
        discount=0.9,
        gae_lambda=0.95,
    )

    assert buffer.returns is not None
    assert torch.allclose(
        buffer.returns,
        torch.tensor([2.72125, 2.0]),
        atol=1e-5,
    )
    batches = list(buffer.batches(batch_size=2, shuffle=False))
    assert batches[0].token_ids.shape == (2, 1)


def test_ppo_update_is_finite_and_changes_parameters() -> None:
    torch.manual_seed(7)
    policy, model_config = make_policy()
    grammar = EquationGrammar(model_config.max_equation_tokens)
    simulator = PythonGraphwarSimulator(model_config.max_equation_tokens, 64)
    scenario = ScenarioConfig(
        obstacle_count=0,
        max_turns=4,
        observation_width=48,
        observation_height=28,
    )
    state = simulator.reset(9, scenario)
    buffer = RolloutBuffer()

    for _ in range(6):
        observation = states_to_batch([state], model_config.max_entities, "cpu")
        with torch.no_grad():
            embedding = policy.encode_state(
                observation.terrain,
                observation.entities,
                observation.entity_mask,
                observation.turn_features,
            )
            policy_step = policy.act(embedding, grammar, observation.modes)[0]
        next_state, result = simulator.step(policy_step.action)
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
                reward=-1.0 if not result.valid else 0.2,
                terminated=result.terminated or result.truncated,
            )
        )
        state = (
            simulator.reset(state.scenario_seed + 1, scenario)
            if result.terminated or result.truncated
            else next_state
        )

    observation = states_to_batch([state], model_config.max_entities, "cpu")
    with torch.no_grad():
        embedding = policy.encode_state(
            observation.terrain,
            observation.entities,
            observation.entity_mask,
            observation.turn_features,
        )
        last_value = policy.value(embedding)[0]
    buffer.compute_returns_and_advantages(last_value, 0.99, 0.95)
    optimizer = Adam(policy.parameters(), lr=3e-4)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    stats = update_ppo(
        policy,
        optimizer,
        buffer,
        PPOConfig(update_epochs=1, batch_size=3),
    )

    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.approximate_kl,
            stats.clip_fraction,
        )
    )
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, policy.parameters())
    )


def test_reinforce_loss_is_finite() -> None:
    loss = compute_reinforce_loss(
        log_probabilities=torch.tensor([-0.3, -0.8]),
        returns=torch.tensor([1.0, -0.5]),
        entropies=torch.tensor([0.4, 0.2]),
        entropy_coefficient=0.01,
    )

    assert torch.isfinite(loss)
