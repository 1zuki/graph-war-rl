"""Grammar-constrained equation policy and value model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.observations import states_to_batch
from graphwar_rl.types import EquationAction, GameMode, GameState, PolicyStep


class EquationDecoder(nn.Module):
    """Generate equation tokens and continuous constants autoregressively."""

    def __init__(
        self,
        vocabulary_size: int,
        state_dim: int,
        token_embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.begin_token = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size + 1, token_embedding_dim)
        self.initial_hidden = nn.Linear(state_dim, hidden_dim)
        self.recurrent = nn.GRU(
            token_embedding_dim + state_dim,
            hidden_dim,
            batch_first=True,
        )
        self.token_head = nn.Linear(hidden_dim, vocabulary_size)
        self.constant_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_embedding: Tensor,
        prefix_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return next-token logits and constant means for every prefix."""

        if state_embedding.ndim != 2:
            raise ValueError("state_embedding must have shape (B, D)")
        if prefix_tokens.ndim != 2:
            raise ValueError("prefix_tokens must have shape (B, T)")
        if prefix_tokens.shape[0] != state_embedding.shape[0]:
            raise ValueError("Batch dimensions must match")

        batch_size = state_embedding.shape[0]
        begin = torch.full(
            (batch_size, 1),
            self.begin_token,
            dtype=torch.long,
            device=state_embedding.device,
        )
        safe_prefix = prefix_tokens.clamp(min=0, max=self.vocabulary_size - 1)
        decoder_tokens = torch.cat([begin, safe_prefix], dim=1)
        embedded = self.token_embedding(decoder_tokens)
        repeated_state = state_embedding.unsqueeze(1).expand(
            -1,
            embedded.shape[1],
            -1,
        )
        recurrent_input = torch.cat([embedded, repeated_state], dim=-1)
        hidden = torch.tanh(self.initial_hidden(state_embedding)).unsqueeze(0)
        outputs, _ = self.recurrent(recurrent_input, hidden)
        return self.token_head(outputs), self.constant_head(outputs).squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """Generate complete equations and estimate state values."""

    def __init__(
        self,
        state_encoder: nn.Module,
        equation_decoder: EquationDecoder,
        state_dim: int,
        constant_limit: float,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.equation_decoder = equation_decoder
        self.state_dim = state_dim
        self.constant_limit = constant_limit
        self.constant_standard_deviation = 1.0
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, 1),
        )

    def encode_state(
        self,
        terrain: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        turn_features: Tensor,
    ) -> Tensor:
        """Convert a batched observation into state embeddings."""

        return self.state_encoder(
            terrain,
            entities,
            entity_mask,
            turn_features,
        )

    def act(
        self,
        state_embedding: Tensor,
        grammar: EquationGrammar,
        modes: tuple[GameMode, ...],
        deterministic: bool = False,
    ) -> tuple[PolicyStep, ...]:
        """Sample complete grammar-valid equation actions."""

        if state_embedding.shape[0] != len(modes):
            raise ValueError("modes must contain one entry per state")
        steps: list[PolicyStep] = []
        vocabulary_size = self.equation_decoder.vocabulary_size
        for batch_index, mode in enumerate(modes):
            tokens: list[TokenKind] = []
            constant_values: list[float] = []
            masks: list[Tensor] = []
            total_log_probability = torch.zeros(
                (),
                dtype=state_embedding.dtype,
                device=state_embedding.device,
            )
            total_entropy = torch.zeros_like(total_log_probability)

            while not grammar.is_complete(tokens):
                prefix = torch.tensor(
                    [[int(token) for token in tokens]],
                    dtype=torch.long,
                    device=state_embedding.device,
                )
                logits, constant_means = self.equation_decoder(
                    state_embedding[batch_index : batch_index + 1],
                    prefix,
                )
                next_logits = logits[0, -1]
                valid_tokens = grammar.valid_next_tokens(tokens, mode)
                grammar_mask = torch.zeros(
                    vocabulary_size,
                    dtype=torch.bool,
                    device=state_embedding.device,
                )
                grammar_mask[
                    torch.tensor(
                        [int(token) for token in valid_tokens],
                        dtype=torch.long,
                        device=state_embedding.device,
                    )
                ] = True
                distribution = Categorical(
                    logits=next_logits.masked_fill(~grammar_mask, -torch.inf)
                )
                token_tensor = (
                    distribution.probs.argmax()
                    if deterministic
                    else distribution.sample()
                )
                token = TokenKind(int(token_tensor.item()))
                tokens.append(token)
                masks.append(grammar_mask)
                total_log_probability = (
                    total_log_probability + distribution.log_prob(token_tensor)
                )
                total_entropy = total_entropy + distribution.entropy()

                if token is TokenKind.CONSTANT:
                    mean = constant_means[0, -1]
                    base_distribution = Normal(
                        mean,
                        torch.as_tensor(
                            self.constant_standard_deviation,
                            dtype=mean.dtype,
                            device=mean.device,
                        ),
                    )
                    raw_value = mean if deterministic else base_distribution.sample()
                    squashed = torch.tanh(raw_value)
                    constant = self.constant_limit * squashed
                    correction = torch.log(
                        self.constant_limit * (1.0 - squashed.square()) + 1e-6
                    )
                    total_log_probability = total_log_probability + (
                        base_distribution.log_prob(raw_value) - correction
                    )
                    total_entropy = total_entropy + base_distribution.entropy()
                    constant_values.append(float(constant.item()))
                else:
                    constant_values.append(0.0)

            action_constants = tuple(
                constant_values[token_index]
                for token_index, token in enumerate(tokens)
                if token is TokenKind.CONSTANT
            )
            action = grammar.build_action(
                tokens,
                action_constants,
                angle=None,
                mode=mode,
            )
            steps.append(
                PolicyStep(
                    action=action,
                    token_ids=torch.tensor(
                        [int(token) for token in tokens],
                        dtype=torch.long,
                        device=state_embedding.device,
                    ),
                    constant_values=torch.tensor(
                        constant_values,
                        dtype=state_embedding.dtype,
                        device=state_embedding.device,
                    ),
                    grammar_masks=torch.stack(masks),
                    log_probability=total_log_probability,
                    entropy=total_entropy,
                    value=self.value(state_embedding[batch_index : batch_index + 1])[0],
                )
            )
        return tuple(steps)

    def evaluate_actions(
        self,
        state_embedding: Tensor,
        token_ids: Tensor,
        grammar_masks: Tensor,
        constant_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return total action log probabilities, entropy, and values."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, T)")
        if grammar_masks.shape != (
            token_ids.shape[0],
            token_ids.shape[1],
            self.equation_decoder.vocabulary_size,
        ):
            raise ValueError("grammar_masks must have shape (B, T, V)")
        if constant_values.shape != token_ids.shape:
            raise ValueError("constant_values must have shape (B, T)")

        logits, constant_means = self.equation_decoder(state_embedding, token_ids)
        logits = logits[:, : token_ids.shape[1]]
        constant_means = constant_means[:, : token_ids.shape[1]]
        valid_positions = token_ids >= 0
        safe_token_ids = token_ids.clamp(min=0)
        masked_logits = logits.masked_fill(~grammar_masks, -1e9)
        masked_logits = torch.where(
            valid_positions.unsqueeze(-1),
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        token_distribution = Categorical(logits=masked_logits)
        token_log_probabilities = token_distribution.log_prob(safe_token_ids)
        token_log_probabilities = (
            token_log_probabilities * valid_positions.to(logits.dtype)
        )
        token_entropies = (
            token_distribution.entropy() * valid_positions.to(logits.dtype)
        )

        constant_positions = valid_positions & (
            token_ids == int(TokenKind.CONSTANT)
        )
        normalized_constants = (
            constant_values / self.constant_limit
        ).clamp(-0.999999, 0.999999)
        raw_constants = torch.atanh(normalized_constants)
        base_distribution = Normal(
            constant_means,
            torch.full_like(
                constant_means,
                self.constant_standard_deviation,
            ),
        )
        correction = torch.log(
            self.constant_limit * (1.0 - normalized_constants.square()) + 1e-6
        )
        constant_log_probabilities = (
            base_distribution.log_prob(raw_constants) - correction
        ) * constant_positions.to(logits.dtype)
        constant_entropies = base_distribution.entropy() * constant_positions.to(
            logits.dtype
        )
        return (
            token_log_probabilities.sum(dim=1)
            + constant_log_probabilities.sum(dim=1),
            token_entropies.sum(dim=1) + constant_entropies.sum(dim=1),
            self.value(state_embedding),
        )

    def value(self, state_embedding: Tensor) -> Tensor:
        """Estimate expected discounted return."""

        return self.value_head(state_embedding).squeeze(-1)


class PolicyAgent:
    """Inference adapter exposing the common Agent protocol."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        grammar: EquationGrammar,
        max_entities: int,
        deterministic: bool = True,
    ) -> None:
        self.policy = policy
        self.grammar = grammar
        self.max_entities = max_entities
        self.deterministic = deterministic

    def act(self, state: GameState) -> EquationAction:
        """Choose one policy action for a game state."""

        target_device = next(self.policy.parameters()).device
        batch = states_to_batch([state], self.max_entities, target_device)
        with torch.no_grad():
            embedding = self.policy.encode_state(
                batch.terrain,
                batch.entities,
                batch.entity_mask,
                batch.turn_features,
            )
            return self.policy.act(
                embedding,
                self.grammar,
                batch.modes,
                deterministic=self.deterministic,
            )[0].action
