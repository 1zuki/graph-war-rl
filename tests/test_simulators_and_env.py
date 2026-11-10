from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from graphwar_rl.config import RewardConfig
from graphwar_rl.envs.graphwar_env import GraphwarEnv
from graphwar_rl.equations.ast import TokenKind
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.opponents import LinearAimAgent, RandomAgent
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.simulator.java_bridge import JavaGraphwarSimulator
from graphwar_rl.simulator.python_sim import PythonGraphwarSimulator
from graphwar_rl.types import EquationAction


ROOT = Path(__file__).resolve().parents[1]
JAR_PATH = ROOT / "java-sim/build/graphwar-headless.jar"


@pytest.fixture(scope="session")
def java_jar() -> Path:
    subprocess.run(
        [str(ROOT / "java-sim/build.sh")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert JAR_PATH.is_file()
    return JAR_PATH


def make_env() -> tuple[GraphwarEnv, EquationGrammar]:
    grammar = EquationGrammar(16)
    simulator = PythonGraphwarSimulator(16, max_trajectory_points=256)
    return GraphwarEnv(simulator, RewardConfig(), grammar), grammar


def test_reset_is_deterministic() -> None:
    scenario = ScenarioConfig(obstacle_count=4)
    first = PythonGraphwarSimulator(16)
    second = PythonGraphwarSimulator(16)

    first_state = first.reset(1234, scenario)
    second_state = second.reset(1234, scenario)

    assert first_state.soldiers == second_state.soldiers
    assert torch.equal(first_state.terrain, second_state.terrain)
    assert first.snapshot()["obstacle_circles"] == second.snapshot()["obstacle_circles"]


def test_invalid_action_cannot_mutate_state() -> None:
    simulator = PythonGraphwarSimulator(16)
    state = simulator.reset(8, ScenarioConfig(obstacle_count=2))
    before = simulator.snapshot()
    invalid = EquationAction(
        prefix_tokens=(int(TokenKind.ADD), int(TokenKind.X)),
        constants=(),
    )

    next_state, result = simulator.step(invalid)
    after = simulator.snapshot()

    assert not result.valid
    assert result.error_code == "invalid_equation"
    assert next_state.turn_index == state.turn_index
    assert next_state.soldiers == state.soldiers
    assert torch.equal(before["terrain"], after["terrain"])
    assert before["random_state"] == after["random_state"]


def test_linear_aim_hits_and_reward_decomposition_sums() -> None:
    env, grammar = make_env()
    state = env.reset(
        123,
        ScenarioConfig(obstacle_count=0, max_turns=8),
    )

    action = LinearAimAgent(grammar).act(state)
    _, reward, _, info = env.step(action)

    assert info.shot.valid
    assert info.shot.enemy_hits
    assert reward == pytest.approx(
        info.combat_reward + info.beauty_reward + info.multihit_reward
    )


def test_random_agent_always_produces_syntactically_valid_actions() -> None:
    env, grammar = make_env()
    state = env.reset(20, ScenarioConfig(obstacle_count=0))
    agent = RandomAgent(grammar, seed=99)

    for _ in range(50):
        action = agent.act(state)
        grammar.decode(action, state.mode)


@pytest.mark.parametrize(
    "tokens,constants",
    [
        ((TokenKind.X,), ()),
        ((TokenKind.SIN, TokenKind.X), ()),
        ((TokenKind.MULTIPLY, TokenKind.CONSTANT, TokenKind.X), (0.35,)),
    ],
)
def test_java_python_trajectory_parity(
    java_jar: Path,
    tokens: tuple[TokenKind, ...],
    constants: tuple[float, ...],
) -> None:
    scenario = ScenarioConfig(obstacle_count=5, max_turns=8)
    python_simulator = PythonGraphwarSimulator(16, 256)
    java_simulator = JavaGraphwarSimulator(java_jar, 16, 256)
    python_simulator.reset(2026, scenario)
    java_simulator.reset(2026, scenario)
    action = EquationAction(
        prefix_tokens=tuple(int(token) for token in tokens),
        constants=constants,
    )

    python_state, python_result = python_simulator.step(action)
    java_state, java_result = java_simulator.step(action)
    java_simulator.close()

    assert python_result.enemy_hits == java_result.enemy_hits
    assert python_result.ally_hits == java_result.ally_hits
    assert python_result.closest_enemy_distance == pytest.approx(
        java_result.closest_enemy_distance
    )
    assert python_result.trajectory is not None
    assert java_result.trajectory is not None
    assert torch.allclose(
        python_result.trajectory,
        java_result.trajectory,
        atol=1e-4,
        rtol=1e-6,
    )
    assert python_state.turn_index == java_state.turn_index
    assert python_state.soldiers == java_state.soldiers


def test_java_bridge_rejects_missing_jar(tmp_path: Path) -> None:
    simulator = JavaGraphwarSimulator(tmp_path / "missing.jar", 8)

    with pytest.raises(FileNotFoundError, match="Run java-sim/build.sh"):
        simulator.start()


def test_simulator_snapshot_restores_identical_next_transition() -> None:
    scenario = ScenarioConfig(obstacle_count=2, max_turns=8)
    grammar = EquationGrammar(16)
    first = PythonGraphwarSimulator(16, 128)
    state = first.reset(55, scenario)
    first.step(LinearAimAgent(grammar).act(state))
    snapshot = first.snapshot()
    restored = PythonGraphwarSimulator(16, 128)
    restored_state = restored.restore(snapshot)
    action = LinearAimAgent(grammar).act(restored_state)

    first_state, first_result = first.step(action)
    restored_state, restored_result = restored.step(action)

    assert first_state.soldiers == restored_state.soldiers
    assert torch.equal(first_state.terrain, restored_state.terrain)
    assert first_result.enemy_hits == restored_result.enemy_hits
    assert first_result.ally_hits == restored_result.ally_hits
    assert first_result.trajectory is not None
    assert restored_result.trajectory is not None
    assert torch.equal(first_result.trajectory, restored_result.trajectory)
