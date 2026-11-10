"""Benchmark non-RL agents on the held-out evaluation protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from graphwar_rl.config import load_config
from graphwar_rl.equations.grammar import EquationGrammar
from graphwar_rl.evaluation.evaluate import evaluate_agent
from graphwar_rl.opponents.heuristic_agent import LinearAimAgent
from graphwar_rl.opponents.random_agent import RandomAgent
from graphwar_rl.simulator.base import ScenarioConfig
from graphwar_rl.training.train import build_environment


def main() -> None:
    """Run random and linear-aim baseline matchups."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baselines.json"),
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    grammar = EquationGrammar(config.model.max_equation_tokens)
    scenario = ScenarioConfig(
        observation_width=config.simulator.observation_width,
        observation_height=config.simulator.observation_height,
    )

    reports = {}
    for agent_name, agent in (
        (
            "random",
            RandomAgent(
                grammar,
                seed=17,
                constant_limit=config.model.constant_limit,
            ),
        ),
        (
            "linear_aim",
            LinearAimAgent(
                grammar,
                constant_limit=config.model.constant_limit,
            ),
        ),
    ):
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
        env = build_environment(config)
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
        reports[agent_name] = {
            "summary": asdict(summary),
            "episodes": [asdict(episode) for episode in episodes],
        }

    output = {
        "rules_version": "normal-v1",
        "evaluation_seeds": list(config.evaluation.seeds),
        "reports": reports,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=int) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: report["summary"]
                for name, report in reports.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
