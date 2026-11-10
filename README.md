# Graphwar RL

Graphwar RL is a deterministic, headless normal-mode Graphwar environment with
grammar-constrained equation policies, non-RL baselines, PPO/REINFORCE
implementations, resumable checkpoints, and held-out side-swapped evaluation.

The project is deliberately scoped to Graphwar's normal `y = f(x)` mode.
First- and second-order ODE modes fail closed and are not presented as
implemented.

## What is included

- A deterministic Python simulator that follows the upstream normal-mode
  coordinate transforms, firing-angle initialization, adaptive step halving,
  hit ordering, obstacle collision, and terrain explosions.
- A long-lived Java trajectory/collision oracle using the same constants and a
  parity-tested line protocol.
- Prefix AST encoding, strict grammar masks, finite constant validation,
  simplification, canonical rendering, and bounded equation-style rewards.
- Random-valid and geometry-based linear aiming baselines.
- CNN/Deep-Sets state encoding and an autoregressive GRU equation decoder with
  grammar-masked token sampling and continuous constants.
- PPO and REINFORCE losses, generalized advantage estimation, curriculum
  scenarios, atomic checkpoints, exact RNG/live-episode resume, and JSONL
  metrics.
- Held-out map seeds, both side assignments, fixed baseline opponents, and
  machine-readable evaluation artifacts.

## Local setup

This repository uses its own virtual environment:

```bash
cd /home/izu/Projects/graphwar-rl
/home/izu/.local/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Build the optional Java backend:

```bash
./java-sim/build.sh
```

The default configuration uses the Python backend. Set
`simulator.backend = "java"` in a copied TOML config to route trajectories
through the Java oracle.

## Commands

Run the complete test suite:

```bash
.venv/bin/python -m pytest
```

Measure the two non-RL baselines:

```bash
.venv/bin/graphwar-baselines --config configs/default.toml
```

Run a short end-to-end training smoke:

```bash
.venv/bin/graphwar-train --config configs/default.toml --total-steps 64
```

Run the configured training experiment:

```bash
.venv/bin/graphwar-train --config configs/default.toml
```

Resume exactly from the latest checkpoint:

```bash
.venv/bin/graphwar-train \
  --config configs/default.toml \
  --resume runs/default/checkpoints/latest.pt
```

Evaluate a checkpoint on held-out seeds and both sides:

```bash
.venv/bin/graphwar-evaluate \
  runs/default/checkpoints/latest.pt \
  --config configs/default.toml
```

## Reproducibility contract

- Training starts at seed `42` by default and advances deterministic scenario
  seeds as episodes complete.
- Evaluation uses the disjoint fixed seeds `10001` through `10005`.
- Checkpoints contain model and optimizer state, Python/PyTorch/CUDA RNG state,
  model configuration, curriculum position, terrain damage, alive soldiers,
  current turn, and simulator RNG state.
- Invalid or non-finite equations receive the configured penalty and cannot
  mutate simulator state.
- Java/Python parity compares identical generated scenarios and actions.
- Evaluation always includes both team assignments to expose side bias.

Outputs are written beneath `runs/` and `artifacts/`, which are ignored by Git.
Training metrics use JSON Lines; evaluation and baseline reports use JSON.

## Research claims and limitations

This repository provides a reproducible experimental system, not a pretrained
agent or a claim that PPO already outperforms the baselines. A serious result
requires running multiple training seeds, retaining all configs/checkpoints,
and reporting uncertainty against the fixed held-out protocol. Beauty reward
is an explicit engineered preference, not a validated human aesthetic metric.
The simulator covers the normal-function mechanics used here; it is not a
complete replacement for every upstream game mode, network rule, or UI
behavior.

## Upstream boundary and license

`graphwar/` is the untouched upstream Graphwar Git checkout. It remains
separate so provenance is inspectable. The upstream game and this derivative
normal-mode mechanics work are distributed under GNU GPL v3 or later; see
`LICENSE` and `graphwar/COPYING`.
