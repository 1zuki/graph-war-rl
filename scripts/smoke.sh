#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

./java-sim/build.sh
.venv/bin/python -m pytest
.venv/bin/graphwar-train --config configs/default.toml --total-steps 64
