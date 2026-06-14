#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR"

"$PYTHON_BIN" -m unittest tests/test_demo.py
"$PYTHON_BIN" train_neural_q.py
"$PYTHON_BIN" run_experiments.py
"$PYTHON_BIN" run_vision_experiments.py

echo "Reproduction complete. Outputs are in $ROOT_DIR/outputs"
