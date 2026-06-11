#!/usr/bin/env bash
# Lint + type-check + test in one shot, using the LUT venv (.venv).
#
# Usage:
#   bash scripts/check.sh                 # full suite (incl. ~25 s slow tests)
#   bash scripts/check.sh -m "not slow"   # fast lane; extra args go to pytest
#
# Tools are invoked as `python -m ...` on purpose: the venvs' bin/ entry-point
# shebangs went stale when the repo directory was moved, but the python binary
# itself resolves correctly through its symlink.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A globally exported PYTHONPATH (e.g. ROS's setup.bash) leaks into venvs and
# can crash pytest via auto-loaded third-party plugins. Run tools clean.
unset PYTHONPATH

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found — run 'bash scripts/setup_laptop.sh' first" >&2
  exit 1
fi

echo "[check] ruff"
"$PY" -m ruff check .

echo "[check] mypy"
"$PY" -m mypy

echo "[check] pytest"
"$PY" -m pytest "$@"

echo "[check] all green"
