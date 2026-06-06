#!/usr/bin/env bash
# End-to-end smoke test: probes device, runs 3 cfgs through the full pipeline,
# and prints the resulting lut.jsonl tail.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[sanity] probing device..."
python -m lut.orchestrate.probe_device

echo "[sanity] running a 3-row mini-sweep (conv3x3)..."
python -m lut.orchestrate.run_sweep --blocks conv3x3 --limit 3

echo
echo "[sanity] tail of data/lut.jsonl:"
tail -3 data/lut.jsonl | python -m json.tool --json-lines 2>/dev/null || tail -3 data/lut.jsonl
