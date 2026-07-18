#!/usr/bin/env bash
# Beat-n program (procedure.md 2026-07-18) — weight-free pre-bench of the emitted spec graphs.
#
# Latency is weight-independent: these probe ONNX carry l2-pruned/random weights, so every
# candidate GRAPH is measured BEFORE any 100-ep recovery buys in (the HALP lesson, applied
# proactively — the pruned-currency law extrapolates above its 158–203 MB support and this
# bench is what retires that risk).
#
# Run on the LAPTOP (.venv) with the Nano up:
#   1. bash scripts/setup_jetson.sh          # mode 0 + locked clocks — NEVER bench without it
#   2. verify `docker ps` on the board is empty (contention law, models/README.md 2026-07-17)
#   3. bash scripts/bench_beatn_probes.sh
#   4. bash scripts/teardown_jetson.sh
#
# Gates per graph: fp32 < 12.74 AND fp16 < 7.75 (the deployed baseline's measured bars);
# compare against each spec's predicted_fp32_ms / fp16_estimate_ms (prune/specs/*.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROBES=(
  data/allocate_dense/s39d_act240_640.onnx
  data/allocate_dense/s39d_act252_640.onnx
  data/allocate_dense/s39d_act259_640.onnx
  data/allocate_dense/s39d_act264_640.onnx
  data/allocate_v2/v2_topdown_act307_640.onnx
  data/allocate_v2/v2_topdown_act314_640.onnx
  data/allocate_v2/v2_pan_act307_640.onnx
)

for p in "${PROBES[@]}"; do
  [ -f "$p" ] || { echo "missing probe $p — re-run the allocator that emits it"; exit 1; }
  n="$(basename "${p%.onnx}")"
  echo "=== $n fp32 ==="
  .venv/bin/python -m lut.orchestrate.bench_model --onnx "$p" --name "probe_${n}" \
      --imgsz 640 --out "data/e2e/probe_${n}.json"
  echo "=== $n fp16 (median-of-3 fresh-cache) ==="
  .venv/bin/python -m lut.orchestrate.bench_model --onnx "$p" --name "probe_${n}_fp16" \
      --precision fp16 --repeat 3 --fresh-cache --imgsz 640 \
      --out "data/e2e/probe_${n}_fp16.json"
done

.venv/bin/python -m lut.orchestrate.audit_e2e
echo
echo "[verdict] for each probe: measured fp32 < 12.74 AND fp16 < 7.75 passes the bar;"
echo "          the winning spec is the highest-act one that passes with margin."
