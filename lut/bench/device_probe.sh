#!/usr/bin/env bash
# Runs on the Jetson, inside the lut-runner container. Emits device_info.json
# on stdout so the laptop can capture it with ssh + redirect.
set -euo pipefail

power_mode() {
  if command -v nvpmodel >/dev/null; then
    nvpmodel -q 2>/dev/null | awk -F':' '/NV Power Mode/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}'
  elif [[ -r /var/lib/nvpmodel/status ]]; then
    awk -F':' '{print int($2)}' /var/lib/nvpmodel/status
  else
    echo "unknown"
  fi
}

bandwidth_gbps() {
  local bin
  for bin in /usr/local/bin/bandwidthTest \
             /usr/local/cuda/extras/demo_suite/bandwidthTest \
             /usr/local/cuda/samples/bin/x86_64/linux/release/bandwidthTest \
             /usr/local/cuda-samples/Samples/1_Utilities/bandwidthTest/bandwidthTest; do
    if [[ -x "$bin" ]]; then
      # Matches lines starting with numbers, splits by comma, and strips the hidden space
      "$bin" --device=0 --dtod --csv 2>/dev/null | awk -F',' '/^[0-9]+/ {gsub(/ /, "", $2); print $2; exit}'
      return
    fi
  done
  echo "0.0"
}

gpu_clock_mhz() {
  if [[ -r /sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq ]]; then
    awk '{print int($1/1000000)}' /sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq
  else
    # Fall back: read any available gpu devfreq entry
    cat /sys/class/devfreq/*gpu*/max_freq 2>/dev/null | head -1 | awk '{print int($1/1000000)}'
  fi
}

emc_clock_mhz() {
  cat /sys/kernel/debug/clk/emc/clk_rate 2>/dev/null | awk '{print int($1/1000000)}' \
    || echo 0
}

trt_version() { python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo "unknown"; }
cuda_version() { nvcc --version 2>/dev/null | awk '/release/ {gsub(",","",$6); print $6; exit}' \
                 || echo "unknown"; }

BW=$(bandwidth_gbps)
PM=$(power_mode)
GPU=$(gpu_clock_mhz)
EMC=$(emc_clock_mhz)
TRT=$(trt_version)
CUDA=$(cuda_version)

python3 - "$BW" "$PM" "$GPU" "$EMC" "$TRT" "$CUDA" <<'PY'
import json, sys, datetime
bw, pm, gpu, emc, trt, cuda = sys.argv[1:]
print(json.dumps({
    "device": "Jetson Orin Nano",
    "power_mode": pm or "unknown",
    "gpu_clock_mhz_max": int(gpu or 0),
    "emc_clock_mhz": int(emc or 0),
    "peak_dram_gbps_measured": float(bw or 0),
    "trt_version": trt,
    "cuda_version": cuda,
    "probed_at": datetime.datetime.utcnow().isoformat() + "Z",
}, indent=2))
PY
