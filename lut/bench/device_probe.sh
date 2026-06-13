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
  local bin val
  for bin in /usr/local/bin/bandwidthTest \
             /usr/local/cuda/extras/demo_suite/bandwidthTest \
             /usr/local/cuda/samples/bin/x86_64/linux/release/bandwidthTest \
             /usr/local/cuda-samples/Samples/1_Utilities/bandwidthTest/bandwidthTest; do
    if [[ -x "$bin" ]]; then
      # CSV line looks like:
      #   bandwidthTest-D2D, Bandwidth = 62.8 GB/s, Time = ..., Size = ...
      # Grab the number after "Bandwidth =" regardless of field splitting.
      val=$("$bin" --device=0 --dtod --csv 2>/dev/null \
            | awk 'match($0, /Bandwidth = [0-9.]+/) {
                     s = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", s)
                     print s; exit
                   }' || true)
      if [[ -n "$val" ]]; then
        echo "$val"
      else
        echo "WARN: $bin ran but no bandwidth parsed from its --csv output" >&2
        echo "0.0"
      fi
      return
    fi
  done
  echo "WARN: bandwidthTest not found in any known path; reporting 0.0" >&2
  echo "0.0"
}

_gpu_devfreq_mhz() {  # $1 = cur_freq | min_freq | max_freq
  local f
  for f in "/sys/devices/gpu.0/devfreq/17000000.ga10b/$1" \
           /sys/class/devfreq/*gpu*/"$1"; do
    if [[ -r "$f" ]]; then
      awk '{print int($1/1000000)}' "$f"
      return
    fi
  done
  echo 0
}

gpu_clock_mhz()     { _gpu_devfreq_mhz max_freq; }
gpu_clock_mhz_cur() { _gpu_devfreq_mhz cur_freq; }

# jetson_clocks locks the GPU by pinning the devfreq floor to the ceiling.
# min_freq == max_freq (> 0) is therefore the canonical "clocks are locked"
# test — and it does NOT survive a reboot, which is why run_sweep re-probes
# this at every sweep start.
clocks_locked() {
  local mn mx
  mn=$(_gpu_devfreq_mhz min_freq)
  mx=$(_gpu_devfreq_mhz max_freq)
  if [[ "$mn" -gt 0 && "$mn" -eq "$mx" ]]; then echo "true"; else echo "false"; fi
}

emc_clock_mhz() {
  cat /sys/kernel/debug/clk/emc/clk_rate 2>/dev/null | awk '{print int($1/1000000)}' \
    || echo 0
}

trt_version() { python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo "unknown"; }
cuda_version() { nvcc --version 2>/dev/null | awk '/release/ {gsub(/[V,]/,"",$6); print $6; exit}' \
                 || echo "unknown"; }

# Host BSP (L4T) release from /etc/nv_tegra_release — mounted read-only from the
# host by orchestrate/probe_device.py (the container's own copy is the base
# image's stamp, not the device's). Line 1 looks like:
#   # R39 (release), REVISION: 2.0, GCID: ...   ->   R39.2.0
jetpack_l4t() {
  [[ -r /etc/nv_tegra_release ]] || { echo "unknown"; return; }
  awk 'NR==1 {
         rel=""; rev=""
         if (match($0, /R[0-9]+/))           rel=substr($0, RSTART+1, RLENGTH-1)
         if (match($0, /REVISION: [0-9.]+/)) rev=substr($0, RSTART+10, RLENGTH-10)
         if (rel != "" && rev != "") print "R" rel "." rev
         else if (rel != "")         print "R" rel
         else                        print "unknown"
         exit
       }' /etc/nv_tegra_release
}

BW=$(bandwidth_gbps)
PM=$(power_mode)
GPU=$(gpu_clock_mhz)
GPUCUR=$(gpu_clock_mhz_cur)
LOCKED=$(clocks_locked)
EMC=$(emc_clock_mhz)
TRT=$(trt_version)
CUDA=$(cuda_version)
JP=$(jetpack_l4t)

python3 - "$BW" "$PM" "$GPU" "$GPUCUR" "$LOCKED" "$EMC" "$TRT" "$CUDA" "$JP" <<'PY'
import json, sys, datetime
bw, pm, gpu, gpucur, locked, emc, trt, cuda, jp = sys.argv[1:]
print(json.dumps({
    "device": "Jetson Orin Nano",
    "power_mode": pm or "unknown",
    "gpu_clock_mhz_max": int(gpu or 0),
    "gpu_clock_mhz_cur": int(gpucur or 0),
    "clocks_locked": locked == "true",
    "emc_clock_mhz": int(emc or 0),
    "peak_dram_gbps_measured": float(bw or 0),
    "trt_version": trt,
    "cuda_version": cuda,
    "jetpack": jp or None,
    "probed_at": datetime.datetime.now(datetime.timezone.utc)
                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2))
PY
