#!/usr/bin/env bash
# Restores the Jetson to its low-power idle state after benchmarking.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$ROOT/config.yaml"
CFG_LOCAL="$ROOT/config.local.yaml"

# Minimal YAML reader — expects simple `key: value` lines; strips inline
# `# comments` (the committed template uses them).
_yaml_get_from() { awk -v k="$2" '
  /^[[:space:]]*[a-z_]+:/ {
    val=$0; sub(/^[^:]+:[[:space:]]*/,"",val)
    sub(/[[:space:]]*#.*$/,"",val)
    gsub(/^[[:space:]]+|[[:space:]]+$/,"",val)
    key=$1; sub(/:/,"",key)
    if (key==k) { print val; exit }
  }' "$1"; }

# config.local.yaml (real endpoint, gitignored) overrides config.yaml.
yaml_get() {
  local v=""
  if [[ -f "$CFG_LOCAL" ]]; then v="$(_yaml_get_from "$CFG_LOCAL" "$1")"; fi
  if [[ -z "$v" ]]; then v="$(_yaml_get_from "$CFG" "$1")"; fi
  printf '%s\n' "$v"
}

HOST="$(yaml_get host)"
PORT="$(yaml_get port)"
PORT="${PORT:-22}"   # optional in config; default to standard SSH port
USER="$(yaml_get user)"
IDLE_MODE="$(yaml_get idle_power_mode)"
IDLE_MODE="${IDLE_MODE:-1}"   # Orin Nano mode 1 = 7W
TARGET="${USER}@${HOST}"

echo "[teardown] Target: $TARGET (Port: $PORT)"

# 1. Unlock the clocks FIRST, while still in the power mode that was active
#    when setup_jetson.sh stored the snapshot — restoring a mode-0 snapshot
#    under a different mode applies mismatched frequency caps.
echo "[teardown] Restoring dynamic clock scaling..."
ssh -t -p "$PORT" "$TARGET" "sudo jetson_clocks --restore"

# 2. Then drop to the idle power mode.
echo "[teardown] Restoring power mode to -m $IDLE_MODE..."
ssh -t -p "$PORT" "$TARGET" "sudo nvpmodel -m $IDLE_MODE"

echo "[teardown] Jetson is now resting."
