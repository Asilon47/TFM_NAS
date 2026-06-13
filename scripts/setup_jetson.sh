#!/usr/bin/env bash
# One-time Jetson bootstrap. Runs on the LAPTOP. Uses SSH to:
#   1. verify docker + nvidia runtime on the Jetson
#   2. rsync the lut/bench/ directory over (lands at $REMOTE_DIR/bench/)
#   3. build the lut-runner:latest image on the Jetson
#   4. run a smoke test inside the image
#
# Pre-requisite: you've already run `ssh-copy-id <user>@<jetson-host>` so SSH
# works without a password. Edit config.yaml to point at your Jetson.
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
PORT="${PORT:-22}"
USER="$(yaml_get user)"
REMOTE_DIR="$(yaml_get remote_workdir)"
IMAGE="$(yaml_get docker_image)"
POWER_MODE="$(yaml_get power_mode)"
LOCK_CLOCKS="$(yaml_get lock_clocks)"

TARGET="${USER}@${HOST}"

echo "[setup_jetson] Target: $TARGET"
echo "[setup_jetson] SSH port: $PORT"
echo "[setup_jetson] Remote workdir: $REMOTE_DIR"
echo "[setup_jetson] Docker image tag: $IMAGE"

ssh -o BatchMode=yes -p "$PORT" "$TARGET" 'echo ok' >/dev/null || {
  echo "SSH to $TARGET failed. Run: ssh-copy-id $TARGET"; exit 1; }

echo "[setup_jetson] Setting power mode to -m $POWER_MODE..."
ssh -t -p "$PORT" "$TARGET" "sudo nvpmodel -m $POWER_MODE"

if [ "$LOCK_CLOCKS" = "true" ]; then
    echo "[setup_jetson] Locking clocks to maximum (preserving idle state if it exists)..."
    ssh -t -p "$PORT" "$TARGET" "sudo bash -c 'if [ ! -f /root/.jetsonclocks_conf.txt ]; then jetson_clocks --store; fi; jetson_clocks'"
fi

echo "[setup_jetson] Syncing Jetson clock to laptop time to fix TLS certificates..."
# Both sides in UTC: sending laptop-local time to `date -s` would let a
# timezone mismatch between the two machines skew the Jetson's clock.
CURRENT_TIME=$(date -u +"%Y-%m-%d %H:%M:%S")
ssh -t -p "$PORT" "$TARGET" "sudo date -u -s '$CURRENT_TIME'"

echo "[setup_jetson] Verifying docker on Jetson..."
ssh -p "$PORT" "$TARGET" 'docker --version && docker info --format "{{json .Runtimes}}" | grep -q nvidia' || {
  echo "docker + nvidia runtime not available on Jetson. JetPack 6 ships it — check user is in the docker group."
  exit 1
}

echo "[setup_jetson] Creating remote workdir..."
ssh -p "$PORT" "$TARGET" "mkdir -p $REMOTE_DIR/bench $REMOTE_DIR/job $REMOTE_DIR/results"

echo "[setup_jetson] Copying lut/bench/ to Jetson..."
rsync -az -e "ssh -p $PORT" --delete "$ROOT/lut/bench/" "$TARGET:$REMOTE_DIR/bench/"

echo "[setup_jetson] Pulling base image (this can take a while)..."
ssh -p "$PORT" "$TARGET" 'docker pull nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel'

echo "[setup_jetson] Building $IMAGE on Jetson..."
ssh -p "$PORT" "$TARGET" "cd $REMOTE_DIR/bench && docker build -f Dockerfile.runner -t $IMAGE ."
echo "[setup_jetson] Smoke test inside the runner image..."
ssh -p "$PORT" "$TARGET" "docker run --rm --runtime nvidia $IMAGE python3 -c 'import tensorrt, pycuda.autoinit; print(\"trt\", tensorrt.__version__)'"

echo "[setup_jetson] Done."
