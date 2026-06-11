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

# Minimal YAML reader — expects simple `key: value` lines.
yaml_get() { awk -v k="$1" '
  /^[[:space:]]*[a-z_]+:/ {
    val=$0; sub(/^[^:]+:[[:space:]]*/,"",val)
    gsub(/^[[:space:]]+|[[:space:]]+$/,"",val)
    key=$1; sub(/:/,"",key)
    if (key==k) { print val; exit }
  }' "$CFG"; }

HOST="$(yaml_get host)"
USER="$(yaml_get user)"
REMOTE_DIR="$(yaml_get remote_workdir)"
IMAGE="$(yaml_get docker_image)"

TARGET="${USER}@${HOST}"

echo "[setup_jetson] Target: $TARGET"
echo "[setup_jetson] Remote workdir: $REMOTE_DIR"
echo "[setup_jetson] Docker image tag: $IMAGE"

ssh -o BatchMode=yes "$TARGET" 'echo ok' >/dev/null || {
  echo "SSH to $TARGET failed. Run: ssh-copy-id $TARGET"; exit 1; }

echo "[setup_jetson] Verifying docker on Jetson..."
ssh "$TARGET" 'docker --version && docker info --format "{{json .Runtimes}}" | grep -q nvidia' || {
  echo "docker + nvidia runtime not available on Jetson. JetPack 6 ships it — check user is in the docker group."
  exit 1
}

echo "[setup_jetson] Creating remote workdir..."
ssh "$TARGET" "mkdir -p $REMOTE_DIR/bench $REMOTE_DIR/job $REMOTE_DIR/results"

echo "[setup_jetson] Copying lut/bench/ to Jetson..."
rsync -az --delete "$ROOT/lut/bench/" "$TARGET:$REMOTE_DIR/bench/"

echo "[setup_jetson] Pulling base image (this can take a while)..."
ssh "$TARGET" 'docker pull nvcr.io/nvidia/l4t-tensorrt:r36.3.0-runtime'

echo "[setup_jetson] Building $IMAGE on Jetson..."
ssh "$TARGET" "cd $REMOTE_DIR/bench && docker build -f Dockerfile.runner -t $IMAGE ."

echo "[setup_jetson] Smoke test inside the runner image..."
ssh "$TARGET" "docker run --rm --runtime nvidia $IMAGE python3 -c 'import tensorrt, pycuda.autoinit; print(\"trt\", tensorrt.__version__)'"

echo "[setup_jetson] Done."
