#!/usr/bin/env bash
# Push + build + run the CP 3.3 search on an AGX Jetson over SSH — a Kaggle-quota-free
# continuation of the @640 BO campaign. Runs on the LAPTOP.
#
#   export XAVIER_HOST=user@jetson           # required (the Jetson SSH target)
#   bash jetson/deploy.sh --sync             # rsync code (build context) + data (mount)
#   bash jetson/deploy.sh --build            # SSH: auto-detect L4T, docker build natively
#   bash jetson/deploy.sh --run              # SSH: max clocks, run the container detached
#   bash jetson/deploy.sh --stop             # SSH: docker stop (progress is saved in the cache)
#   bash jetson/deploy.sh --resume           # SSH: re-run, continuing from the cache (skips calibrate)
#   bash jetson/deploy.sh --logs             # SSH: docker logs -f
#   bash jetson/deploy.sh --status           # SSH: container state + current verdict
#   bash jetson/deploy.sh --pull             # rsync results back into data/cp33_kaggle_out/
#   bash jetson/deploy.sh                     # (default) --sync then --build then --run
#
# Override the auto-detected base image with L4T_BASE=dustynv/ultralytics:<tag>.
# Tune the run with BUDGET=50 CALIBRATE=1 (env, passed into the container).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${XAVIER_HOST:?set XAVIER_HOST=user@host (the Jetson SSH target)}"
IMG="${IMAGE:-tfm-nas-cp33:latest}"
NAME="cp33"
DONOR="$ROOT/runs/pose/experiments/gate_baseline/weights/best.pt"
LOCAL_OUT="$ROOT/data/cp33_kaggle_out"

# Resolve the remote $HOME once so every path (incl. docker -v, which needs an absolute
# path) is unambiguous — no fragile tilde / nested-variable expansion across ssh + rsync.
RHOME="$(ssh "$HOST" 'echo "$HOME"')"
[ -n "$RHOME" ] || { echo "could not resolve remote HOME via 'ssh $HOST' — check SSH access"; exit 1; }
CODE="${XAVIER_CODE:-$RHOME/TFM_NAS_code}"   # remote build context (code only)
DATA="${XAVIER_DATA:-$RHOME/tfm_nas_data}"   # remote data plane (bind-mounted at /data)

do_sync() {
  [ -f "$DONOR" ] || { echo "Missing donor $DONOR (train the gate baseline first)"; exit 1; }
  # Build context = SOURCE PACKAGES ONLY. The repo root also holds multi-GB data/model
  # artifacts (dataset/, dataset.zip, runs/, 20_jan_a2rl_WITH_SYNTH/, *.onnx, .vscode, ...)
  # that must NEVER enter it — so ship an explicit allowlist of code dirs, not the whole tree
  # with excludes (too easy to miss a new big folder — that was the 4.7 GB stall). CODE is
  # wiped first so no stale junk lingers for the Dockerfile's COPY .
  echo "rsync code -> $HOST:$CODE (source dirs only)"
  ssh "$HOST" "rm -rf '$CODE'; mkdir -p '$CODE' '$DATA/out'"
  local srcs=() d f
  for d in catalog detect distill eval expand lut net2net search state supernet jetson scripts; do
    [ -d "$ROOT/$d" ] && srcs+=("$ROOT/./$d")
  done
  for f in "$ROOT"/*.py "$ROOT"/*.toml; do
    [ -e "$f" ] && srcs+=("$ROOT/./$(basename "$f")")
  done
  rsync -a --info=progress2 -R "${srcs[@]}" "$HOST:$CODE/"

  # Data plane (bind-mounted at /data). The dataset is ~1.6 GB — progress2 shows it moving.
  echo "rsync data -> $HOST:$DATA (dataset ~1.6 GB — this one is legitimately large)"
  rsync -a --info=progress2 "$ROOT/dataset/" "$HOST:$DATA/dataset/"
  rsync -a "$DONOR" "$HOST:$DATA/gate_best.pt"      # run_search.py find()s 'gate_best.pt'
  for f in lut.jsonl cp33_acc_memo.json phase3_nsga2_frontier.json; do
    if [ -f "$ROOT/data/$f" ]; then rsync -a "$ROOT/data/$f" "$HOST:$DATA/"; else echo "  (skip data/$f)"; fi
  done
  # The resume shards — what makes this a continuation, not a restart.
  if compgen -G "$LOCAL_OUT/cp33_bo_cache_r640.*.jsonl" >/dev/null; then
    rsync -a "$LOCAL_OUT"/cp33_bo_cache_r640.*.jsonl "$HOST:$DATA/out/"
    echo "  resume shards shipped: $(ls "$LOCAL_OUT"/cp33_bo_cache_r640.*.jsonl | wc -l)"
  else
    echo "  (no @640 resume shards found — first session will start fresh)"
  fi
}

do_build() {
  local L4T REL BASE
  L4T="$(ssh "$HOST" 'head -1 /etc/nv_tegra_release 2>/dev/null || true')"
  REL="$(grep -oE 'R[0-9]+' <<<"$L4T" | head -1 || true)"
  case "$REL" in
    R38|R39) BASE="ultralytics/ultralytics:latest-nvidia-arm64" ;;  # JetPack 7 (CUDA 13), torch 2.x
    R36)     BASE="dustynv/ultralytics:r36.2.0" ;;   # JetPack 6,   torch 2.3+
    R35)     BASE="dustynv/ultralytics:r35.4.1" ;;   # JetPack 5.1, torch 2.1
    *)   echo "Detected L4T='$L4T' ($REL) — no base-image mapping. Need torch>=2 for botorch."
         echo "Set L4T_BASE=<image> and re-run (JetPack 7 -> ultralytics/ultralytics:latest-nvidia-arm64)."; exit 1 ;;
  esac
  BASE="${L4T_BASE:-$BASE}"
  echo "L4T=$REL -> base image $BASE"
  ssh "$HOST" "cd '$CODE' && docker build --build-arg L4T_BASE='$BASE' -f jetson/Dockerfile -t '$IMG' ."
}

do_run() {
  ssh "$HOST" "sudo nvpmodel -m 0 && sudo jetson_clocks" 2>/dev/null \
    || echo "(could not set MAXN clocks via sudo — set them manually for full throughput)"
  ssh "$HOST" "docker rm -f '$NAME' 2>/dev/null || true; \
    docker run -d --name '$NAME' --runtime nvidia --ipc=host -v '$DATA':/data \
      -e BUDGET='${BUDGET:-50}' -e CALIBRATE='${CALIBRATE:-1}' '$IMG'"
  echo "started '$NAME' detached. Follow: bash jetson/deploy.sh --logs"
}

case "${1:-all}" in
  --sync|sync)     do_sync ;;
  --build|build)   do_build ;;
  --run|run)       do_run ;;
  --stop|stop)     ssh "$HOST" "docker stop '$NAME'"
                   echo "stopped '$NAME'. Progress is safe in $DATA/out (>=1 in-flight eval is just"
                   echo "recomputed on resume). Resume: bash jetson/deploy.sh --resume" ;;
  --resume|resume) : "${CALIBRATE:=0}"; do_run ;;   # same as --run, but skip the redundant calibrate
  --logs|logs)     ssh -t "$HOST" "docker logs -f '$NAME'" ;;
  --status|status) ssh "$HOST" "docker ps -a --filter name='$NAME'; echo '--- verdict ---'; \
                     cat '$DATA/out/cp33_bo.json' 2>/dev/null || echo '(no cp33_bo.json yet)'" ;;
  --pull|pull)     mkdir -p "$LOCAL_OUT"; rsync -a "$HOST:$DATA/out/" "$LOCAL_OUT/"; \
                   echo "results -> $LOCAL_OUT" ;;
  all)             do_sync; do_build; do_run ;;
  *) echo "usage: XAVIER_HOST=user@host bash jetson/deploy.sh [--sync|--build|--run|--stop|--resume|--logs|--status|--pull]"; exit 1 ;;
esac
