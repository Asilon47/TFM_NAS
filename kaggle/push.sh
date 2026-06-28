#!/usr/bin/env bash
# Push the CP 3.3 search to Kaggle via the API — no manual uploads.
# Runs on the LAPTOP (.venv). Token read from the gitignored secrets/kaggle.json.
#
#   bash kaggle/push.sh --data     # (occasional) create/version the data Dataset
#   bash kaggle/push.sh            # (default) push/run the kernel
#   bash kaggle/push.sh --status   # kernel run status
#   bash kaggle/push.sh --pull     # download the kernel output into data/
#
# One-time: save your KGAT_ token at secrets/access_token + your Kaggle username
# at secrets/kaggle_username (or a legacy secrets/kaggle.json), then
# `pip install kaggle`. See kaggle/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$ROOT/kaggle"
SECRETS_DIR="$ROOT/secrets"
BUILD="$KAGGLE_DIR/_build"
DATASET_SLUG="tfm-nas-gate-pose"
KERNEL_SLUG="tfm-nas-cp33-search"
DONOR="$ROOT/runs/pose/experiments/gate_baseline/weights/best.pt"

# --- credentials (local, gitignored) -----------------------------------------
# New-style token (KGAT_...) at secrets/access_token + username at
# secrets/kaggle_username; or a legacy secrets/kaggle.json ({username,key}).
TOKEN_FILE="$SECRETS_DIR/access_token"
LEGACY_JSON="$SECRETS_DIR/kaggle.json"
USER_FILE="$SECRETS_DIR/kaggle_username"
for f in access_token kaggle.json kaggle_username; do
  if git -C "$ROOT" ls-files --error-unmatch "secrets/$f" >/dev/null 2>&1; then
    echo "REFUSING: secrets/$f is git-tracked — untrack it before using credentials."; exit 1
  fi
done
export KAGGLE_CONFIG_DIR="$SECRETS_DIR"              # legacy kaggle.json lookup
if [ -f "$TOKEN_FILE" ]; then
  chmod 600 "$TOKEN_FILE" 2>/dev/null || true
  export KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
elif [ -f "$LEGACY_JSON" ]; then
  chmod 600 "$LEGACY_JSON" 2>/dev/null || true
else
  echo "Missing credentials — save a KGAT_ token at secrets/access_token (Kaggle > Settings > API > Create New Token). See kaggle/README.md"; exit 1
fi
command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found — pip install kaggle"; exit 1; }
# Username for the dataset/kernel slugs (the KGAT_ token doesn't embed it).
if   [ -n "${KAGGLE_USERNAME:-}" ]; then KUSER="$KAGGLE_USERNAME"
elif [ -f "$USER_FILE" ];           then KUSER="$(tr -d '[:space:]' < "$USER_FILE")"
elif [ -f "$LEGACY_JSON" ];         then KUSER="$(python3 -c "import json;print(json.load(open('$LEGACY_JSON'))['username'])")"
else echo "Missing Kaggle username — put it in secrets/kaggle_username (e.g. asilarnous) or export KAGGLE_USERNAME."; exit 1
fi

sub() { sed "s/__KAGGLE_USERNAME__/$KUSER/g" "$1"; }   # username into the metadata templates

case "${1:-kernel}" in
  --data|data)
    # Stage the data Dataset via HARDLINKS: no 1.6 GB copy, and (unlike symlinks)
    # os.walk/the Kaggle uploader follows them. dataset/ + the @224/@640 LUT + the
    # NSGA-II warm-start seeds + the frozen gate-head donor.
    [ -f "$DONOR" ] || { echo "Missing donor $DONOR (train the gate baseline first)"; exit 1; }
    D="$BUILD/data"; rm -rf "$D"; mkdir -p "$D"
    cp -al "$ROOT/dataset" "$D/dataset"
    cp "$DONOR" "$D/gate_best.pt"
    for f in lut.jsonl phase3_nsga2_frontier.json; do
      [ -f "$ROOT/data/$f" ] && cp "$ROOT/data/$f" "$D/$f" || echo "  (skip absent data/$f)"
    done
    sub "$KAGGLE_DIR/dataset-metadata.json" > "$D/dataset-metadata.json"
    if kaggle datasets files "$KUSER/$DATASET_SLUG" >/dev/null 2>&1; then
      echo "Versioning existing $KUSER/$DATASET_SLUG ..."
      kaggle datasets version -p "$D" -m "update $(date -u +%FT%TZ)" --dir-mode zip
    else
      echo "Creating $KUSER/$DATASET_SLUG ..."
      kaggle datasets create -p "$D" --dir-mode zip
    fi
    ;;
  --status|status)
    kaggle kernels status "$KUSER/$KERNEL_SLUG"
    ;;
  --pull|pull)
    OUT="$ROOT/data/cp33_kaggle_out"; mkdir -p "$OUT"
    kaggle kernels output "$KUSER/$KERNEL_SLUG" -p "$OUT"
    echo "kernel output -> $OUT"
    ;;
  *)
    # Push (and run) the kernel. Build dir carries run.py + the substituted metadata.
    K="$BUILD/kernel"; rm -rf "$K"; mkdir -p "$K"
    cp "$KAGGLE_DIR/run.py" "$K/run.py"
    sub "$KAGGLE_DIR/kernel-metadata.json" > "$K/kernel-metadata.json"
    kaggle kernels push -p "$K"
    echo "pushed $KUSER/$KERNEL_SLUG — status: bash kaggle/push.sh --status | output: bash kaggle/push.sh --pull"
    ;;
esac
