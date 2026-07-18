#!/usr/bin/env bash
# Push the CP 3.3 search to Kaggle via the API — no manual uploads.
# Runs on the LAPTOP (.venv). Token read from the gitignored secrets/kaggle.json.
#
#   bash kaggle/push.sh --data     # (occasional) create/version the data Dataset
#   bash kaggle/push.sh            # (default) push/run the kernel
#   bash kaggle/push.sh --status   # kernel run status
#   bash kaggle/push.sh --pull     # download the kernel output into data/
#   bash kaggle/push.sh --cache    # version the resume cache Dataset from the pulled output
#   bash kaggle/push.sh --resume   # between sessions: --pull then --cache (then re-run in the UI)
#
# One-time: save your KGAT_ token at secrets/access_token + your Kaggle username
# at secrets/kaggle_username (or a legacy secrets/kaggle.json), then
# `pip install kaggle`. See kaggle/README.md.
#
# Multi-account (2026-07-07): KACCT=2|3 selects the "(Copy)"/"(Copy 2)" credential
# pair in secrets/ (three accounts run campaigns in parallel; each account owns its
# own copy of the gate-pose + cache Datasets — run `KACCT=N bash kaggle/push.sh --data`
# once per new account). KMODE=<mode> pushes the kernel with run.py's MODE line
# rewritten in the STAGED copy only (repo file untouched) so each account can run a
# different campaign from one codebase.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$ROOT/kaggle"
SECRETS_DIR="$ROOT/secrets"
BUILD="$KAGGLE_DIR/_build"
DATASET_SLUG="tfm-nas-gate-pose"
CACHE_SLUG="tfm-nas-cp33-bo-cache"   # small Dataset the eval caches round-trip through (resume store)
KERNEL_SLUG="tfm-nas-cp3-3-search"   # Kaggle derives the slug from the title ("CP3.3" -> "cp3-3")
DONOR="$ROOT/runs/pose/experiments/gate_baseline/weights/best.pt"

# --- credentials (local, gitignored) -----------------------------------------
# New-style token (KGAT_...) at secrets/access_token + username at
# secrets/kaggle_username; or a legacy secrets/kaggle.json ({username,key}).
# KACCT=2/3 -> the "(Copy)" / "(Copy 2)" pairs (parallel campaign accounts).
KACCT="${KACCT:-1}"
case "$KACCT" in
  1) SFX="" ;;
  2) SFX=" (Copy)" ;;
  3) SFX=" (Copy 2)" ;;
  *) echo "KACCT must be 1, 2 or 3 (got '$KACCT')"; exit 1 ;;
esac
TOKEN_FILE="$SECRETS_DIR/access_token$SFX"
LEGACY_JSON="$SECRETS_DIR/kaggle.json"
USER_FILE="$SECRETS_DIR/kaggle_username$SFX"
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
    for f in lut.jsonl phase3_nsga2_frontier.json cp33_acc_memo.json; do
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
    # Account 1 keeps the historical dir; extra accounts pull into their own.
    if [ "$KACCT" = "1" ]; then OUT="$ROOT/data/cp33_kaggle_out"
    else OUT="$ROOT/data/kaggle_out_$KUSER"; fi
    mkdir -p "$OUT"
    kaggle kernels output "$KUSER/$KERNEL_SLUG" -p "$OUT"
    echo "kernel output -> $OUT"
    ;;
  --cache|cache)
    # Persist the BO eval caches as a small Dataset so they survive across Kaggle
    # sessions — a notebook CANNOT attach its own output as its own input, so the
    # caches round-trip through this dataset instead. Seeds an empty dataset on the
    # first call; afterwards versions it from whatever `--pull` last fetched.
    # run.py rglobs cp33_bo_cache_r*.jsonl out of /kaggle/input to restore them.
    # The _r<RES> glob excludes pre-DoD validation shards (cp33_bo_cache.seedN.jsonl,
    # a different 2-seed format) that may linger in the pulled output.
    C="$BUILD/cache"; rm -rf "$C"; mkdir -p "$C"
    OUT="$ROOT/data/cp33_kaggle_out"
    n=0
    if [ -d "$OUT" ]; then
      for f in "$OUT"/cp33_bo_cache_r*.jsonl; do
        [ -e "$f" ] || continue
        cp "$f" "$C/"; n=$((n + 1))
      done
    fi
    # Small donor .pt files ride the same Dataset (beat-n program: PB_DONOR names one and
    # run.py find()s it under /kaggle/input) — drop them in data/kaggle_donors/.
    DONORS="$ROOT/data/kaggle_donors"
    if [ -d "$DONORS" ]; then
      for f in "$DONORS"/*; do
        [ -e "$f" ] || continue
        cp "$f" "$C/"; n=$((n + 1))
      done
    fi
    printf 'CP 3.3 BO eval caches (cp33_bo_cache_r<RES>.seed<N>.{bo,rs}.jsonl)\n+ small donor checkpoints (data/kaggle_donors/ -> PB_DONOR).\nResume store for the cross-session search; versioned by kaggle/push.sh --cache.\n' > "$C/cache_readme.txt"
    echo "staging $n file(s) -> $KUSER/$CACHE_SLUG"
    sub "$KAGGLE_DIR/cache-metadata.json" > "$C/dataset-metadata.json"
    if kaggle datasets files "$KUSER/$CACHE_SLUG" >/dev/null 2>&1; then
      kaggle datasets version -p "$C" -m "cache $(date -u +%FT%TZ)"
    else
      kaggle datasets create -p "$C"
    fi
    ;;
  --resume|resume)
    # One laptop command between sessions: pull the finished session's output, then
    # push its caches into the resume Dataset. Then in the Kaggle UI bump that dataset
    # to its newest version and Save & Run All — or just re-run `bash kaggle/push.sh`
    # (kernel-metadata now pins machine_shape=NvidiaTeslaT4, so an API re-push no
    # longer resets the accelerator to the P100 default; that reset caused the
    # 2026-07-06 'no kernel image for device' failure — torch dropped sm_60).
    bash "$KAGGLE_DIR/push.sh" --pull
    bash "$KAGGLE_DIR/push.sh" --cache
    echo "resume staged. Refresh the '$CACHE_SLUG' input to its latest version and"
    echo "Save & Run All — or re-push via 'bash kaggle/push.sh' (T4 pinned in metadata)."
    ;;
  *)
    # Push (and run) the kernel. Build dir carries run.py + the substituted metadata.
    # The cache Dataset is listed in kernel-metadata, so a push fails if it is missing
    # — seed it empty on the very first run so session 1 is friction-free.
    if ! kaggle datasets files "$KUSER/$CACHE_SLUG" >/dev/null 2>&1; then
      echo "Seeding empty resume cache dataset $KUSER/$CACHE_SLUG ..."
      bash "$KAGGLE_DIR/push.sh" --cache
    fi
    # Tripwire (2026-07-07 incident): the kernel `git clone`s the repo from GitHub, so any
    # repo-module the run needs must be PUSHED before the kernel launches — a locally-committed
    # but unpushed module fails with ModuleNotFoundError minutes later on Kaggle.
    if git -C "$ROOT" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
      git -C "$ROOT" fetch -q origin || true
      if [ "$(git -C "$ROOT" rev-parse HEAD)" != "$(git -C "$ROOT" rev-parse '@{u}')" ] \
         && [ -z "${KPUSH_ANYWAY:-}" ]; then
        echo "REFUSING: local HEAD != upstream — the kernel clones GitHub, push first"
        echo "          (or KPUSH_ANYWAY=1 to override)."; exit 1
      fi
    fi
    K="$BUILD/kernel"; rm -rf "$K"; mkdir -p "$K"
    if [ -n "${KMODE:-}" ]; then
      # Rewrite the MODE line in the STAGED copy only — one codebase, per-account campaigns.
      sed -E "s/^MODE       = \"[a-z_]+\"/MODE       = \"$KMODE\"/" "$KAGGLE_DIR/run.py" > "$K/run.py"
      grep -q "^MODE       = \"$KMODE\"" "$K/run.py" || { echo "KMODE sed failed"; exit 1; }
      echo "staged run.py with MODE=$KMODE"
    else
      cp "$KAGGLE_DIR/run.py" "$K/run.py"
    fi
    if [ -n "${KSET:-}" ]; then
      # KSET="NAME=VALUE;NAME2=VALUE2" rewrites knob constants in the STAGED copy only
      # (VALUE is a python literal). Per-account campaign knobs without touching the repo:
      #   KSET='PB_SPEC="prune/specs/s39d_act252.json";PB_KD=0;PB_SEED=1'
      IFS=';' read -ra _kvs <<< "$KSET"
      for kv in "${_kvs[@]}"; do
        name="${kv%%=*}"; val="${kv#*=}"
        grep -qE "^${name}[[:space:]]*=" "$K/run.py" || { echo "KSET: unknown knob '$name'"; exit 1; }
        sed -i -E "s|^${name}[[:space:]]*=[^#]*|${name} = ${val}  |" "$K/run.py"
        grep -qF "${name} = ${val}" "$K/run.py" || { echo "KSET sed failed for '$name'"; exit 1; }
      done
      echo "staged run.py with KSET: $KSET"
    fi
    sub "$KAGGLE_DIR/kernel-metadata.json" > "$K/kernel-metadata.json"
    kaggle kernels push -p "$K"
    echo "pushed $KUSER/$KERNEL_SLUG — status: bash kaggle/push.sh --status | output: bash kaggle/push.sh --pull"
    ;;
esac
