#!/usr/bin/env bash
# Drive one winner-v2-OFA graft run on a Google Colab CLI session (terminal-rented free GPU).
#
#   bash cloud/colab_job.sh <session> [--gpu T4] -- <run_prune_graft.py args...>
#   bash cloud/colab_job.sh w1a -- --spec prune/specs/v2_act292.json --seed 0
#   KACCT=2 bash cloud/colab_job.sh w1b -- --ratios 0.50 --technique global_taylor --seed 0
#
# The VM clones the repo from GitHub (push first — same tripwire as kaggle/push.sh). Kernel
# state persists across `colab exec` calls; the train loop checkpoints every 10 epochs into
# /content/tfm_out, and `cloud/colab_pull.sh <session>` tarballs + downloads it (safe to run
# mid-train for ckpt insurance). ALWAYS `colab stop -s <session>` when done — idle VMs burn
# the free compute budget.
#
# One-time CLI auth (interactive — run it yourself, then re-run this script):
#   gcloud auth application-default login --scopes=openid,\
# https://www.googleapis.com/auth/cloud-platform,\
# https://www.googleapis.com/auth/userinfo.email,\
# https://www.googleapis.com/auth/colaboratory
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="$ROOT/.venv-cloud/bin/colab"
SECRETS_DIR="$ROOT/secrets"
[ -x "$COLAB" ] || { echo "missing $COLAB — python3 -m venv .venv-cloud && .venv-cloud/bin/pip install google-colab-cli"; exit 1; }

SESSION="${1:?usage: colab_job.sh <session> [--gpu T4] -- <entry args>}"; shift
GPU="T4"
if [ "${1:-}" = "--gpu" ]; then GPU="$2"; shift 2; fi
[ "${1:-}" = "--" ] && shift

# --- Kaggle dataset credentials (multi-account suffixes, same contract as kaggle/push.sh) ---
KACCT="${KACCT:-1}"
case "$KACCT" in
  1) SFX="" ;;
  2) SFX=" (Copy)" ;;
  3) SFX=" (Copy 2)" ;;
  *) echo "KACCT must be 1, 2 or 3 (got '$KACCT')"; exit 1 ;;
esac
TOKEN_FILE="$SECRETS_DIR/access_token$SFX"
USER_FILE="$SECRETS_DIR/kaggle_username$SFX"
for f in "access_token$SFX" "kaggle_username$SFX"; do
  if git -C "$ROOT" ls-files --error-unmatch "secrets/$f" >/dev/null 2>&1; then
    echo "REFUSING: secrets/$f is git-tracked — untrack it first."; exit 1
  fi
done
[ -f "$TOKEN_FILE" ] && [ -f "$USER_FILE" ] || { echo "missing $TOKEN_FILE / $USER_FILE"; exit 1; }

# --- tripwire: the VM clones GitHub, so local HEAD must be pushed -------------------------
if git -C "$ROOT" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
  git -C "$ROOT" fetch -q origin || true
  if [ "$(git -C "$ROOT" rev-parse HEAD)" != "$(git -C "$ROOT" rev-parse '@{u}')" ] \
     && [ -z "${KPUSH_ANYWAY:-}" ]; then
    echo "REFUSING: local HEAD != upstream — the VM clones GitHub, push first"
    echo "          (or KPUSH_ANYWAY=1 to override)."; exit 1
  fi
fi

# --- session (idempotent) ------------------------------------------------------------------
if ! "$COLAB" sessions 2>/dev/null | grep -q "$SESSION"; then
  echo "[colab_job] provisioning session '$SESSION' (--gpu $GPU)"
  "$COLAB" new -s "$SESSION" --gpu "$GPU"
fi

# --- secrets → VM (files, never argv) ------------------------------------------------------
echo "import os; os.makedirs('/content/tfm_secrets/secrets', exist_ok=True)" \
  | "$COLAB" exec -s "$SESSION"
"$COLAB" upload -s "$SESSION" "$TOKEN_FILE" /content/tfm_secrets/secrets/access_token
"$COLAB" upload -s "$SESSION" "$USER_FILE" /content/tfm_secrets/secrets/kaggle_username

# --- run (clone/refresh repo, then the platform-agnostic entry) ----------------------------
ENTRY_ARGS="$*"
REMOTE_CMD="set -e; [ -d /content/TFM_NAS ] || git clone https://github.com/Asilon47/TFM_NAS.git /content/TFM_NAS; cd /content/TFM_NAS; git fetch origin && git reset --hard origin/main; python colab/run_prune_graft.py --secrets-root /content/tfm_secrets --out-dir /content/tfm_out $ENTRY_ARGS"
echo "[colab_job] $REMOTE_CMD"
"$COLAB" exec -s "$SESSION" <<PYEOF
import subprocess, sys
rc = subprocess.run("""$REMOTE_CMD""", shell=True).returncode
print(f"[colab_job] remote rc={rc}", flush=True)
sys.exit(rc)
PYEOF
echo "[colab_job] done — pull artifacts: bash cloud/colab_pull.sh $SESSION"
echo "[colab_job] then release the VM: $COLAB stop -s $SESSION"
