#!/usr/bin/env bash
# Drive one winner-v2-OFA graft run on a Google Colab CLI session (terminal-rented free GPU).
#
#   bash cloud/colab_job.sh launch <session> [--gpu T4] -- <run_prune_graft.py args...>
#   bash cloud/colab_job.sh poll   <session>            # tail the VM log once (rc line = done)
#   bash cloud/colab_job.sh pull   <session> [dest]     # download /content/tfm_out
#   bash cloud/colab_job.sh stop   <session>            # release the VM
#
#   bash cloud/colab_job.sh launch w1a -- --spec prune/specs/v2_act292.json --seed 0
#
# WHY launch+poll (not one exec): a 100-epoch run far outlives the `colab exec` websocket
# read timeout. So `launch` starts the entry DETACHED (nohup > VM log) via a SHORT exec and
# returns; the VM kernel + the nohup child both survive a client disconnect. `poll` tails the
# VM log with another short exec — the sentinel line `ENTRY_EXIT=<rc>` marks completion. The
# entry checkpoints every 10 epochs into /content/tfm_out, so `pull` is safe mid-run.
# ALWAYS `stop` when done — idle VMs burn the free compute budget.
#
# The VM clones the repo from GitHub (push first — same tripwire as kaggle/push.sh).
# One-time CLI auth (interactive): `.venv-cloud/bin/colab whoami` and follow the URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="$ROOT/.venv-cloud/bin/colab"
SECRETS_DIR="$ROOT/secrets"
TMP="${TMPDIR:-/tmp}"
[ -x "$COLAB" ] || { echo "missing $COLAB — python3 -m venv .venv-cloud && .venv-cloud/bin/pip install google-colab-cli"; exit 1; }

ACTION="${1:?usage: colab_job.sh launch|poll|pull|stop <session> ...}"; shift
SESSION="${1:?need a session name}"; shift || true

vlog() { "$COLAB" exec -s "$SESSION" -f "$1" 2>&1 | sed 's/\x1b\[[0-9;]*m//g'; }

case "$ACTION" in
  launch)
    GPU="T4"
    [ "${1:-}" = "--gpu" ] && { GPU="$2"; shift 2; }
    [ "${1:-}" = "--" ] && shift
    ENTRY_ARGS="$*"

    # --- Kaggle dataset creds (multi-account suffixes, kaggle/push.sh contract) ---------
    KACCT="${KACCT:-1}"
    case "$KACCT" in
      1) SFX="" ;; 2) SFX=" (Copy)" ;; 3) SFX=" (Copy 2)" ;;
      *) echo "KACCT must be 1, 2 or 3 (got '$KACCT')"; exit 1 ;;
    esac
    TOKEN_FILE="$SECRETS_DIR/access_token$SFX"; USER_FILE="$SECRETS_DIR/kaggle_username$SFX"
    for f in "access_token$SFX" "kaggle_username$SFX"; do
      git -C "$ROOT" ls-files --error-unmatch "secrets/$f" >/dev/null 2>&1 && \
        { echo "REFUSING: secrets/$f is git-tracked — untrack it first."; exit 1; }
    done
    [ -f "$TOKEN_FILE" ] && [ -f "$USER_FILE" ] || { echo "missing $TOKEN_FILE / $USER_FILE"; exit 1; }

    # --- tripwire: the VM clones GitHub, so local HEAD must be pushed -------------------
    if git -C "$ROOT" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
      git -C "$ROOT" fetch -q origin || true
      if [ "$(git -C "$ROOT" rev-parse HEAD)" != "$(git -C "$ROOT" rev-parse '@{u}')" ] \
         && [ -z "${KPUSH_ANYWAY:-}" ]; then
        echo "REFUSING: local HEAD != upstream — the VM clones GitHub, push first"
        echo "          (or KPUSH_ANYWAY=1 to override)."; exit 1
      fi
    fi

    if ! "$COLAB" sessions 2>/dev/null | grep -q "$SESSION"; then
      echo "[colab_job] provisioning '$SESSION' (--gpu $GPU)"
      "$COLAB" new -s "$SESSION" --gpu "$GPU"
    fi

    # Colab CLI /content is EPHEMERAL — a VM recycle loses the resume ckpt. If a prior
    # `poll --pull` snapshotted ckpts to the laptop, re-push them so recover_graft resumes
    # (durability parity with Lightning's persistent disk).
    printf "import os\nos.makedirs('/content/tfm_out', exist_ok=True)\n" > "$TMP/mkout.py"
    vlog "$TMP/mkout.py" >/dev/null || true
    for ck in "$ROOT/data/colab_out/$SESSION"/ckpt_*.pt; do
      [ -e "$ck" ] || continue
      echo "[colab_job] re-pushing $(basename "$ck") for resume"
      "$COLAB" upload -s "$SESSION" "$ck" "/content/tfm_out/$(basename "$ck")"
    done

    # secrets → VM as files (never argv)
    printf "import os\nos.makedirs('/content/tfm_secrets/secrets', exist_ok=True)\n" \
      > "$TMP/mkdir_sec.py"; vlog "$TMP/mkdir_sec.py" >/dev/null
    "$COLAB" upload -s "$SESSION" "$TOKEN_FILE" /content/tfm_secrets/secrets/access_token
    "$COLAB" upload -s "$SESSION" "$USER_FILE"  /content/tfm_secrets/secrets/kaggle_username

    # launch DETACHED: clone/refresh repo, nohup the entry, append rc sentinel, return now
    cat > "$TMP/launch_${SESSION}.py" << PYEOF
import subprocess
REMOTE = r'''set -e
[ -d /content/TFM_NAS ] || git clone https://github.com/Asilon47/TFM_NAS.git /content/TFM_NAS
cd /content/TFM_NAS && git fetch origin && git reset --hard origin/main
LOG=/content/run_${SESSION}.log
nohup bash -c "python colab/run_prune_graft.py --secrets-root /content/tfm_secrets --out-dir /content/tfm_out $ENTRY_ARGS; echo ENTRY_EXIT=\\\$? >> \$LOG" > \$LOG 2>&1 &
echo LAUNCHED pid=\$!'''
print(subprocess.run(REMOTE, shell=True, capture_output=True, text=True).stdout)
PYEOF
    echo "[colab_job] launching detached on '$SESSION'"
    vlog "$TMP/launch_${SESSION}.py"
    echo "[colab_job] poll:  bash cloud/colab_job.sh poll $SESSION"
    echo "[colab_job] pull:  bash cloud/colab_job.sh pull $SESSION"
    echo "[colab_job] stop:  bash cloud/colab_job.sh stop $SESSION"
    ;;

  poll)
    cat > "$TMP/poll_${SESSION}.py" << PYEOF
import subprocess
print(subprocess.run(
    "tail -n 40 /content/run_${SESSION}.log 2>/dev/null; "
    "echo '---'; pgrep -f run_prune_graft >/dev/null && echo RUNNING || echo NOT_RUNNING",
    shell=True, capture_output=True, text=True).stdout)
PYEOF
    vlog "$TMP/poll_${SESSION}.py"
    ;;

  pull)
    DEST="${1:-$ROOT/data/colab_out/$SESSION}"; mkdir -p "$DEST"
    printf "import subprocess\nsubprocess.run('tar czf /content/tfm_out.tgz -C /content tfm_out', shell=True)\n" \
      > "$TMP/tar_${SESSION}.py"; vlog "$TMP/tar_${SESSION}.py" >/dev/null
    T="$(mktemp -u "$TMP/tfm_out_XXXX.tgz")"
    "$COLAB" download -s "$SESSION" /content/tfm_out.tgz "$T"
    tar xzf "$T" -C "$DEST" --strip-components=1; rm -f "$T"
    echo "[colab_job] -> $DEST"; ls -la "$DEST" | tail -n +2
    ;;

  stop)
    "$COLAB" stop -s "$SESSION"; echo "[colab_job] stopped $SESSION"
    ;;

  *) echo "unknown action '$ACTION' (launch|poll|pull|stop)"; exit 1 ;;
esac
