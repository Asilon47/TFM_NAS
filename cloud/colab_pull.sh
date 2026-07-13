#!/usr/bin/env bash
# Pull /content/tfm_out from a Colab CLI session into data/colab_out/<session>/.
# Safe to run mid-train as ckpt insurance (tar + download of the durable out-dir).
#
#   bash cloud/colab_pull.sh <session> [dest-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="$ROOT/.venv-cloud/bin/colab"
SESSION="${1:?usage: colab_pull.sh <session> [dest-dir]}"
DEST="${2:-$ROOT/data/colab_out/$SESSION}"

mkdir -p "$DEST"
"$COLAB" exec -s "$SESSION" <<'PYEOF'
import subprocess, sys
rc = subprocess.run(
    "tar czf /content/tfm_out.tgz -C /content tfm_out && ls -la /content/tfm_out.tgz",
    shell=True).returncode
sys.exit(rc)
PYEOF
TMP="$(mktemp -u /tmp/tfm_out_XXXX.tgz)"
"$COLAB" download -s "$SESSION" /content/tfm_out.tgz "$TMP"
tar xzf "$TMP" -C "$DEST" --strip-components=1
rm -f "$TMP"
echo "[colab_pull] -> $DEST"
ls -la "$DEST" | tail -n +2
