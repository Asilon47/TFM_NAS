#!/usr/bin/env bash
# Run a command inside the GAP8 toolchain image with this repo mounted at
# /workspace/TFM_NAS (so data/, models/, mcu/ scripts are all visible).
#
#   mcu/run.sh                      interactive shell (source
#                                   /opt/gap_sdk/sourceme.sh --board gapuino
#                                   yourself if you need the SDK env)
#   mcu/run.sh 'cmd ...'            run cmd with the SDK env pre-sourced
#
# GAP8_IMAGE overrides the image (e.g. the community-proven fallback:
#   GAP8_IMAGE=cbezaitis/gap:latest mcu/run.sh '...').
set -euo pipefail

IMAGE="${GAP8_IMAGE:-tfm-gap8:cp10.1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -eq 0 ]]; then
    exec docker run --rm -it -v "${REPO}:/workspace/TFM_NAS" -w /opt/gap_sdk \
        "${IMAGE}" bash
fi

exec docker run --rm -v "${REPO}:/workspace/TFM_NAS" -w /opt/gap_sdk \
    "${IMAGE}" bash -lc "source sourceme.sh --board gapuino >/dev/null 2>&1 && $*"
