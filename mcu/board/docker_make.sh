#!/usr/bin/env bash
# Build a net-bench app dir INSIDE the Bitcraze docker, handling the footguns a
# plain `make` hits on this deck:
#   1. make must run in-container with the GAP SDK sourced -- a host `make` dies
#      with "Source sourceme in gap_sdk first".
#   2. this image's nntool crashes on `np.float` (old sklearn imports it, but the
#      image ships numpy 1.24.3 which removed the alias). Pinning numpy<1.24
#      restores it. Verified 2026-07-23; the streamer/helloworld never hit it
#      because they have no neural net -> no nntool step.
#
#   mcu/board/docker_make.sh cand_a5fddcc354bd
#   mcu/board/docker_make.sh cand_a5fddcc354bd clean all image EXTRA_CFLAGS=-DBENCH_SMOKE=3
set -uo pipefail

MODEL="${1:?usage: docker_make.sh <model> [make args...]}"
shift || true
EX_ROOT="${AIDECK_EXAMPLES:-$HOME/aideck-gap8-examples}"
IMAGE="${AIDECK_IMAGE:-bitcraze/aideck}"
APP="examples/ai/net-bench-${MODEL}"
MAKEARGS="${*:-clean all image}"

[[ -d "${EX_ROOT}/${APP}" ]] || {
    echo "NO app dir: ${EX_ROOT}/${APP} (run mcu/board/build_bench.sh ${MODEL} <res> first)"
    exit 2
}

echo "docker: ${IMAGE}   app: ${APP}   make ${MAKEARGS}"
exec docker run --rm -v "${EX_ROOT}:/module" "${IMAGE}" /bin/bash -c \
    "pip3 install 'numpy<1.24' -q 2>/dev/null; \
     source /gap_sdk/configs/ai_deck.sh; \
     cd /module/${APP} && make ${MAKEARGS}"
