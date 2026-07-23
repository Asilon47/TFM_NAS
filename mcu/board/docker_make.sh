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
#   mcu/board/docker_make.sh cand_a5fddcc354bd clean model image EXTRA_CFLAGS=-DBENCH_SMOKE=3
#
# Default target is `clean model image`, NOT `all`: the SDK's `all:: build image flash_fs`
# chains into a JTAG flash (gap8-openocd + an Olimex adapter we don't have), which aborts
# make AFTER the .img is already written. `model` runs the AutoTiler codegen; `image` builds
# the ELF + packs target.board.devices.flash.img. We flash that over RADIO with cfloader.
set -uo pipefail

MODEL="${1:?usage: docker_make.sh <model> [make args...]}"
shift || true
EX_ROOT="${AIDECK_EXAMPLES:-$HOME/aideck-gap8-examples}"
IMAGE="${AIDECK_IMAGE:-bitcraze/aideck}"
APP="examples/ai/net-bench-${MODEL}"
MAKEARGS="${*:-clean model image}"

[[ -d "${EX_ROOT}/${APP}" ]] || {
    echo "NO app dir: ${EX_ROOT}/${APP} (run mcu/board/build_bench.sh ${MODEL} <res> first)"
    exit 2
}

echo "docker: ${IMAGE}   app: ${APP}   make ${MAKEARGS}"
# In-container prep before make (the image is fresh each run -> --rm):
#   - pin numpy<1.24 (nntool's sklearn uses the removed np.float alias);
#   - drop the vendored LibTile.a (build_bench.sh shipped it) into the paths AutoTiler
#     links -- the image ships the generators but not the closed-source blob;
#   - apply patch 0001 to the image's nntool so MBv3 h-swish stays a standalone
#     expression kernel instead of a _Custom DW-conv activation GenTile can't generate.
exec docker run --rm -v "${EX_ROOT}:/module" "${IMAGE}" /bin/bash -c \
    "pip3 install 'numpy<1.24' -q 2>/dev/null; \
     if [ -f /module/${APP}/LibTile.a ]; then \
       mkdir -p /gap_sdk/tools/autotiler_v3/Autotiler; \
       cp /module/${APP}/LibTile.a /gap_sdk/tools/autotiler_v3/Autotiler/LibTile.a; \
       cp /module/${APP}/LibTile.a /gap_sdk/tools/autotiler_v3/libtile.4.3.5.a; \
     fi; \
     if [ -f /module/${APP}/nntool_no_expr_fusion.patch ]; then \
       patch -p1 -d /gap_sdk -i /module/${APP}/nntool_no_expr_fusion.patch || echo 'PATCH FAILED (nntool expr-fusion)'; \
     fi; \
     source /gap_sdk/configs/ai_deck.sh; \
     cd /module/${APP} && make ${MAKEARGS}"
