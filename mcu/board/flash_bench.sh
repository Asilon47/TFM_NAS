#!/usr/bin/env bash
# Flash a built net_bench image to the AI-deck GAP8 over radio (the proven path;
# the wifi-img-streamer flashed the same way). Dongle within a few cm of the deck,
# deck powered. deck-bcAI:gap8-fw = the GAP8 HyperFlash target.
#
#   mcu/board/flash_bench.sh cand_a5fddcc354bd
#   CF_URI=radio://0/80/2M mcu/board/flash_bench.sh cand_a5fddcc354bd
set -uo pipefail

MODEL="${1:?usage: flash_bench.sh <model>}"
EX_ROOT="${AIDECK_EXAMPLES:-$HOME/aideck-gap8-examples}"
URI="${CF_URI:-radio://0/80/2M}"
IMG="${EX_ROOT}/examples/ai/net-bench-${MODEL}/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img"

[[ -f "${IMG}" ]] || { echo "NO IMAGE: ${IMG} (run build_bench.sh + the docker make first)"; exit 2; }
echo "flashing ${IMG}"
echo "     -> ${URI}  deck-bcAI:gap8-fw"
exec uvx --from cfclient cfloader flash "${IMG}" deck-bcAI:gap8-fw -w "${URI}"
