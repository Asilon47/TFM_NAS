#!/usr/bin/env bash
# Build the CP 10.1 GAP8 toolchain image (tag: tfm-gap8:cp10.1).
# Fetches + verifies the recovered AutoTiler blob first (mcu/vendor/ is
# gitignored; the image build fails without it).
set -euo pipefail

MCU_DIR="$(cd "$(dirname "$0")" && pwd)"

"${MCU_DIR}/fetch_tiler.sh"

exec docker build -t tfm-gap8:cp10.1 "${MCU_DIR}"
