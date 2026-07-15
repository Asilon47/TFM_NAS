#!/usr/bin/env bash
# CP 10.1 bring-up smoke: proves the reconstructed toolchain end to end.
#   1. helloworld compiled for GAP8 and executed on GVSOC ("Test success !")
#   2. nntool importable (the NN quantization front-end)
# Exits non-zero if either leg fails.
set -euo pipefail

MCU_DIR="$(cd "$(dirname "$0")" && pwd)"

echo '=== [1/2] helloworld on GVSOC ==='
out="$("${MCU_DIR}/run.sh" 'cd examples/gap8/basic/helloworld && make clean all run PMSIS_OS=freertos platform=gvsoc io=host 2>&1 | tail -20')"
echo "${out}"
grep -q 'Test success' <<<"${out}" || { echo 'SMOKE FAIL: no "Test success" from GVSOC helloworld' >&2; exit 1; }

echo '=== [2/2] nntool availability ==='
"${MCU_DIR}/run.sh" 'which nntool && nntool --help 2>&1 | head -5 || { echo "nntool CLI not on PATH — trying module import"; python3 -c "import sys; sys.path.insert(0, \"tools/nntool\"); import nntool; print(\"nntool import OK from tools/nntool\")"; }'

echo 'SMOKE PASS'
