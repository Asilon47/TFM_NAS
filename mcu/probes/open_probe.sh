#!/usr/bin/env bash
# CP 10.1 feasibility step 1: can the pinned nntool IMPORT each res224 ONNX?
# (import != deployable — quantization/codegen come next — but an import
# failure already decides the probe, so it gets its own recorded verdict.)
#
# Runs the toolchain image once per model via mcu/run.sh; full nntool output
# lands in data/mcu/probes/<model>.open.log. A failed import is a RESULT to
# record, not a script error, so the loop never aborts on one model.
set -uo pipefail

MCU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${MCU_DIR}/../data/mcu/probes"
mkdir -p "${OUT_DIR}"

# Probe-A ladder first (full graft op17 -> op12 -> backbone-only), then the
# yolo baselines (probe B).
MODELS=(graft_noneck_224 graft_noneck_224_op12 graft_backbone_224
        yolo11n_pose_224 dense_ctrl_n_224)

for m in "${MODELS[@]}"; do
    log="${OUT_DIR}/${m}.open.log"
    "${MCU_DIR}/run.sh" \
        "printf 'open /workspace/TFM_NAS/models/res224/${m}.onnx\nshow\nquit\n' | nntool" \
        >"${log}" 2>&1
    # `show` prints one row per graph step on success; exceptions surface as
    # tracebacks/ValueErrors in the same stream.
    steps="$(grep -cE '^\| *[0-9]+' "${log}" || true)"
    errs="$(grep -ciE 'traceback|valueerror|notimplemented|no.*importer|cannot|unsupported' "${log}" || true)"
    if [[ "${steps}" -gt 0 && "${errs}" -eq 0 ]]; then
        echo "OPEN_OK   ${m}  steps=${steps}  (${log})"
    else
        echo "OPEN_FAIL ${m}  steps=${steps} err_lines=${errs}  (${log})"
        grep -iE 'error|unsupported|not implemented|exception' "${log}" | head -5 | sed 's/^/    /'
    fi
done
