#!/usr/bin/env bash
# CP 10.1 feasibility step 2: adjust -> fusions --scale8 -> int8 aquant ->
# save_state, per model (the three AT-pipeline gates after import).
#
# Calibration here uses a small deterministic RGB sample straight from the
# dataset: quantization *statistics* don't affect cycle counts (cycles follow
# graph structure + scheme), so the honest grayscale set only enters at
# CP 10.2 with the retrained gray nets. A failed stage is a RESULT to record.
set -uo pipefail

MCU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "${MCU_DIR}/.." && pwd)"
OUT_DIR="${REPO}/data/mcu/probes"
SAMPLE_DIR="${OUT_DIR}/aq_sample"
mkdir -p "${OUT_DIR}" "${SAMPLE_DIR}"

# Deterministic 8-image calib sample (first 8 sorted train jpgs).
if [[ -z "$(ls -A "${SAMPLE_DIR}" 2>/dev/null)" ]]; then
    ls "${REPO}/dataset/images/train" | sort | head -8 \
        | while read -r f; do cp "${REPO}/dataset/images/train/${f}" "${SAMPLE_DIR}/"; done
fi

# Positional args override the model list.
MODELS=(graft_noneck_224_op12 graft_backbone_224 yolo11n_pose_224 dense_ctrl_n_224)
[[ $# -gt 0 ]] && MODELS=("$@")

for m in "${MODELS[@]}"; do
    log="${OUT_DIR}/${m}.quant.log"
    script="${OUT_DIR}/${m}.nn"
    state="/workspace/TFM_NAS/data/mcu/probes/${m}_state"
    cat > "${script}" <<EOF
open /workspace/TFM_NAS/models/res224/${m}.onnx
adjust
fusions --scale8
aquant /workspace/TFM_NAS/data/mcu/probes/aq_sample/*.jpg -H 224 -W 224 -T
save_state ${state}
EOF
    "${MCU_DIR}/run.sh" "nntool -s /workspace/TFM_NAS/data/mcu/probes/${m}.nn" \
        >"${log}" 2>&1
    if ls "${REPO}/data/mcu/probes/${m}_state"*.json >/dev/null 2>&1; then
        echo "QUANT_OK   ${m}  (${log})"
    else
        echo "QUANT_FAIL ${m}  (${log})"
        grep -iE 'exception|error|not implemented|unsupported|cannot' "${log}" | head -4 | sed 's/^/    /'
    fi
done
