#!/usr/bin/env bash
# CP 10.1 feasibility step 3: AutoTiler codegen per model.
#   nntool -s (state) -> nntool -g (AT model C) -> gcc vs LibTile.a (GenTile)
#   -> GenTile --L1/--L2/--L3 (kernels + memory plan for GAP8)
# Success artifact: <build>/<model>Kernels.c. GenTile's stdout (captured in
# the log) carries the memory-allocation report — the L2-fit verdict.
# A failed stage is a RESULT to record, not a script error.
set -uo pipefail

MCU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "${MCU_DIR}/.." && pwd)"
OUT_DIR="${REPO}/data/mcu/at_build"
mkdir -p "${OUT_DIR}"

MODELS=(graft_backbone_224_mcu graft_noneck_224_mcu yolo11n_pose_224 dense_ctrl_n_224)
[[ $# -gt 0 ]] && MODELS=("$@")

for m in "${MODELS[@]}"; do
    build_host="${OUT_DIR}/${m}"
    build_ctr="/workspace/TFM_NAS/data/mcu/at_build/${m}"
    log="${build_host}.gen.log"
    mkdir -p "${build_host}"
    # model_rules.mk runs `nntool -s SCRIPT model.onnx`: the model arrives via
    # argv (no `open` line) and save_state must match MODEL_STATE.
    # ONE fusions call listing both groups: `fusions` re-runs adjust_order on
    # every invocation, and a second adjust_order over a detection head's
    # reshape/transpose set raises "axes don't match array" (it killed the
    # full graft AND yolo11n identically). -a takes nargs='+' and builds a
    # single match group, so adjust_order runs once.
    # expression_matcher is required: --scale8 alone leaves the h-swish
    # Mul/HSigmoid chains without an AT kernel ("Don't know how to generate
    # kernel for parameter type ... mul"). Image patch 0001 keeps those
    # expressions OUT of the convs (no KOP_CUSTOM DW kernel exists).
    cat > "${build_host}/nntool_script" <<EOF
adjust
fusions -a scaled_match_group expression_matcher
aquant /workspace/TFM_NAS/data/mcu/probes/aq_sample/*.jpg -H 224 -W 224 -T
save_state ${build_ctr}/${m}
EOF
    "${MCU_DIR}/run.sh" \
        "make -C /workspace/TFM_NAS/mcu/probes/at model \
            MODEL_PREFIX=${m} \
            MODEL_BUILD=${build_ctr} \
            NNTOOL_SCRIPT=${build_ctr}/nntool_script" \
        >"${log}" 2>&1
    if [[ -f "${build_host}/${m}Kernels.c" ]]; then
        l1=$(grep -oE 'L1 Memory size \(Bytes\)[^,]*' "${log}" | tail -1)
        echo "GEN_OK   ${m}  (${l1:-memory report in log})  (${log})"
    else
        echo "GEN_FAIL ${m}  (${log})"
        grep -iE 'error|exception|failed|cannot|unable' "${log}" | head -5 | sed 's/^/    /'
    fi
done
