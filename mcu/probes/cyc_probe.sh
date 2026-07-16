#!/usr/bin/env bash
# CP 10.1 DoD: simulated cycle counts per model.
#   nntool (cycle monitor ON, outputs -> HyperRAM) -> AutoTiler -> GAP8 kernels
#   -> link net_cyc.c app -> run on GVSOC -> parse AT_GraphPerf -> JSON.
#
# Emits data/mcu/cyc/<model>.{run.log,json}. The JSON is the CP 10.1 deliverable;
# cycles are SIM numbers and RANKING-ONLY (procedure.md, Stage 0 discipline) --
# they are never comparable to a Jetson millisecond.
#
#   mcu/probes/cyc_probe.sh                      # the CP 10.1 matched pair
#   mcu/probes/cyc_probe.sh graft_raw_224_mcu    # one model
#   CYC_L2=200000 mcu/probes/cyc_probe.sh        # re-tile with a tighter L2
#
# A failed stage is a RESULT to record (documented infeasibility satisfies the
# DoD too), not a script error -- hence no `set -e`.
set -uo pipefail

MCU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "${MCU_DIR}/.." && pwd)"
OUT_DIR="${REPO}/data/mcu/cyc"
mkdir -p "${OUT_DIR}"

# AutoTiler's L2 budget, and it is the binding constraint of this whole probe.
# AutoTiler is GREEDY (it consumes whatever it is given), but it shares GAP8's
# single 512 KB L2 with the binary itself -- and the generated graph's .text
# MEASURES ~329 KB for these 214-node models, leaving only ~180 KB of L2 heap.
# So the SDK's stock 400000 is not merely optimistic, it is unrunnable; that is
# what the app-less CP 10.1 codegen assumed (it only ever checked HyperRAM/flash
# fit, never the L2 an actual app leaves behind). Measured ladder: 240000 tiles
# but dies at runtime with construct rc=3 (L2 alloc); 70000 fails to tile at
# codegen ("Failed to allocate Buffer ... used in Kernel Bias"). ALL graph IO is
# pushed to HyperRAM (see net_cyc.c) to leave the heap to AutoTiler alone.
CYC_L2="${CYC_L2:-160000}"

# Cluster master stack. The SDK default resolves to 2048 B, and the generated
# <prefix>CNN() frame alone is ~2 KB (one HyperRAM event struct per DMA channel)
# -> PE0 overran it and GVSOC aborted ("SP-based access outside stack"). This
# must be raised AND passed to both make calls: model_decl.mk derives
# MODEL_L1_MEMORY = TARGET_L1_SIZE - (CLUSTER_STACK_SIZE + slave*7), so the
# stack comes straight out of AutoTiler's L1 share and codegen must see the same
# value the app does. Identical for every model, so the comparison stays fair.
CYC_MASTER_STACK="${CYC_MASTER_STACK:-8192}"

# Probe resolution. CP 10.1 used 224; CP 10.2's task shape is 160-320, and the
# resolution leg of that question is testable here with no training.
CYC_RES="${CYC_RES:-224}"

# Graph IO byte counts are read from each model's ONNX, never assumed. A closed
# form over CYC_RES holds only for the UNPRUNED raw-head pair; a spec-pruned
# graft has narrower adapter feats (71,925 out vs 91,525 at 160), and feeding the
# harness undersized output buffers would let the CNN write out of bounds. int8,
# so numel == bytes. Needs .venv-nas (onnx) -- the same venv that produced the
# exports.
NAS_PY="${REPO}/.venv-nas/bin/python"
io_bytes() {   # $1 = host path to the .onnx -> prints "IN O1 O2 O3 O4 O5 O6"
    "${NAS_PY}" - "$1" <<'PY'
import sys
import onnx
m = onnx.load(sys.argv[1], load_external_data=False)
def numel(t):
    n = 1
    for d in t.type.tensor_type.shape.dim:
        n *= (d.dim_value or 0)
    return n
vals = [numel(m.graph.input[0])] + [numel(o) for o in m.graph.output]
if len(vals) != 7:
    sys.exit(f"expected 1 input + 6 outputs, got {len(vals)-1} outputs")
print(" ".join(str(v) for v in vals))
PY
}

MODELS=(graft_raw_${CYC_RES}_mcu yolo11n_pose_${CYC_RES}_raw)
[[ $# -gt 0 ]] && MODELS=("$@")

for m in "${MODELS[@]}"; do
    build_host="${OUT_DIR}/${m}"
    build_ctr="/workspace/TFM_NAS/data/mcu/cyc/${m}"
    genlog="${build_host}.gen.log"
    runlog="${build_host}.run.log"
    mkdir -p "${build_host}"

    onnx_host="${REPO}/models/res${CYC_RES}/${m}.onnx"
    if ! read -r IN_B O1_B O2_B O3_B O4_B O5_B O6_B < <(io_bytes "${onnx_host}"); then
        echo "IO_FAIL  ${m}  (could not read IO shapes from ${onnx_host})"
        continue
    fi
    echo "=== ${m}: IO in=${IN_B} out=$((O1_B+O2_B+O3_B+O4_B+O5_B+O6_B)) B (from ONNX)"

    # IDENTICAL script for every model -- the pair only differs in its backbone,
    # so any cycle delta must not come from the quantization/codegen recipe.
    # One `fusions` call: each invocation re-runs adjust_order and a second pass
    # over a detection head raises "axes don't match array" (CP 10.1 fact 4).
    # default_{input,output}_* -> HyperRAM: with ~329 KB of .text there is not
    # enough L2 left for AutoTiler's working set AND the 330 KB of graph IO.
    cat > "${build_host}/nntool_script" <<EOF
adjust
fusions -a scaled_match_group expression_matcher
set default_input_home_location AT_MEM_L3_HRAM
set default_input_exec_location AT_MEM_L3_HRAM
set default_output_home_location AT_MEM_L3_HRAM
set default_output_exec_location AT_MEM_L3_HRAM
set l3_ram_device AT_MEM_L3_HRAM
set l3_flash_device AT_MEM_L3_HFLASH
set graph_produce_node_names true
set graph_produce_operinfos true
set graph_monitor_cycles true
aquant /workspace/TFM_NAS/data/mcu/probes/aq_sample/*.jpg -H ${CYC_RES} -W ${CYC_RES} -T
save_state ${build_ctr}/${m}
EOF

    # AutoTiler unconditionally emits `#include "<prefix>.h"` into its Kernels.h
    # and expects the app to supply it (the SDK ships a hand-written one per
    # example, e.g. mnist/mnist.h). Its only job is to declare the _L3_Flash
    # symbol the app defines. Ours is model-named, so generate it; -I$(MODEL_BUILD)
    # is already on the include path. The app-less at/ codegen probe never
    # compiled Kernels.h, which is why CP 10.1 never needed this.
    cat > "${build_host}/${m}.h" <<EOF
#ifndef __${m}_APP_H__
#define __${m}_APP_H__
#include "Gap.h"
extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE ${m}_L3_Flash;
#endif
EOF

    echo "=== ${m}: codegen (AutoTiler L2 budget ${CYC_L2}) ==="
    "${MCU_DIR}/run.sh" \
        "make -C /workspace/TFM_NAS/mcu/probes/cyc model \
            MODEL_PREFIX=${m} \
            MODEL_BUILD=${build_ctr} \
            TRAINED_MODEL=/workspace/TFM_NAS/models/res${CYC_RES}/${m}.onnx \
            MODEL_L2_MEMORY=${CYC_L2} \
            CLUSTER_STACK_SIZE=${CYC_MASTER_STACK} \
            NNTOOL_SCRIPT=${build_ctr}/nntool_script" \
        >"${genlog}" 2>&1
    if [[ ! -f "${build_host}/${m}Kernels.c" ]]; then
        echo "GEN_FAIL ${m}  (${genlog})"
        grep -iE 'error|abort|cannot|unable' "${genlog}" | head -5 | sed 's/^/    /'
        continue
    fi

    # Construct()'s HyperRAM arena size is a codegen output; the app needs it to
    # place its output buffers immediately above the arena.
    arena=$(grep -oE 'AT_HYPERRAM_ALLOC\(&HyperRam, [0-9]+\)' \
            "${build_host}/${m}Kernels.c" | head -1 | grep -oE '[0-9]+')
    if [[ -z "${arena}" ]]; then
        echo "GEN_FAIL ${m}  (no HyperRAM arena found -- graph has no L3 tensors?)"
        continue
    fi
    echo "    L3 arena ${arena} B; building app + running GVSOC (slow: full int8 CNN)"

    # net_cyc.o bakes in -DAT_MODEL_PREFIX, but make only tracks timestamps, so
    # switching models would relink the PREVIOUS model's object against these
    # kernels -- undefined refs if the names differ, the wrong graph silently
    # measured if they ever did not. Force the recompile. (Per-model kernel .o
    # live under their own MODEL_BUILD path and do not collide.)
    "${MCU_DIR}/run.sh" \
        "find /workspace/TFM_NAS/mcu/probes/cyc/BUILD -name 'net_cyc.o' -delete 2>/dev/null; \
         make -C /workspace/TFM_NAS/mcu/probes/cyc all run platform=gvsoc \
            MODEL_PREFIX=${m} \
            MODEL_BUILD=${build_ctr} \
            TRAINED_MODEL=/workspace/TFM_NAS/models/res${CYC_RES}/${m}.onnx \
            AT_INPUT_BYTES=${IN_B} \
            AT_OUT1_BYTES=${O1_B} AT_OUT2_BYTES=${O2_B} AT_OUT3_BYTES=${O3_B} \
            AT_OUT4_BYTES=${O4_B} AT_OUT5_BYTES=${O5_B} AT_OUT6_BYTES=${O6_B} \
            MODEL_L2_MEMORY=${CYC_L2} \
            CLUSTER_STACK_SIZE=${CYC_MASTER_STACK} \
            AT_L3_ARENA=${arena} \
            NNTOOL_SCRIPT=${build_ctr}/nntool_script" \
        >"${runlog}" 2>&1

    if grep -q '^CYC_TOTAL' "${runlog}"; then
        REPO="${REPO}" python3 - "${m}" "${runlog}" "${genlog}" "${CYC_L2}" "${arena}" <<'PY'
import json, os, re, sys

model, runlog, genlog, l2, arena = sys.argv[1:6]
run = open(runlog, errors="replace").read()
gen = open(genlog, errors="replace").read()

tot = re.search(r'^CYC_TOTAL sum_nodes=(\d+) at_total=(\d+) operations=(\d+) nodes=(\d+)',
                run, re.M)
nodes = [{"idx": int(i), "name": n, "cycles": int(c), "operations": int(o)}
         for i, n, c, o in re.findall(r'^CYC_NODE (\d+) (\S+) (\d+) (\d+)', run, re.M)]
mem = re.search(r'^CYC_MEM arena_base=(\d+) arena_bytes=(\d+) out_base=(\d+) '
                r'out_bytes=(\d+) in_l2_bytes=(\d+)', run, re.M)

def gm(pat):
    m = re.search(pat, gen)
    return int(m.group(1)) if m else None

cyc = int(tot.group(2)) or int(tot.group(1))
FREQ_CL_MHZ = 175  # GAP8 V3 cluster ceiling; the AI-deck 1.1 figure

rec = {
    "model": model,
    "source": "gap8_gvsoc",
    "precision": "int8",
    "note": "SIM cycles, RANKING-ONLY -- not comparable to measured Jetson ms",
    "cycles_sum_nodes": int(tot.group(1)),
    "cycles_at_total": int(tot.group(2)),
    "operations": int(tot.group(3)),
    "n_nodes": int(tot.group(4)),
    "ops_per_cycle": round(int(tot.group(3)) / cyc, 4) if cyc else None,
    "ms_at_175mhz": round(cyc / (FREQ_CL_MHZ * 1e6) * 1e3, 3) if cyc else None,
    "fps_at_175mhz": round((FREQ_CL_MHZ * 1e6) / cyc, 2) if cyc else None,
    "autotiler_l2_budget": int(l2),
    "l3_arena_bytes": int(arena),
    "mem": {
        "l1_used": gm(r'Shared L1 Memory size \(Bytes\)\s+: Given:\s+\d+, Used:\s+(\d+)'),
        "l2_used": gm(r'L2 Memory size \(Bytes\)\s+: Given:\s+\d+, Used:\s+(\d+)'),
        "hyperram_used": gm(r'HyperRam Memory size \(Bytes\)\s+: Given:\s+\d+, Used:\s+(\d+)'),
        "hyperflash_used": gm(r'HyperFlash Memory size \(Bytes\)\s+: Given:\s+\d+, Used:\s+(\d+)'),
        "input_l2_bytes": int(mem.group(5)) if mem else None,
        "output_l3_bytes": int(mem.group(4)) if mem else None,
    },
    "nodes": sorted(nodes, key=lambda n: -n["cycles"]),
}
path = os.path.join(os.environ["REPO"], "data", "mcu", "cyc", model + ".json")
with open(path, "w") as f:
    json.dump(rec, f, indent=2)
print(f"RUN_OK   {model}  {rec['cycles_at_total']:>12,} cycles  "
      f"{rec['ms_at_175mhz']:>8} ms @175MHz  ({rec['fps_at_175mhz']} FPS)  -> {path}")
PY
    else
        echo "RUN_FAIL ${m}  (${runlog})"
        grep -iE '^CYC_ERROR|error|abort|cannot|not enough|overflow' "${runlog}" \
            | head -6 | sed 's/^/    /'
    fi
done
