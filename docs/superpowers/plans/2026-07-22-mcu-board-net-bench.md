# MCU board net-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flash each of the 4 MCU finalists to the real GAP8 and report its measured `AT_GraphPerf` cycles + wall-clock FPS back over the radio/WiFi CPX console.

**Architecture:** Fork Bitcraze's `examples/ai/classification` (a working board NN app: FreeRTOS + CPX + the nntool→AutoTiler board build) and transplant `mcu/probes/cyc/net_cyc.c`'s core (HyperRAM IO placement + `AT_GraphPerf` read), adding a timing loop. Everything model-specific is generated per-model into a self-contained throwaway example dir; the firmware C and Makefile are model-agnostic. One `.img` per model (L2 can't hold two 400 KB graphs).

**Tech Stack:** GAP8 / GAP-SDK (nntool, AutoTiler, PMSIS, FreeRTOS), CPX, `bitcraze/aideck-toolchain` docker, `cfloader` (cflib) radio flashing, Python 3 + cflib host receiver, pytest.

## Global Constraints

- Source of truth lives in **TFM_NAS** under `mcu/board/`; it is copied into `~/aideck-gap8-examples/examples/ai/` at build time. Never edit the sim probes (`mcu/probes/`) — this is additive.
- **The board build docker is `bitcraze/aideck-toolchain`** (the image that built the wifi-img-streamer for `GAP8_V2`). NOT the `tfm-gap8` sim image (GVSOC-only, cannot produce a FreeRTOS binary).
- The four finalists, exact files + resolution:
  - `cand_a5fddcc354bd` @192 → `models/res192/cand_a5fddcc354bd.onnx` (winner)
  - `cand_19efff428be8` @192 → `models/res192/cand_19efff428be8.onnx` (speed)
  - `cand_863c75818953` @192 → `models/res192/cand_863c75818953.onnx` (acc-max)
  - `yolo11n_pose_160_raw` @160 → `models/res160/yolo11n_pose_160_raw.onnx` (baseline)
- All four are raw-head exports: **1 int8 input, 6 int8 outputs**. The CNN entry call is 7-arg.
- Cluster clock is set to **175 MHz** on the board (the sim's basis) so board-cyc and sim-cyc share one clock.
- Cycles are **ranking-only** (LUT/Stage-0 discipline). The wall-clock ms is the honest absolute.
- Per-model tunables (starting values): `MODEL_L2_MEMORY` = 84000 (grafts) / 160000 (baseline) — lower if `Construct` returns rc=3; `CLUSTER_STACK_SIZE` = 8192.
- Firmware prints **integers only** over CPX (GAP8 newlib-nano `%f` is unreliable); the host derives ms/FPS.
- Calibration set for `aquant`: `data/mcu/probes/aq_sample/*.jpg` (the sim probe's set; `aquant` resizes via `-H/-W`).
- Commit after each task. No `Co-Authored-By:` / "Generated with Claude Code" trailer.

---

### Task 1: `net_bench.c` — the firmware (model-agnostic)

**Files:**
- Create: `mcu/board/net_bench/net_bench.c`

**Interfaces:**
- Consumes (compile-time `-D`, supplied by Task 2/3): `AT_MODEL_PREFIX` (bare token), `AT_MODEL_KERNELS_H` (string), `AT_INPUT_BYTES`, `AT_OUT1_BYTES`..`AT_OUT6_BYTES`, `AT_MODEL_RES`, `STACK_SIZE`, `SLAVE_STACK_SIZE`, optional `BENCH_ITERS`/`BENCH_WARMUP`/`FREQ_CL`/`FREQ_FC`/`BENCH_SMOKE`.
- Consumes (from the generated AutoTiler graph, via `AT_MODEL_KERNELS_H`): `<prefix>CNN`, `<prefix>CNN_Construct`, `<prefix>CNN_Destruct`, `<prefix>_L3_Flash`, `<prefix>_L3_Memory`; and under `-DPERF`: `AT_GraphPerf`, `AT_GraphPerf_CNN_Total`.
- Produces: a CPX console line `BENCH model=<s> res=<i> cyc=<i> nodes=<i> clk_us=<i> n=<i> fcl=<i>` consumed by Task 4's parser.

- [ ] **Step 1: Write the firmware**

Note on verification: this file targets the GAP8 and **cannot be compiled or run in this environment** — the real gate is Task 6 (build + flash on hardware). The local checks in Steps 2–3 are static-consistency greps against the two reference files it is composed from (`mcu/probes/cyc/net_cyc.c`, `~/aideck-gap8-examples/examples/ai/classification/classification.c`).

Write `mcu/board/net_bench/net_bench.c`:

```c
/*
 * mcu/board/net_bench/net_bench.c
 *
 * On-silicon cycle + FPS bench for one AutoTiler int8 graph on the GAP8 AI-deck.
 * Composition (see docs/superpowers/specs/2026-07-22-mcu-board-net-bench-design.md):
 *   - board shell (freq/uart/cpx/cluster/FreeRTOS) from Bitcraze classification.c
 *   - graph-run + HyperRAM-IO + AT_GraphPerf core from mcu/probes/cyc/net_cyc.c
 *   - a timing loop over N runs -> wall-clock FPS
 * No camera: the input is a fixed, uninitialised HyperRAM buffer. int8 AutoTiler
 * cycles are data-independent (fixed tile loops, no sparsity/branch path), so its
 * contents cannot move the number this bench produces (net_cyc.c argues the same).
 *
 * Numbers are SIM-discipline: AT_GraphPerf cycles are RANKING-ONLY; the wall-clock
 * ms/FPS is the honest absolute. Output goes over CPX (LOG_TO_CRTP) -> readable in
 * cfclient's Console tab over radio, or over WiFi.
 */

#include "pmsis.h"
#include "bsp/bsp.h"
#include "cpx.h"
#include AT_MODEL_KERNELS_H

/* Paste the bare AT_MODEL_PREFIX token onto the AutoTiler symbol names. The extra
 * indirection forces the argument to expand before ## sees it (net_cyc.c trick). */
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)
#define CNN_RUN CAT(AT_MODEL_PREFIX, CNN)
#define CNN_CONSTRUCT CAT(AT_MODEL_PREFIX, CNN_Construct)
#define CNN_DESTRUCT CAT(AT_MODEL_PREFIX, CNN_Destruct)
#define CNN_L3_FLASH CAT(AT_MODEL_PREFIX, _L3_Flash)
#define CNN_L3_MEMORY CAT(AT_MODEL_PREFIX, _L3_Memory)

/* Stringify the prefix token for the report line (no reliance on an external -D). */
#define BENCH_XSTR(s) #s
#define BENCH_STR(s) BENCH_XSTR(s)

#ifndef STACK_SIZE
#define STACK_SIZE 8192
#endif
#ifndef SLAVE_STACK_SIZE
#define SLAVE_STACK_SIZE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif
#ifndef BENCH_WARMUP
#define BENCH_WARMUP 2
#endif
#ifndef FREQ_CL
#define FREQ_CL 175
#endif
#ifndef FREQ_FC
#define FREQ_FC 250
#endif
#ifndef AT_MODEL_RES
#define AT_MODEL_RES 0
#endif

/* The app owns _L3_Flash (AutoTiler's Kernels.h references it); 0 = base of the
 * READFS-flashed tensor file. Same contract as classification.c / net_cyc.c. */
AT_HYPERFLASH_FS_EXT_ADDR_TYPE CNN_L3_FLASH = 0;
/* Base of the arena CNN_Construct() allocated in its own HyperRAM device. Valid
 * only after Construct(). */
extern AT_HYPERRAM_POINTER CNN_L3_MEMORY;

#define L3_ALIGN 16u
#define ALIGN_UP(x) (((x) + (L3_ALIGN - 1u)) & ~(L3_ALIGN - 1u))

/* Graph IO lives in HyperRAM (forced): the generated .text is ~330-405 KB of the
 * 512 KB L2, so neither the input nor the ~180 KB of raw-head outputs fit in L2.
 * The nntool script sets default_{input,output}_location = HRAM; the app packs the
 * buffers into the free HyperRAM immediately BELOW the arena (extern_alloc serves
 * from the top, so [0, CNN_L3_MEMORY) is free). Derived from the observed arena
 * base, so it is independent of the true device size (net_cyc.c). */
static AT_HYPERRAM_POINTER In_L3;
static AT_HYPERRAM_POINTER Out_L3[6];
static const unsigned int OutBytes[6] = {AT_OUT1_BYTES, AT_OUT2_BYTES, AT_OUT3_BYTES,
                                         AT_OUT4_BYTES, AT_OUT5_BYTES, AT_OUT6_BYTES};

static int cnn_err;
static struct pi_device cluster_dev;
static struct pi_cluster_task *cl_task;

/* Cluster entry: reset the hw timer, run the 7-arg raw-head graph. */
static void run_cnn(void *arg)
{
    (void)arg;
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    cnn_err = CNN_RUN((signed char *)In_L3, (signed char *)Out_L3[0],
                      (signed char *)Out_L3[1], (signed char *)Out_L3[2],
                      (signed char *)Out_L3[3], (signed char *)Out_L3[4],
                      (signed char *)Out_L3[5]);
}

static void net_bench(void)
{
    /* Clocks: CL=175 to match the sim's basis; report FREQ_CL in the line so
     * cyc<->ms is reconstructable on the host. Voltage first, as classification. */
    __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC * 1000 * 1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL * 1000 * 1000);

    struct pi_uart_conf uart_conf;
    struct pi_device uart_dev;
    pi_uart_conf_init(&uart_conf);
    uart_conf.baudrate_bps = 115200;
    pi_open_from_conf(&uart_dev, &uart_conf);
    if (pi_uart_open(&uart_dev)) {
        pmsis_exit(-1);
    }

    cpxInit();
    cpxEnableFunction(CPX_F_WIFI_CTRL);
    cpxPrintToConsole(LOG_TO_CRTP, "*** net_bench %s res=%d iters=%d ***\n",
                      BENCH_STR(AT_MODEL_PREFIX), AT_MODEL_RES, BENCH_ITERS);

    /* Staged bring-up (ported from net_cyc.c CYC_SMOKE): a GAP8 crash reports as
     * SILENCE (buffered output lost on abort), so staging is the only way to see
     * how far an L2-fit failure got. 1 boot 2 cluster 3 construct 4 io 5 dispatch. */
#if BENCH_SMOKE == 1
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 1 boot ok\n");
    pmsis_exit(0);
    return;
#endif

    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev)) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR cluster open\n");
        pmsis_exit(-2);
    }

#if BENCH_SMOKE == 2
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 2 cluster ok\n");
    pmsis_exit(0);
    return;
#endif

    cnn_err = CNN_CONSTRUCT();
    if (cnn_err) {
        cpxPrintToConsole(LOG_TO_CRTP,
                          "ERR construct rc=%d (1 flash,2 L3,3 L2,4 L1)\n", cnn_err);
        pmsis_exit(-3);
    }

#if BENCH_SMOKE == 3
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 3 construct ok arena_base=%u\n",
                      (unsigned int)CNN_L3_MEMORY);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    /* Pack IO into the free HyperRAM below the arena. Bound-checked against the
     * runtime arena base (CNN_L3_MEMORY) -- AT_L3_ARENA is NOT needed (net_cyc.c
     * uses it only in a diagnostic), which keeps this build single-pass. */
    unsigned int out_span = ALIGN_UP(AT_INPUT_BYTES);
    for (int i = 0; i < 6; i++) out_span += ALIGN_UP(OutBytes[i]);
    if (out_span > (unsigned int)CNN_L3_MEMORY) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR IO !fit arena_base=%u io=%u\n",
                          (unsigned int)CNN_L3_MEMORY, out_span);
        pmsis_exit(-4);
    }
    In_L3 = (CNN_L3_MEMORY - out_span) & ~(L3_ALIGN - 1u);
    Out_L3[0] = In_L3 + ALIGN_UP(AT_INPUT_BYTES);
    for (int i = 1; i < 6; i++) Out_L3[i] = Out_L3[i - 1] + ALIGN_UP(OutBytes[i - 1]);

#if BENCH_SMOKE == 4
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 4 io placed base=%u span=%u\n",
                      (unsigned int)In_L3, out_span);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    /* Heap cluster task; set BOTH stacks (the grafts overran the 2 KB master
     * default in sim -> "SP-based access outside stack"). classification idiom. */
    cl_task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    if (!cl_task) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR task alloc\n");
        pmsis_exit(-6);
    }
    memset(cl_task, 0, sizeof(struct pi_cluster_task));
    cl_task->entry = &run_cnn;
    cl_task->stack_size = STACK_SIZE;
    cl_task->slave_stack_size = SLAVE_STACK_SIZE;
    cl_task->arg = NULL;

#if BENCH_SMOKE == 5
    pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 5 dispatch ok rc=%d\n", cnn_err);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    for (int i = 0; i < BENCH_WARMUP; i++) {
        pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
    }

    /* Forever: N timed runs -> mean us/inference; AT_GraphPerf cluster cycles. */
    while (1) {
        unsigned int t0 = pi_time_get_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
            if (cnn_err) {
                cpxPrintToConsole(LOG_TO_CRTP, "ERR CNN rc=%d\n", cnn_err);
                pmsis_exit(-5);
            }
        }
        unsigned int t1 = pi_time_get_us();

        unsigned int n_nodes = sizeof(AT_GraphPerf) / sizeof(unsigned int);
        unsigned int sum_nodes = 0;
        for (unsigned int i = 0; i < n_nodes; i++) sum_nodes += AT_GraphPerf[i];
        unsigned int at_total = AT_GraphPerf_CNN_Total;
        unsigned int us_per = (t1 - t0) / BENCH_ITERS;

        cpxPrintToConsole(LOG_TO_CRTP,
                          "BENCH model=%s res=%d cyc=%u nodes=%u clk_us=%u n=%d fcl=%d\n",
                          BENCH_STR(AT_MODEL_PREFIX), AT_MODEL_RES, at_total,
                          sum_nodes, us_per, BENCH_ITERS, FREQ_CL);
    }
}

int main(void)
{
    pi_bsp_init();
    return pmsis_kickoff((void *)net_bench);
}
```

- [ ] **Step 2: Verify the AutoTiler core matches the sim probe**

Run:
```bash
diff <(grep -oE 'gap_cl_starttimer|gap_cl_resethwtimer|CNN_L3_MEMORY|AT_GraphPerf_CNN_Total|ALIGN_UP|CNN_RUN\(\(signed char' mcu/probes/cyc/net_cyc.c | sort -u) \
     <(grep -oE 'gap_cl_starttimer|gap_cl_resethwtimer|CNN_L3_MEMORY|AT_GraphPerf_CNN_Total|ALIGN_UP|CNN_RUN\(\(signed char' mcu/board/net_bench/net_bench.c | sort -u)
```
Expected: empty diff (the transplanted core uses the same symbols). The 7-arg `CNN_RUN((signed char *)In_L3, ...)` with 6 outputs must be present.

- [ ] **Step 3: Verify the board shell matches the classification example's API**

Run:
```bash
for sym in cpxInit cpxEnableFunction cpxPrintToConsole pi_cluster_send_task_to_cl pi_bsp_init pmsis_kickoff pi_freq_set; do
  grep -q "$sym" mcu/board/net_bench/net_bench.c \
    && grep -q "$sym" ~/aideck-gap8-examples/examples/ai/classification/classification.c \
    && echo "OK  $sym" || echo "MISSING  $sym"; done
```
Expected: every line prints `OK` (the firmware only uses CPX/PMSIS symbols the shipped board example also uses).

- [ ] **Step 4: Commit**

```bash
git add mcu/board/net_bench/net_bench.c
git commit -m "mcu/board: net_bench firmware (classification shell + net_cyc core + timing loop)"
```

---

### Task 2: `Makefile` + `nntool_script.tmpl` — model-agnostic build integration

**Files:**
- Create: `mcu/board/net_bench/Makefile`
- Create: `mcu/board/net_bench/nntool_script.tmpl`

**Interfaces:**
- Consumes: a `model_config.mk` (generated by Task 3) defining `MODEL_PREFIX`, `TRAINED_MODEL`, `MODEL_L2_MEMORY`, `CLUSTER_STACK_SIZE`, `ONNX`, and the `AT_*_BYTES`/`AT_MODEL_RES` cflags; plus `NNTOOL_SCRIPT` pointing at the rendered script.
- Produces: `BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img` (the flashable image Task 5 consumes).

- [ ] **Step 1: Write the Makefile (forked from classification's, model vars externalised)**

Write `mcu/board/net_bench/Makefile`:

```makefile
# mcu/board/net_bench/Makefile
# Forked from aideck-gap8-examples/examples/ai/classification/Makefile.
# Model-agnostic: everything model-specific is include'd from model_config.mk,
# which mcu/board/build_bench.sh generates per model. Build target GAP8_V2 (board).
ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

# --- model-specific config (generated by build_bench.sh) ---
include model_config.mk

io=uart
PMSIS_OS = freertos
QUANT_BITS=8
BUILD_DIR=BUILD
MODEL_SQ8=1
MODEL_SUFFIX = _SQ8BIT
APP_CFLAGS += -DMODEL_QUANTIZED

# Our nntool script (rendered by build_bench.sh) quantises from ONNX via aquant,
# so no pre-quantized -q flag and no MODEL_PREQUANTIZED branch.
NNTOOL_SCRIPT ?= model/nntool_script
NNTOOL_EXTRA_FLAGS =

include model_decl.mk

CLUSTER_STACK_SIZE ?= 8192
CLUSTER_SLAVE_STACK_SIZE ?= 1024
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY ?= 84000
MODEL_L3_MEMORY=8000000
FREQ_CL ?= 175
FREQ_FC ?= 250

CPX_TXQ_SIZE=5
CPX_RXQ_SIZE=5
MODEL_L3_EXEC=hram
MODEL_L3_CONST=hflash

pulpChip = GAP
PULP_APP = net_bench
USE_PMSIS_BSP=1

APP = net_bench
APP_SRCS += net_bench.c ../../../lib/cpx/src/com.c ../../../lib/cpx/src/cpx.c $(MODEL_GEN_C) $(CNN_LIB)

APP_CFLAGS += -g -Os -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(realpath $(MODEL_BUILD))
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DAT_MODEL_KERNELS_H='"$(MODEL_PREFIX)Kernels.h"'
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DconfigUSE_TIMERS=1 -DINCLUDE_xTimerPendFunctionCall=1 -DFS_PARTITIONTABLE_OFFSET=0x40000
APP_CFLAGS += -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DTXQ_SIZE=$(CPX_TXQ_SIZE) -DRXQ_SIZE=$(CPX_RXQ_SIZE)
# AT_*_BYTES and AT_MODEL_RES come in via model_config.mk's BENCH_MODEL_CFLAGS.
APP_CFLAGS += $(BENCH_MODEL_CFLAGS)
APP_INC = ../../../lib/cpx/inc

READFS_FILES=$(abspath $(MODEL_TENSORS))

all:: model
clean:: clean_model

include model_rules.mk
RUNNER_CONFIG = $(CURDIR)/config.ini
include $(RULES_DIR)/pmsis_rules.mk
```

- [ ] **Step 2: Write the nntool script template**

Write `mcu/board/net_bench/nntool_script.tmpl` (build_bench.sh substitutes `@RES@`; the raw-head IO→HyperRAM + cycle-monitor recipe is the sim probe's, verbatim, because the .text tax is identical on the board):

```
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
aquant aq_sample/*.jpg -H @RES@ -W @RES@ -T
save_state
```

- [ ] **Step 3: Verify the Makefile keeps the classification build contract**

Run:
```bash
for v in 'include model_decl.mk' 'include model_rules.mk' 'io=uart' 'PMSIS_OS = freertos' 'READFS_FILES' 'MODEL_GEN_C' '-DPERF' 'ONNX'; do
  grep -q "$v" mcu/board/net_bench/Makefile && echo "OK  $v" || echo "MISSING  $v"; done
grep -q 'ONNX' mcu/board/net_bench/Makefile || echo "NOTE: ONNX=1 is set in generated model_config.mk (Task 3), not here"
```
Expected: `model_decl.mk`, `model_rules.mk`, `io=uart`, `PMSIS_OS = freertos`, `READFS_FILES`, `MODEL_GEN_C`, `-DPERF` all `OK`.

- [ ] **Step 4: Commit**

```bash
git add mcu/board/net_bench/Makefile mcu/board/net_bench/nntool_script.tmpl
git commit -m "mcu/board: model-agnostic Makefile + nntool script template for net_bench"
```

---

### Task 3: `build_bench.sh` — assemble a self-contained example dir + generate per-model config

**Files:**
- Create: `mcu/board/build_bench.sh`

**Interfaces:**
- Consumes: `net_bench.c`, `Makefile`, `nntool_script.tmpl` (Tasks 1–2); the model ONNX; `.venv-nas` (onnx, to read IO byte counts); the calib set `data/mcu/probes/aq_sample/`.
- Produces: `~/aideck-gap8-examples/examples/ai/net-bench-<model>/` containing `net_bench.c`, `Makefile`, generated `model_config.mk` + `model/nntool_script`, `<model>.onnx`, `aq_sample/`, `config.ini`; and it prints the exact `make` command to run in the docker.

- [ ] **Step 1: Write the script**

Write `mcu/board/build_bench.sh`:

```bash
#!/usr/bin/env bash
# Assemble a self-contained aideck-gap8-examples app dir for one model, so it can
# be built with the SAME docker + `make` flow that built the wifi-img-streamer.
# Reads the model's 1-in/6-out int8 byte counts from its ONNX (feeding AT_*_BYTES;
# a wrong count would let the CNN write out of bounds). No `set -e`: a fit/codegen
# failure is a RESULT to record (LUT/Stage-0 discipline), not a script error.
#
#   mcu/board/build_bench.sh cand_a5fddcc354bd 192
#   mcu/board/build_bench.sh yolo11n_pose_160_raw 160 160000 8192   # L2, stack overrides
set -uo pipefail

MODEL="${1:?usage: build_bench.sh <model> <res> [L2] [stack]}"
RES="${2:?need res}"
L2="${3:-84000}"
STACK="${4:-8192}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
EX_ROOT="${AIDECK_EXAMPLES:-$HOME/aideck-gap8-examples}"
SRC="${REPO}/mcu/board/net_bench"
ONNX="${REPO}/models/res${RES}/${MODEL}.onnx"
CALIB="${REPO}/data/mcu/probes/aq_sample"
NAS_PY="${REPO}/.venv-nas/bin/python"
APPDIR="${EX_ROOT}/examples/ai/net-bench-${MODEL}"

[[ -f "${ONNX}" ]] || { echo "NO ONNX: ${ONNX}"; exit 2; }
[[ -d "${EX_ROOT}/examples/ai/classification" ]] || { echo "NO aideck examples at ${EX_ROOT}"; exit 2; }
compgen -G "${CALIB}/*.jpg" >/dev/null || { echo "NO calib jpgs at ${CALIB}"; exit 2; }

# 1-input + 6-output int8 byte counts from the ONNX (int8 -> numel == bytes).
read -r IN O1 O2 O3 O4 O5 O6 < <("${NAS_PY}" - "${ONNX}" <<'PY'
import sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
def numel(t):
    n = 1
    for d in t.type.tensor_type.shape.dim:
        n *= (d.dim_value or 0)
    return n
vals = [numel(m.graph.input[0])] + [numel(o) for o in m.graph.output]
assert len(vals) == 7, f"expected 1 input + 6 outputs, got {len(vals)-1} outputs"
print(*vals)
PY
) || { echo "IO_READ_FAIL ${MODEL} (need .venv-nas with onnx)"; exit 3; }
echo "IO in=${IN} out=$((O1+O2+O3+O4+O5+O6)) B  (6 outputs: ${O1} ${O2} ${O3} ${O4} ${O5} ${O6})"

# Assemble the self-contained app dir.
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/model" "${APPDIR}/aq_sample"
cp "${SRC}/net_bench.c" "${SRC}/Makefile" "${APPDIR}/"
cp "${EX_ROOT}/examples/ai/classification/model_decl.mk" \
   "${EX_ROOT}/examples/ai/classification/model_rules.mk" \
   "${EX_ROOT}/examples/ai/classification/config.ini" "${APPDIR}/"
cp "${ONNX}" "${APPDIR}/${MODEL}.onnx"
cp "${CALIB}"/*.jpg "${APPDIR}/aq_sample/"
sed "s/@RES@/${RES}/g" "${SRC}/nntool_script.tmpl" > "${APPDIR}/model/nntool_script"

# Per-model make config the Makefile include's.
cat > "${APPDIR}/model_config.mk" <<EOF
# generated by mcu/board/build_bench.sh -- do not edit
ONNX = 1
MODEL_PREFIX = ${MODEL}
TRAINED_MODEL = \$(CURDIR)/${MODEL}.onnx
NNTOOL_SCRIPT = model/nntool_script
MODEL_L2_MEMORY = ${L2}
CLUSTER_STACK_SIZE = ${STACK}
BENCH_MODEL_CFLAGS = -DAT_INPUT_BYTES=${IN} \\
  -DAT_OUT1_BYTES=${O1} -DAT_OUT2_BYTES=${O2} -DAT_OUT3_BYTES=${O3} \\
  -DAT_OUT4_BYTES=${O4} -DAT_OUT5_BYTES=${O5} -DAT_OUT6_BYTES=${O6} \\
  -DAT_MODEL_RES=${RES}
EOF

echo
echo "ASSEMBLED  ${APPDIR}"
echo "Build it in the Bitcraze toolchain docker (the one that built the streamer):"
echo "    cd examples/ai/net-bench-${MODEL} && make clean model all image"
echo "Flashable image will be at:"
echo "    ${APPDIR}/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img"
```

- [ ] **Step 2: Make it executable and run it against the real winner ONNX**

Run:
```bash
chmod +x mcu/board/build_bench.sh
mcu/board/build_bench.sh cand_a5fddcc354bd 192
```
Expected: prints `IO in=<n> out=<n> B (6 outputs: ...)` with **seven** integers, then `ASSEMBLED …`. (Needs `.venv-nas` present with `onnx`. If `.venv-nas` is absent, that is a documented environment gap — note it and move to Step 3's structural check instead.)

- [ ] **Step 3: Verify the assembled dir is self-contained and the config is correct**

Run:
```bash
APP=~/aideck-gap8-examples/examples/ai/net-bench-cand_a5fddcc354bd
ls "$APP" "$APP/model" | sort
echo "--- model_config.mk ---"; cat "$APP/model_config.mk"
echo "--- nntool_script (res substituted) ---"; grep aquant "$APP/model/nntool_script"
```
Expected: dir contains `net_bench.c Makefile model_config.mk model_decl.mk model_rules.mk config.ini cand_a5fddcc354bd.onnx aq_sample/ model/`; `model_config.mk` sets `ONNX = 1`, `MODEL_PREFIX = cand_a5fddcc354bd`, `AT_INPUT_BYTES=110592` (= 192·192·3) and six `AT_OUT*_BYTES`; the `aquant` line reads `-H 192 -W 192`.

- [ ] **Step 4: Commit**

```bash
git add mcu/board/build_bench.sh
git commit -m "mcu/board: build_bench.sh assembles a self-contained per-model aideck app"
```

---

### Task 4: `bench_receiver.py` — host CPX console reader + BENCH-line parser (tested)

**Files:**
- Create: `mcu/board/bench_receiver.py`
- Create: `tests/test_bench_receiver.py`

**Interfaces:**
- Consumes: the `BENCH …` console line produced by Task 1's firmware; `cflib` (runtime only, for the live console).
- Produces: `parse_bench_line(line: str) -> dict | None` — returns `{model, res, cyc, nodes, clk_us, n, fcl, ms, fps}` (with `ms = clk_us/1000.0`, `fps = 1e6/clk_us`) or `None` for a non-BENCH line. This pure function is the unit-tested surface; the cflib live loop is a thin shell around it.

- [ ] **Step 1: Write the failing test**

Write `tests/test_bench_receiver.py`:

```python
import importlib.util
import pathlib

import pytest

_MOD = pathlib.Path(__file__).resolve().parents[1] / "mcu" / "board" / "bench_receiver.py"
_spec = importlib.util.spec_from_file_location("bench_receiver", _MOD)
bench_receiver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_receiver)
parse_bench_line = bench_receiver.parse_bench_line


def test_parses_bench_line_and_derives_ms_fps():
    line = "BENCH model=cand_a5fddcc354bd res=192 cyc=58394932 nodes=58390000 clk_us=333000 n=20 fcl=175"
    rec = parse_bench_line(line)
    assert rec is not None
    assert rec["model"] == "cand_a5fddcc354bd"
    assert rec["res"] == 192
    assert rec["cyc"] == 58394932
    assert rec["clk_us"] == 333000
    assert rec["n"] == 20
    assert rec["fcl"] == 175
    assert rec["ms"] == pytest.approx(333.0)
    assert rec["fps"] == pytest.approx(3.003, abs=1e-3)


def test_non_bench_line_returns_none():
    assert parse_bench_line("*** net_bench cand_x res=192 iters=20 ***") is None
    assert parse_bench_line("SMOKE 3 construct ok arena_base=5248064") is None
    assert parse_bench_line("") is None


def test_zero_clk_us_does_not_crash():
    rec = parse_bench_line("BENCH model=m res=160 cyc=1 nodes=1 clk_us=0 n=20 fcl=175")
    assert rec is not None
    assert rec["fps"] is None and rec["ms"] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH= .venv/bin/python -m pytest tests/test_bench_receiver.py -q`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (bench_receiver.py / parse_bench_line does not exist yet).

- [ ] **Step 3: Write the implementation**

Write `mcu/board/bench_receiver.py`:

```python
#!/usr/bin/env python3
"""Read the GAP8 net_bench CPX console over radio and print measured cyc + FPS.

The firmware emits integer-only lines (GAP8 newlib-nano %f is unreliable):
    BENCH model=<s> res=<i> cyc=<i> nodes=<i> clk_us=<i> n=<i> fcl=<i>
This host side derives ms = clk_us/1000 and fps = 1e6/clk_us, appends every row
to data/mcu/board/<model>.txt, and also works with `cfclient` (Console tab) as a
zero-code alternative. cyc is RANKING-ONLY vs the sim column; ms/fps is the honest
absolute. Transport is CPX console over CRTP (radio); the WiFi console is the same
stream if preferred.
"""
import argparse
import pathlib
import re

_BENCH = re.compile(
    r"BENCH\s+model=(?P<model>\S+)\s+res=(?P<res>\d+)\s+cyc=(?P<cyc>\d+)\s+"
    r"nodes=(?P<nodes>\d+)\s+clk_us=(?P<clk_us>\d+)\s+n=(?P<n>\d+)\s+fcl=(?P<fcl>\d+)"
)


def parse_bench_line(line):
    """Parse one firmware console line -> dict with derived ms/fps, or None.

    fps is None when clk_us == 0 (a not-yet-timed / degenerate line).
    """
    m = _BENCH.search(line or "")
    if not m:
        return None
    rec = {k: int(v) for k, v in m.groupdict().items() if k != "model"}
    rec["model"] = m.group("model")
    rec["ms"] = rec["clk_us"] / 1000.0
    rec["fps"] = (1_000_000.0 / rec["clk_us"]) if rec["clk_us"] else None
    return rec


def _fmt(rec):
    fps = f"{rec['fps']:.3f}" if rec["fps"] is not None else "n/a"
    return (f"{rec['model']:<22} res={rec['res']}  cyc={rec['cyc']:>12,}  "
            f"ms={rec['ms']:>8.2f}  fps={fps:>7}  (n={rec['n']}, fcl={rec['fcl']} MHz)")


def _run_live(uri, out_dir):  # pragma: no cover - needs radio hardware
    import cflib.crtp
    from cflib.cpx import CPX  # transport helper; console packets arrive as text
    from cflib.crtp.radiodriver import RadioManager  # noqa: F401

    cflib.crtp.init_drivers()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"listening on {uri} (Ctrl-C to stop) ...")
    # cflib exposes the CPX console over CRTP port 0 as UTF-8 text; buffer to lines.
    buf = ""
    with CPX(uri) as cpx:
        while True:
            buf += cpx.receive_console()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                rec = parse_bench_line(line)
                if rec is None:
                    print(f"  {line}")
                    continue
                print(_fmt(rec))
                (out_dir / f"{rec['model']}.txt").open("a").write(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description="GAP8 net_bench CPX console reader")
    ap.add_argument("--uri", default="radio://0/80/2M", help="Crazyradio URI")
    ap.add_argument("--out", default="data/mcu/board", help="dir for per-model logs")
    args = ap.parse_args()
    _run_live(args.uri, pathlib.Path(args.out))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH= .venv/bin/python -m pytest tests/test_bench_receiver.py -q`
Expected: PASS (3 passed). The pure parser has no cflib dependency, so it passes without radio hardware.

- [ ] **Step 5: Commit**

```bash
git add mcu/board/bench_receiver.py tests/test_bench_receiver.py
git commit -m "mcu/board: bench_receiver.py CPX console reader + tested BENCH-line parser"
```

---

### Task 5: `flash_bench.sh` + `README.md` + results scaffold

**Files:**
- Create: `mcu/board/flash_bench.sh`
- Create: `mcu/board/README.md`

**Interfaces:**
- Consumes: the `target.board.devices.flash.img` produced by Task 2/3; `cfloader` (via `uvx --from cfclient cfloader` or an installed cfclient).
- Produces: a flashed GAP8; documentation of the full per-model loop.

- [ ] **Step 1: Write the flash wrapper**

Write `mcu/board/flash_bench.sh`:

```bash
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
```

- [ ] **Step 2: Write the workflow README (with the 4-model loop + a results table to fill)**

Write `mcu/board/README.md`:

```markdown
# mcu/board — on-silicon cycle + FPS bench (GAP8 AI-deck)

Confirms the Phase-10 GVSOC **sim** cycle numbers on real GAP8 silicon, for the
four MCU finalists. Design: `docs/superpowers/specs/2026-07-22-mcu-board-net-bench-design.md`.
Cycles are **ranking-only** (LUT/Stage-0 discipline); the wall-clock ms/FPS is the
honest absolute — expect it HIGHER than sim (`BOARD.md`: real HyperRAM latency).

## Prereqs
- The Bitcraze toolchain docker that built the wifi-img-streamer (`GAP8_V2`), and
  a working `~/aideck-gap8-examples`. **Not** the `tfm-gap8` sim image.
- `.venv-nas` (onnx) on the host — build_bench.sh reads the model's IO shapes.
- A flashed AI-deck (ESP + GAP8 reachable over radio) + Crazyradio. Deck firmware
  flashing is radio-only; see [[crazyflie-is-cf21bl]] for the host-drone caveat.

## The loop (per model)
```bash
# 1) assemble the app dir (host, .venv-nas): <model> <res> [L2] [stack]
mcu/board/build_bench.sh cand_a5fddcc354bd 192
# 2) build in the Bitcraze docker (the one that built the streamer):
#      cd examples/ai/net-bench-cand_a5fddcc354bd && make clean model all image
# 3) flash over radio:
mcu/board/flash_bench.sh cand_a5fddcc354bd
# 4) read the numbers (either):
python mcu/board/bench_receiver.py            # prints + logs data/mcu/board/<model>.txt
#   or open cfclient -> Console tab -> watch the BENCH lines
```
Then repeat for the other three:
```
cand_19efff428be8   192
cand_863c75818953   192
yolo11n_pose_160_raw 160 160000     # baseline: more L2 headroom
```

## Tunables / troubleshooting
- `construct rc=3` (L2 alloc) → lower `MODEL_L2_MEMORY` (4th arg), e.g. 70000.
- Silence after flashing → run a smoke build: add `-DBENCH_SMOKE=3` (stops after
  Construct) via `make ... APP_CFLAGS+=-DBENCH_SMOKE=3`; the last SMOKE line reached
  localises the failure (1 boot · 2 cluster · 3 construct · 4 io · 5 dispatch).
- `%f`-looking garbage → the firmware prints integers only; use the receiver/host
  to derive ms + fps.

## Results (fill from silicon; sim columns from state/winner_mcu/winner.json)
| model | res | sim cyc | sim FPS | **meas cyc** | **meas ms** | **meas FPS** |
|---|---|---|---|---|---|---|
| yolo11n_pose_160_raw | 160 | 59.85 M | 2.92 | | | |
| cand_a5fddcc354bd | 192 | 58.39 M | 3.0 | | | |
| cand_19efff428be8 | 192 | 43.26 M | 4.05 | | | |
| cand_863c75818953 | 192 | 60.44 M | 2.91 | | | |
```

- [ ] **Step 3: Verify the glue is consistent (paths + targets line up across scripts)**

Run:
```bash
chmod +x mcu/board/flash_bench.sh
grep -o 'net-bench-[^/"]*' mcu/board/build_bench.sh mcu/board/flash_bench.sh | sort -u
grep -c 'deck-bcAI:gap8-fw' mcu/board/flash_bench.sh
grep -o 'target.board.devices.flash.img' mcu/board/build_bench.sh mcu/board/flash_bench.sh | sort -u
```
Expected: both scripts reference the same `net-bench-<model>` dir pattern and the same `target.board.devices.flash.img` artifact; `flash_bench.sh` names `deck-bcAI:gap8-fw` once.

- [ ] **Step 4: Commit**

```bash
git add mcu/board/flash_bench.sh mcu/board/README.md
git commit -m "mcu/board: flash_bench.sh + README (per-model loop, tunables, results table)"
```

---

### Task 6: On-hardware DoD (manual — cannot run in this environment)

**Files:** none (produces `data/mcu/board/<model>.txt` + the filled README table on the user's hardware).

**Interfaces:**
- Consumes: everything above + the Bitcraze docker + the AI-deck on radio.
- Produces: a 4-row measured cyc/ms/FPS table; a decision on whether silicon confirms the `a5fddcc < baseline` sim ranking.

This task is the real gate and is executed by the user on hardware; it is documented here so the plan is complete, not because an agent runs it.

- [ ] **Step 1: Build the winner (smoke first to de-risk L2 fit)**

In the Bitcraze docker:
```bash
cd examples/ai/net-bench-cand_a5fddcc354bd
make clean model all image APP_CFLAGS+=-DBENCH_SMOKE=3   # stops after Construct
```
Then `mcu/board/flash_bench.sh cand_a5fddcc354bd` and read the console.
Expected: `SMOKE 3 construct ok arena_base=<n>` — proves the 400 KB graph + FreeRTOS + CPX fit in L2. If instead `ERR construct rc=3`, lower L2: re-run `build_bench.sh cand_a5fddcc354bd 192 70000` and rebuild.

- [ ] **Step 2: Build + flash the full bench for the winner**

```bash
make clean model all image          # no BENCH_SMOKE
```
`flash_bench.sh cand_a5fddcc354bd`, then `python mcu/board/bench_receiver.py`.
Expected: repeating `BENCH model=cand_a5fddcc354bd res=192 cyc=<c> … clk_us=<u> …` lines; `cyc` in the tens-of-millions range and (per `BOARD.md`) `meas ms` > `cyc / 175e6 * 1000` (the HyperRAM tax the sim omits).

- [ ] **Step 3: Repeat for the other three finalists**

`build_bench.sh` → docker `make` → `flash_bench.sh` → receiver, for `cand_19efff428be8` 192, `cand_863c75818953` 192, and `yolo11n_pose_160_raw` 160 160000.

- [ ] **Step 4: Fill the results table and record the finding**

Paste `data/mcu/board/*.txt` into `mcu/board/README.md`'s table and the `models/README.md` MCU section. State plainly whether silicon confirms `a5fddcc` cycles ≤ baseline (the CP 10.3 claim), or corrects it. A documented L2-infeasibility for any graft is itself a valid recorded result.

---

## Self-Review

**1. Spec coverage** (checked each spec section against a task):
- §2 fork classification + transplant net_cyc core → Task 1 (firmware), Task 2 (Makefile). ✓
- §3.1 firmware (HyperRAM IO, timing, AT_GraphPerf, BENCH line, SMOKE stages) → Task 1. ✓
- §3.2 forked Makefile (per-model tunables) → Task 2 + Task 3 (model_config.mk). ✓
- §3.3 nntool script (IO→HyperRAM, cycle monitor, aquant) → Task 2 (`nntool_script.tmpl`). ✓
- §3.4 build_bench.sh (IO bytes, assemble, single-pass — AT_L3_ARENA dropped) → Task 3. ✓
- §3.5 flash_bench.sh → Task 5. ✓
- §3.6 bench_receiver.py → Task 4 (+ tests). ✓
- §4 workflow + §5 risks (L2 fit, SMOKE staging, freq bookkeeping) → Task 5 README + Task 6 DoD. ✓
- §6 out of scope (camera/NMS/accuracy/power) → not implemented, correct. ✓
- The four finalists, exact files/res → Global Constraints + Task 6 loop. ✓

**2. Placeholder scan:** No TBD/TODO/"handle errors". The README results table is intentionally blank (filled from silicon in Task 6) — a data artifact, not a code placeholder. All code steps show complete code.

**3. Type consistency:** `parse_bench_line(line)->dict|None` fields `{model,res,cyc,nodes,clk_us,n,fcl,ms,fps}` match between the firmware's `BENCH …` line (Task 1), the regex (Task 4 impl), and the test (Task 4). The `AT_*_BYTES` names match between `net_bench.c` (`OutBytes[]`, Task 1), the Makefile `BENCH_MODEL_CFLAGS` (Task 2), and `build_bench.sh`'s generated `model_config.mk` (Task 3). `AT_MODEL_KERNELS_H` is `-D`'d in the Makefile (Task 2) and `#include`d in the firmware (Task 1). The example-dir name `net-bench-<model>` and the `target.board.devices.flash.img` path match between `build_bench.sh` (Task 3) and `flash_bench.sh` (Task 5). `MODEL_L2_MEMORY` default 84000/160000 consistent across Makefile, build_bench.sh, README.
```
