# On-silicon cycle + FPS bench for the 4 MCU finalists

**Date:** 2026-07-22
**Status:** approved design → implementation
**Owner:** Phase 10 (MCU / GAP8 AI-deck), extends `mcu/BOARD.md`

---

## 1. Goal

Convert the GVSOC **sim** cycle numbers behind the CP 10.3 claim into **measured
silicon** numbers, on the real GAP8 of the Crazyflie AI-deck, for the four MCU
finalists. For each model, flash a firmware that runs its int8 AutoTiler graph
on-device and reports back, over the radio/WiFi console, two numbers:

1. **`AT_GraphPerf` cluster cycles** — the *same* counter GVSOC reported, now
   real silicon. Directly confirms/corrects the 58.39 M sim number and the
   `a5fddcc < baseline` ranking.
2. **wall-clock FPS** — `pi_time_get_us()` around the cluster dispatch, so it
   captures the real HyperRAM-DMA stalls the sim omits.

Cycles stay **ranking-only** (LUT/Stage-0 discipline); the wall-clock ms is the
new, honest absolute. This is the silicon step `BOARD.md` step 3 describes,
generalised from the a5fddcc-vs-baseline pair to all four finalists.

### The four finalists (from `models/README.md` MCU leg + `state/winner_mcu/winner.json`)

| # | Model | ONNX | res | sim cyc | sim FPS |
|---|---|---|---|---|---|
| 1 | yolo11n-pose (baseline) | `models/res160/yolo11n_pose_160_raw.onnx` | 160 | 59.85 M | 2.92 |
| 2 | a5fddcc (CP 10.3 winner) | `models/res192/cand_a5fddcc354bd.onnx` | 192 | 58.39 M | 3.0 |
| 3 | 19efff (speed point) | `models/res192/cand_19efff428be8.onnx` | 192 | 43.26 M | 4.05 |
| 4 | 863c (accuracy-max) | `models/res192/cand_863c75818953.onnx` | 192 | 60.44 M | 2.91 |

All four are raw-head exports: **1 int8 input, 6 int8 outputs** (the signature the
existing `mcu/probes/cyc/net_cyc.c` harness already consumes). DFL-decode / anchor
concat / NMS are excluded from the graph on both families (run in C on the FC),
so they cancel from any ratio and are out of scope here.

---

## 2. Architecture

**Fork Bitcraze's `examples/ai/classification`, transplant the `net_cyc.c` core.**
The two existing pieces compose exactly:

- `~/aideck-gap8-examples/examples/ai/classification/` is a **working board NN
  app**: `io=uart`, `PMSIS_OS=freertos`, `include model_rules.mk` (the
  nntool→AutoTiler→board codegen driven by `TRAINED_MODEL` + `NNTOOL_SCRIPT`),
  `-DPERF` (enables `AT_GraphPerf`), `cpxInit()` + `cpxPrintToConsole(LOG_TO_CRTP,
  …)` (radio/WiFi console), and the cluster boilerplate (`pi_cluster_open`,
  `task->entry`, `pi_cluster_send_task_to_cl`). It builds and flashes on this deck.
- `mcu/probes/cyc/net_cyc.c` is the **graph-run + cycle-read core**: it constructs
  an AutoTiler CNN, places its 1-in/6-out IO in HyperRAM (packed below the arena,
  derived from `CNN_L3_MEMORY`), runs it, and reads `AT_GraphPerf`. It targets
  GVSOC/pulp-os, not FreeRTOS/CPX.

The new firmware `net_bench.c` = **classification's shell** (freq/uart/cpx/cluster/
FreeRTOS setup + Makefile + CPX console) with **net_cyc's core** dropped into the
body (HyperRAM IO placement + `AT_GraphPerf` read), plus a **timing loop**. The
camera capture and the 2-class decode of `classification.c` are removed; there is
no camera — a fixed, uninitialised HyperRAM input is correct because int8
AutoTiler cycles are data-independent (fixed tile loops, no sparsity/branch path),
exactly as `net_cyc.c` argues.

**Why not a from-scratch CPX app or a two-stage codegen?** Rejected. The
classification example already solves the one genuinely hard integration — wiring
`model_rules.mk` (nntool + AutoTiler + READFS tensor image + `MODEL_GEN_C` into
`APP_SRCS`) into a FreeRTOS/CPX board build. Re-deriving that is pure risk for no
gain.

### Memory forces one `.img` per model

The generated graph `.text` measured **330–405 KB of GAP8's 512 KB L2** (per
`mcu/README.md` CP 10.1). Two graphs cannot coexist in one binary, so the design
is **one flashable image per model, flashed separately** — not one firmware that
loops over four. "Every model" is handled by a *loop over the four finalists in
the build script*, not four source copies. The firmware is a single parameterised
`net_bench.c`; the model arrives as `-DAT_MODEL_PREFIX=<name>` + `TRAINED_MODEL`
+ `NNTOOL_SCRIPT`, identically to how `net_cyc.c` is model-agnostic today.

---

## 3. Components

Source of truth lives in **TFM_NAS** (repo discipline — mirrors `mcu/probes/`),
and is copied into the Bitcraze examples tree at build time.

```
mcu/board/net_bench/net_bench.c        # firmware: classification shell + net_cyc core + timing loop
mcu/board/net_bench/Makefile           # forked classification Makefile, model via make vars
mcu/board/net_bench/nntool_script.tmpl # nntool: raw-head 6-out → HyperRAM, cycle monitor on
mcu/board/build_bench.sh  <model> <res> [L2] [stack]   # copy → docker codegen+build → .img
mcu/board/flash_bench.sh  <model>                       # cfloader radio flash (deck-bcAI:gap8-fw)
mcu/board/bench_receiver.py                             # cflib console reader → prints/logs BENCH line
mcu/board/README.md                                     # the per-model loop, tunables, troubleshooting
```

### 3.1 `net_bench.c` — the firmware

- **Interface (compile-time `-D`, from the Makefile):** `AT_MODEL_PREFIX` (bare
  token, pasted onto AutoTiler symbols via the `CAT`/`_CAT` trick from
  `net_cyc.c`); `AT_INPUT_BYTES`, `AT_OUT1_BYTES..AT_OUT6_BYTES` (read from the
  ONNX at build time, as `cyc_probe.sh` already does); `AT_L3_ARENA` (the
  `AT_HYPERRAM_ALLOC` size grepped from `<model>Kernels.c`); `AT_MODEL_KERNELS_H`;
  `STACK_SIZE`/`SLAVE_STACK_SIZE`; `BENCH_ITERS` (default 50); `BENCH_WARMUP`
  (default 2); `AT_MODEL_RES` (for the report line only).
- **Boot / shell (from classification.c):** `pi_bsp_init` → `pmsis_kickoff`;
  set FC + **CL frequency explicitly to 175 MHz** (the sim's basis — the AI-deck
  1.1 / GAP8 V3 cluster ceiling behind every repo cycle number) and report it, so
  board cycles↔ms and the board-vs-sim comparison share one clock; open UART;
  `cpxInit()` + `cpxEnableFunction(CPX_F_WIFI_CTRL)`.
- **Graph + IO (from net_cyc.c):** define `__PREFIX(_L3_Flash)=0`; `pi_cluster_open`;
  `__PREFIX(CNN_Construct)()`; place input + 6 outputs in HyperRAM packed below
  `CNN_L3_MEMORY` with the align-up + bound-check from `net_cyc.c`; set the master
  stack explicitly (`task.stack_size`) — the grafts overran the 2 KB default in sim.
- **`RunNetwork` (cluster entry):** `gap_cl_starttimer()` + `gap_cl_resethwtimer()`
  then `__PREFIX(CNN)(In_L3, Out_L3[0..5])` — the **7-arg raw-head** call, not
  classification's 2-arg call.
- **Timing loop:** `BENCH_WARMUP` untimed runs (first pages weights
  HyperFlash→HyperRAM); then `BENCH_ITERS` runs bracketed by `pi_time_get_us()` on
  the FC → mean ms/inference → FPS. After the loop, sum `AT_GraphPerf` locally and
  read `AT_GraphPerf_CNN_Total` (the whole-graph counter). **Do not** print 200
  per-node lines over the radio console — sum on-chip; optionally print the top-3
  hottest nodes only.
- **Report (loops forever so a stable line can be eyeballed):**
  `BENCH model=<prefix> res=<r> cyc=<AT_total> clk_ms=<mean> fps=<f> n=<N> fcl=<CL_MHz>`
  via `cpxPrintToConsole(LOG_TO_CRTP, …)`, then a `BENCH_DONE` marker.
- **Staged bring-up (ported from `net_cyc.c`'s `CYC_SMOKE`):** `BENCH_SMOKE=1..5`
  stops after boot / cluster-open / construct / IO-placed / one dispatch, each
  printing a distinct CPX line. Because a GAP8 crash reports as *silence* (buffered
  stdout lost on abort), staging is the only way to localise an L2-fit failure.

### 3.2 `Makefile` — forked from `classification/Makefile`

Same skeleton (`io=uart`, `PMSIS_OS=freertos`, `include model_decl.mk` /
`model_rules.mk`, `APP_SRCS += … cpx.c $(MODEL_GEN_C) …`, `-DPERF`,
`READFS_FILES=$(MODEL_TENSORS)`), with these parameterised:
`MODEL_PREFIX`, `TRAINED_MODEL` (our ONNX), `NNTOOL_SCRIPT` (our template),
`MODEL_L2_MEMORY` (per-model; **the primary tunable** — grafts start at 84000,
baseline at 160000, lower it if `Construct` returns rc=3), `CLUSTER_STACK_SIZE`
(start 8192 — the grafts need it), the `AT_*_BYTES` / `AT_L3_ARENA` `-D`s, and
`APP_SRCS` pointing at `net_bench.c`.

### 3.3 `nntool_script.tmpl` — quantise + codegen recipe

Mirrors the `cyc_probe.sh` script (the board needs the *same* IO→HyperRAM plan
because the .text tax is identical): `adjust`; `fusions -a scaled_match_group
expression_matcher` (one call — a second `adjust_order` over a detection head
raises "axes don't match array"); `set default_{input,output}_{home,exec}_location
AT_MEM_L3_HRAM`; `set l3_ram_device AT_MEM_L3_HRAM`; `set l3_flash_device
AT_MEM_L3_HFLASH`; `graph_produce_node_names`/`operinfos`/`monitor_cycles true`;
`aquant data/mcu/probes/aq_sample/*.jpg -H <res> -W <res> -T` (the sim probe's
existing calib set — `aquant` resizes via `-H/-W`, so only `<res>` is substituted
per model); `save_state`.

### 3.4 `build_bench.sh <model> <res> [L2] [stack]`

1. Resolve the ONNX (`models/res<res>/<model>.onnx`) and read its 7 IO byte counts
   (reuse the `io_bytes` python from `cyc_probe.sh`, `.venv-nas`).
2. Render `nntool_script` (res + calib path substituted) and the `<model>.h`
   shim (declares `<model>_L3_Flash`, as AutoTiler's `Kernels.h` `#include`s it).
3. Copy `net_bench.c` + the rendered Makefile into a throwaway example dir under
   the Bitcraze tree (e.g. `examples/ai/net-bench-<model>/`).
4. Codegen pass, then read `AT_L3_ARENA` from `<model>Kernels.c`.
5. Full board build with the `AT_*_BYTES` / `AT_L3_ARENA` `-D`s.
6. Emit the flashable `BUILD/GAP8_V2/…/target.board.devices.flash.img` path.

Runs inside the **Bitcraze toolchain docker** (`bitcraze/aideck-toolchain` — the
image that already built the wifi-img-streamer for `GAP8_V2`). **Not** the
`tfm-gap8` sim image: `BOARD.md` is explicit that it is GVSOC-only and cannot
produce a board/FreeRTOS binary. No `set -e`: a codegen/fit failure is a
**result to record**, same as the sim probes.

### 3.5 `flash_bench.sh <model>`

`cfloader flash <img> deck-bcAI:gap8-fw -w radio://…` — the exact path that already
flashed the 9 fps streamer. Dongle within a few cm of the deck; deck powered.

### 3.6 `bench_receiver.py`

A ~30-line `cflib` client that opens the CRTP link, subscribes to the CPX console
(the same channel cfclient's Console tab shows), and prints/appends every `BENCH …`
line to `data/mcu/board/<model>.txt`. Works over radio; the WiFi console path is
the same CPX stream if the user prefers. cfclient's Console tab is the zero-code
alternative.

---

## 4. Workflow (per model, ×4)

```
# in the Bitcraze toolchain docker (the one that built the streamer):
mcu/board/build_bench.sh cand_a5fddcc354bd 192      # → target.board.devices.flash.img
# on the host:
mcu/board/flash_bench.sh cand_a5fddcc354bd          # cfloader radio → deck-bcAI:gap8-fw
python mcu/board/bench_receiver.py                  # or cfclient → Console tab
#   → BENCH model=cand_a5fddcc354bd res=192 cyc=... clk_ms=... fps=... n=50 fcl=175
# record, repeat for the other three.
```

Deliverable: a 4-row table of **measured** GAP8 cycles + wall-clock FPS, alongside
the sim column, in `models/README.md` and `data/mcu/board/`.

---

## 5. Risks & what is verifiable

- **L2 fit (the one real risk).** FreeRTOS + CPX + a 400 KB-`.text` graph in 512 KB
  L2. Mitigation: `MODEL_L2_MEMORY` is the tunable, and the `BENCH_SMOKE` staging
  localises a failure to boot/cluster/construct/IO/dispatch instead of silence. If
  a graft cannot construct at any L2 that leaves room for CPX, that is itself a
  recordable result (documented infeasibility satisfies the goal, same discipline
  as CP 10.1).
- **Cannot build/flash/test here.** All source is writable and self-checkable, but
  building needs the GAP-SDK docker and running needs the deck. Flashing itself is
  already proven on this hardware (the 9 fps stream).
- **Freq bookkeeping.** Cycles are frequency-independent (compare freely to the
  sim's cycle column); the wall-clock ms depends on the real CL freq, so the
  firmware sets CL freq explicitly and reports it in the BENCH line.

## 6. Out of scope

Camera capture; DFL-decode/anchor/NMS on the FC; on-device accuracy (a real
HM01B0-vs-synthetic domain shift); power measurement (a good follow-up — the
deck's whole point is sub-100 mW — but not needed for the cycle/FPS claim);
touching the sim probes (`mcu/probes/`) — this is additive.
