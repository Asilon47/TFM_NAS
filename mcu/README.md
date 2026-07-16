# mcu/ — Phase 10: GAP8 (Crazyflie AI-deck) retarget

CP 10.1 bring-up kit: a reconstructed, fully pinned GAP8 toolchain
(NNTool → AutoTiler → GVSOC) that produces **simulated cycle counts** — the
latency oracle for the MCU leg of the NAS. Sim numbers are **ranking-only**,
same discipline as the Jetson LUT (see procedure.md, Stage 0: latency numbers
never transfer across measurement contexts).

## The toolchain-recovery story (record, 2026-07-15)

GreenWaves Technologies (the GAP8 vendor) is defunct: both
`greenwaves-technologies.com` domains fail DNS, and the official AutoTiler
distribution (registration → emailed personal URL, see upstream
`tools/autotiler_v3/get_tiler.py`) is dead with the company. Everything else
survives as public git repos. The **single closed-source piece** is the
AutoTiler core `LibTile.a` (an x86-64 *host* library that generates tiled GAP8
C code — the chip-side CNN kernels/generators are open in the SDK tree).

Recovery: two independent public forks committed byte-identical copies of
`libtile.4.3.5.a` — exactly the `TILER_VER` gap_sdk master pins:

| source | sha256 |
|---|---|
| `edoardobonura/gap_sdk` (pushed 2024-02) | `541f4978f3e55c6650207b3f8eebcfe0d311acd3320f81eb0a59890ec928c728` |
| `boomer319/gap_sdk` (pushed 2023-12) | same — byte-identical |

`fetch_tiler.sh` downloads from either and verifies the hash; the blob lives
only in `mcu/vendor/` (gitignored — proprietary EULA, mirrored at
`gap_sdk/tools/autotiler_v3/LICENSE`, restricts use to compiling for GAP
targets, which is our exact use). Same never-commit + SHA-pin discipline as
the OFA checkpoint (`supernet/download_ofa.py`).

## Pinned versions

| component | pin |
|---|---|
| base image | `ubuntu:20.04` |
| gap_sdk | `a23026507efe57410b98c5945544cff3150eb996` (upstream master tip, frozen 2024-01-16) |
| GAP RISC-V toolchain | `e6b5fbbfbf57a0ad1092efb280d0417cf3ee7e51` (`gap_riscv_toolchain_ubuntu`, frozen 2024-03) |
| AutoTiler | `libtile.4.3.5.a`, sha256 `541f4978…` (above) |
| board config | `gapuino` (GAP8; the AI-deck runs the same chip — deck-specific config comes with hardware work, out of scope for sim) |

Known-working fallback image (community, recipe source): `cbezaitis/gap:latest`
(Docker Hub, 2026-03-04) — its author verified GVSOC `Test success !` and
AutoTiler MFCC codegen. Use via `GAP8_IMAGE=cbezaitis/gap:latest mcu/run.sh …`.
Our image builds from *upstream* + the doubly-attested tiler instead.

## Bring-up

```bash
mcu/build.sh    # fetch+verify LibTile.a, then docker build -> tfm-gap8:cp10.1
mcu/smoke.sh    # helloworld on GVSOC ("Test success !") + nntool availability
mcu/run.sh      # interactive shell / run a command with the SDK env sourced
```

## Calibration set (int8 PTQ)

```bash
source .venv-nas/bin/activate
python mcu/prep_calib.py    # 64 train images -> gray {224,192,160} PNGs, data/mcu/calib/
```

Draws from the TRAIN split (val stays untouched for accuracy claims);
`data/mcu/calib/manifest.json` records the seeded draw.

## CP 10.1 probes (DoD: reproducible cycle numbers or documented infeasibility)

- **Probe A (NAS family):** `models/res224/graft_noneck_224.onnx` (winner-v1
  arch grafted pose model, opset 17) through NNTool int8 → AutoTiler →
  GVSOC. Fallback if the pose head blocks: backbone-only export.
- **Probe B (baseline):** `models/res224/yolo11n_pose_224.onnx` through the
  identical path. Expected friction: C2PSA attention / DFL head ops — a
  documented infeasibility is itself the baseline result.

### Codegen (`probes/gen_probe.sh`) — DONE

Both families compile to GAP8 kernels + a memory plan. `probes/at/Makefile` is
**app-less on purpose**: `make model` exercises every codegen gate without a
main.c. It answers *"does it compile, and does it fit HyperRAM?"* — nothing more.

### Cycles (`probes/cyc_probe.sh`) — the DoD number, IN PROGRESS

`probes/cyc/` adds the app (`net_cyc.c`) the codegen probe deliberately lacks:
nntool with the cycle monitor on → AutoTiler → link → GVSOC → parse
`AT_GraphPerf` → `data/mcu/cyc/<model>.json`. One harness serves both models —
the graft and the yolo11n twin export an **identical entry signature** (1 int8
input, 6 int8 outputs, matched shapes), so the same object code drives both and
any cycle delta is the backbone alone.

**Two L2 facts the app-less probe could not see** (both measured here, both
resolution-independent, both shaping the whole MCU leg):

1. **The generated graph's `.text` is ~329 KB — 63 % of GAP8's 512 KB L2** (214
   node functions), leaving only ~180 KB of L2 heap. Code size scales with
   **node count**, not input size, so this tax is identical at 160² and 224²:
   the MCU leg favours *shallower* graphs, not merely smaller ones.
2. **The stock `--L2 400000` is unrunnable.** AutoTiler is greedy (it consumes
   whatever budget it is given), and it shares that one L2 with the binary. The
   measured ladder: 400000 tiles but dies at runtime (`construct rc=3`, L2
   alloc); 70000 fails at codegen ("Failed to allocate Buffer … Kernel Bias");
   **160000 tiles and constructs**. All graph IO (150528 B in + 179389 B of
   raw-head outputs) is therefore pushed to HyperRAM, leaving the heap to
   AutoTiler alone.

Status: boot → cluster open → `CNN_Construct` all pass on GVSOC (verified by the
staged `CYC_SMOKE=1|2|3` bisect; arena base reported at 5248064). **The CNN run
itself still aborts** — open blocker, see procedure.md.

Toolchain gotchas encoded here (each cost a debug cycle, none documented upstream):
- AutoTiler emits `#include "<prefix>.h"` into its `Kernels.h` and expects the
  app to supply it (mnist ships a hand-written one); `cyc_probe.sh` generates it.
- **`io=host` is mandatory** — both SDK references that run clean on GVSOC set
  it. Stdout is buffered and lost on SIGABRT, so a crash reports as *silence*;
  that is what the `CYC_SMOKE` staging exists to see through.
- **"Magick: abort due to signal 6" is a red herring** — the board's camera model
  links ImageMagick, whose signal handler catches any SIGABRT. The camera is not
  the fault, and it masks GVSOC's real message.
- PMSIS's `extern_alloc` serves HyperRAM **from the top**, so free space is
  *below* `CNN_L3_MEMORY`, not above it. Derive buffer addresses from the arena
  base rather than assuming it starts at 0.
