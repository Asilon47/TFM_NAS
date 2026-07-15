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
