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

### Cycles (`probes/cyc_probe.sh`) — the DoD number, CLOSED 2026-07-16

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

**Results** (@224 int8, matched raw-head, GAP8 V3 @175 MHz; SIM cycles, RANKING-ONLY):

| model | nodes | .text | AT L2 | cycles | ms | FPS | ops/cyc |
|---|---|---|---|---|---|---|---|
| yolo11n-pose raw-head | 214 | 329 KB | 160 KB (own max) | 94,538,316 | 540 | 1.85 | 3.67 |
| yolo11n-pose raw-head | 214 | 329 KB | 84 KB (matched) | 113,353,831 | 648 | 1.54 | 3.67 |
| graft (winner-v1) | 180 | 405 KB | 84 KB (own max) | 268,207,010 | 1533 | 0.65 | 1.46 |

The **unpruned w1.0** graft loses — 2.37× at matched L2, 2.84× each-at-own-max.
It does 6 % FEWER ops yet burns 2.37× the cycles: 1.46 vs 3.67 ops/cycle.
Depthwise convs are only 7.5 % — the cost is MBv3's inverted bottleneck (1×1
expand/project = 60 %) and unfused activations (26 %, partly patch 0001's doing:
no custom-act DW kernel exists).

**But the PRUNED graft — the actual deliverable — is 1.94× FASTER than the
baseline** (@160, matched 84 KB L2):

| model | params | L2 | cycles | ms | FPS |
|---|---|---|---|---|---|
| yolo11n (deployed baseline) | 2,704,443 | 160 KB | 51,610,157 | 295 | 3.39 |
| graft, unpruned w1.0 | 3,003,835 | 84 KB | 148,218,707 | 847 | 1.18 |
| **graft, PRUNED v2_act292** | **631,851** | 148 KB | **25,541,274** | **146** | **6.85** |

**2.02× faster** each-at-own-L2-ceiling (1.94× at a matched 84 KB). Per-model
ceilings are why `CYC_L2` exists — never carry a budget across models: the
pruned graft's `.text` is 360 KB (heap 154,164 B), the unpruned one's 405 KB.

Pruning bought 4.80× = **3.15× fewer ops × 1.52× better ops/cycle** (narrower
channels shrink the working set, so the memory-bound graft tiles better). The
hardware-conditional finding: the same pruned NAS family is *marginal* on the
Orin (~8 % predicted fp16) and *decisive* on the MCU (**~2×**).

**The bottleneck is not the convolutions.** They run at 0.2–0.4 cycles/op (2.5–5
ops/cycle) — efficient, exactly as the pivot assumed. The cost is elementwise
work: `MatAdd` (MBv3's residual skips) at 45.7 cyc/op and expressions at 91
cyc/op — **44.5 % of cycles for ~3 % of the ops**, pure HyperRAM round-trips.
It is a memory-residency/fusion problem, partly the patch-0001 toolchain gap.

**Deployability gates Phase 10, not the baseline ratio.** The racing bar is
**15–30 FPS**; at 6.85 FPS we are 2.19× short (the baseline, 3.39, is 4.4×
short). Grayscale is worth ~2 % (the stem is 3 % of cycles) — a sensor decision,
not a speed lever. Fusing the elementwise tax → ~13.4 FPS at **no accuracy
cost**; 128 → ~20.9; 96 → ~37 (PULP-Frontnet runs 160×96). Price the shape
BEFORE training it — cycles are weight-independent, so it is free.

**The head is NOT the lever — CP 10.2's Frontnet head swap is dropped** (user,
2026-07-16; PROJECT_PLAN.md amended). Splitting the pruned graft's 25,541,274
cycles: backbone 20,465,724 (80.1 %), adapter 162,966 (0.6 %), **YOLO11-pose head
4,495,634 (17.6 %)** — deleting the head *outright* reaches only **8.3 FPS**,
still 1.8× short. The 44.5 % elementwise tax is **41.7 pts backbone / 2.8 pts
head**, so a head swap cannot touch it. Keeping the head holds the comparison
(same head both families ⇒ any delta is the backbone = the NAS claim), the metric
(OKS-mAP, continuous with the Orin record), and the warm-head donor. The head is
pruned too (`rest_ratio` covers cv2/cv3/cv4) — hence 17.6 % here vs 26.4 M cyc
@224 unpruned, where it matched yolo11n's 26.1 M because it *is* the same module.

**FPS caveat owed by the raw-head export (applies to BOTH families):** the graph
excludes DFL decode + anchor concat + NMS — they run in C on the fabric
controller, not through AutoTiler (`detect/export_grafted_onnx.py:207-216`). It
cancels from the 2.02× ratio; the absolute FPS owes one FC-side measurement.

### The 2.02× does NOT survive accuracy — resolution screen, CLOSED 2026-07-16

`res_screen.py` prices what CP 10.1 never did: accuracy **at the MCU resolution**.
Rebuild the pruned shape (data-free `l2` — importance-invariant, so it matches the
AGX's `global_taylor`), load the trained state_dict, sweep `imgsz` on CPU. The
**640 control is mandatory and passed**: 0.7629 vs the recorded 0.7637 (Δ 0.0008),
bracketing the AGX's fresh 0.7625 — three measurements of one model inside 0.0012.

| imgsz | graft 631 K | yolo11n 2.7 M | gap | graft drop | base drop |
|---|---|---|---|---|---|
| 640 | 0.7629 | 0.8774 | −0.1145 | — | — |
| 320 | 0.6299 | 0.8014 | −0.1714 | −17.4 % | −8.7 % |
| 224 | 0.4882 | 0.7097 | −0.2215 | −36.0 % | −19.1 % |
| **160** | **0.3158** | **0.5974** | **−0.2816** | **−58.6 %** | **−31.9 %** |
| 128 | 0.2205 | 0.4961 | −0.2756 | −71.1 % | −43.5 % |
| 96 | 0.1241 | 0.3848 | −0.2607 | −83.7 % | −56.1 % |

**The graft is ~2× more resolution-fragile**, and the gap *widens* as resolution
falls (−0.11 @640 → −0.28 @160). **At matched FPS the baseline dominates every
operating point above 0.32 mAP50-95**: graft@160 (0.3158 @ 6.85 FPS) loses to
**yolo11n@96 (0.3848 @ 9.42 FPS) — faster AND more accurate**. The 2.02× buys about
one resolution step; one resolution step of the baseline buys more accuracy than the
graft's entire speed advantage. Cycles off-160 are naive r² scaling, which *flatters*
the graft (the L1 cliff worsens it as resolution rises) — so the dominance is
understated. This is the MCU leg's Stage-0 moment.

**Fences:** both arms are **640-trained lower bounds**. The mismatch is *identical*,
so the comparison is fair even though the absolutes are not; a model trained at each
`imgsz` scores higher. Closing −0.28 needs the graft to gain ~2× more from
train-at-resolution than the baseline. Three candidate mechanisms are architectural
rather than train/test: a 40 ch P3 tap vs the baseline's 64, 4.3× fewer params, and
**no neck at all** (1×1 adapters ⇒ zero cross-scale fusion, which is what small
objects at low resolution need most — Phase 5's V3 PAN nano-neck is the repair).

```bash
python -m mcu.res_screen --spec prune/specs/v2_act292.json \
    --ckpt data/lightning_out/l4_v2_act292/recover_graft_v2_act292_kd.pt \
    --expect-map 0.7637      # .venv-nas; the 640 control gates the whole run
```

Fences on the cycle ratio: NOT iso-params (631 K vs 2.70 M) — it is the deliverable comparison the
"NAS-born, not a pruned YOLO" constraint defines, not an architecture-controlled
one; **accuracy is unmeasured and is the whole Pareto question** (AGX run pending;
nearest measured neighbour r50_gtay @760 K = 0.7947). Full analysis in
procedure.md "CP 10.1 CLOSED" + "CP 10.1 AMENDED".

Reproduce the pruned point (no training, no data — the spec pins the shape, so a
data-free `l2` prune is architecture-identical to the AGX's `global_taylor`):
```bash
python -m mcu.export_pruned --spec prune/specs/v2_act292.json --imgsz 160 \
    --out models/res160/graft_r292_160_mcu.onnx     # .venv-nas
CYC_RES=160 CYC_L2=84000 mcu/probes/cyc_probe.sh graft_r292_160_mcu
```

Third L2 fact, from the pair: **`.text` scales with kernel complexity, not node
count** — the graft has FEWER nodes (180 vs 214) yet 405 KB vs 329 KB, i.e. 79 %
of L2, leaving a 90,388-byte heap. It *cannot* be given the 160 KB yolo11n can.

Reproduce: `mcu/probes/cyc_probe.sh` (both), or `CYC_L2=84000 mcu/probes/cyc_probe.sh`
for the matched-budget control. Files: `data/mcu/cyc/<model>.json` (+ the
`.L2_160000.json` variant); each JSON self-describes its `autotiler_l2_budget`.

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
