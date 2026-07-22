# mcu/board — on-silicon cycle + FPS bench (GAP8 AI-deck)

Confirms the Phase-10 GVSOC **sim** cycle numbers on real GAP8 silicon, for the
four MCU finalists. Design: `docs/superpowers/specs/2026-07-22-mcu-board-net-bench-design.md`;
build plan: `docs/superpowers/plans/2026-07-22-mcu-board-net-bench.md`.
Cycles are **ranking-only** (LUT/Stage-0 discipline); the wall-clock ms/FPS is the
honest absolute — expect it HIGHER than sim (`BOARD.md`: real HyperRAM latency the
sim omits).

## Prereqs

- The Bitcraze toolchain docker that built the wifi-img-streamer (`GAP8_V2`), and
  a working `~/aideck-gap8-examples`. **Not** the `tfm-gap8` sim image (GVSOC-only).
- `.venv-nas` (onnx) on the host — `build_bench.sh` reads each model's IO shapes.
- A flashed AI-deck (ESP + GAP8 reachable over radio) + Crazyradio. Deck firmware
  flashing is radio-only; the host drone is a Crazyflie 2.1 **Brushless** — see the
  `crazyflie-is-cf21bl` note before touching CF base firmware.

## The loop (per model)

```bash
# 1) assemble the app dir (host, .venv-nas):  <model> <res> [L2] [stack]
mcu/board/build_bench.sh cand_a5fddcc354bd 192

# 2) build in the Bitcraze docker (the one that built the streamer):
#      cd examples/ai/net-bench-cand_a5fddcc354bd && make clean model all image

# 3) flash over radio:
mcu/board/flash_bench.sh cand_a5fddcc354bd

# 4) read the numbers — any ONE of:
python mcu/board/bench_receiver.py --uri radio://0/80/2M   # live over CRTP (cflib)
#   or watch cfclient -> Console tab (the same CPX stream, zero code)
#   or pipe a captured console into the parser:
#      <some console source> | python mcu/board/bench_receiver.py --stdin
# -> BENCH model=cand_a5fddcc354bd res=192 cyc=<c> nodes=<n> clk_us=<u> n=20 fcl=175
#    receiver prints derived ms + FPS and logs data/mcu/board/<model>.txt
```

Then repeat for the other three:

```
cand_19efff428be8    192
cand_863c75818953    192
yolo11n_pose_160_raw 160 160000        # baseline: more L2 headroom
```

## Tunables / troubleshooting

- `construct rc=3` (L2 alloc) → lower `MODEL_L2_MEMORY` (4th `build_bench.sh` arg),
  e.g. `mcu/board/build_bench.sh cand_a5fddcc354bd 192 70000`.
- **Silence after flashing** → run a smoke build to localise it (a GAP8 crash is
  silent — buffered output is lost on abort):
  `make clean model all image APP_CFLAGS+=-DBENCH_SMOKE=3` stops right after
  `Construct`. The last `SMOKE` line reached tells you the stage:
  1 boot · 2 cluster · 3 construct · 4 io-placed · 5 dispatch.
- Garbled floats → the firmware prints **integers only** (GAP8 `%f` is unreliable);
  the receiver derives ms + FPS on the host.

## Results (fill from silicon; sim columns from `state/winner_mcu/winner.json`)

| model | res | sim cyc | sim FPS | **meas cyc** | **meas ms** | **meas FPS** |
|---|---|---|---|---|---|---|
| yolo11n_pose_160_raw | 160 | 59.85 M | 2.92 | | | |
| cand_a5fddcc354bd | 192 | 58.39 M | 3.0 | | | |
| cand_19efff428be8 | 192 | 43.26 M | 4.05 | | | |
| cand_863c75818953 | 192 | 60.44 M | 2.91 | | | |

The claim silicon tests: does measured `cyc` preserve the sim ranking
`a5fddcc ≤ baseline` (CP 10.3)? A documented L2-infeasibility for any graft is
itself a valid recorded result.
