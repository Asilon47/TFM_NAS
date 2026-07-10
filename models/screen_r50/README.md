# OFA-ResNet50 latency screen — the "different supernet?" decision gate

**Question (D3).** OFA-MBv3 was the only pretrained supernet searched. The one *other* supernet
that is pretrained **and** samplable **and** dense (tensor-core-friendly) is **OFA-ResNet50**.
Before descoping it on a roofline *estimate* (~2× over budget), measure it: does even the
**smallest** OFA-R50 subnet backbone fit the Nano budget at 640?

**Method.** `python -m expand.screen_r50` exports the min / median / max corners of the standard
`ofa_resnet50` design space as **backbones** (classifier stripped, P3/P4/P5 taps) to ONNX @640,
then `lut.orchestrate.bench_model` measures each on the Nano — **mode 0 / 612 MHz, clocks locked,
one process at a time**, TRT 10.3, fp32. Latency at TRT fp32 is weight-value-independent, so
random init is used (no pretrained checkpoint needed). Full record:
`screen_r50_result.json`; ONNX (`*.onnx`) are gitignored/regenerable.

**Budget.** baseline yolo11n-pose = **12.75 ms** fp32 e2e. The graft's pose adapter+head offset is
**3.84 ms** (`data/pose_stem_head_offset.json`) — a *lower* bound for R50, whose adapter carries
~10× the channels — so a backbone must land under **8.91 ms** to give e2e ≤ baseline (the
Phase-3b honest ceiling was even tighter, 7.16 ms).

## Result

| corner | params | GFLOPs | backbone ms (fp32) | eff. TFLOP/s | e2e est. (+3.84) | ×baseline e2e | ×backbone budget |
|---|---|---|---|---|---|---|---|
| **min** | 5.2M | 15.7 | **16.17** | 0.97 | ~20.0 | **1.57×** | **1.82×** |
| mid | 14.2M | 40.5 | 27.28 | 1.49 | ~31.1 | 2.44× | 3.06× |
| max | 46.1M | 122.3 | 58.11 | 2.10 | ~61.9 | 4.86× | 6.52× |

std < 0.05 ms on every point (locked clocks). peak working set 60–95 MiB (fits 8 GB easily —
it is compute, not memory, that is the wall here).

## Verdict — **INFEASIBLE** (a full OFA-R50 re-search is not worth building)

The **smallest** OFA-R50 backbone (16.17 ms) already **exceeds the whole baseline e2e**
(12.75 ms) — **1.82×** the 8.91 ms backbone budget — *before* the pose head runs. The entire
elastic space (16→58 ms backbone) is over budget; **no subnet fits**, so there is nothing for a
search to find. The roofline estimate that descoped it (D3) is now **measured**, not assumed.

**Mechanism (the interesting part).** R50 is hardware-**efficient**: 0.97–2.10 effective TFLOP/s,
its dense 1×1/3×3 bottlenecks saturate the Orin tensor cores — **~2× the throughput of yolo11n**
(~0.5 TFLOP/s) and ~3–7× the MBv3 graft's ~0.30. It loses anyway because it is too
**compute-heavy**: 15.7 GFLOPs at the floor is 2.4× yolo11n's ~6.5. So the two cheap-to-search
supernets fail for *opposite* reasons — **MBv3 is memory-bound, R50 is compute-bound** — and
yolo11's own FLOP/efficiency balance is precisely why *compressing it* (prune/scale) wins.
