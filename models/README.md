# Models — the cross-family comparison set

Every architecture considered for the drone-gate pose task, organised by family. Weights
(`.pt`) and ONNX (`.onnx`) are gitignored (regenerable from the Kaggle pulls / exporters); this
manifest is the tracked record.

**Latencies** — Jetson Orin Nano, TensorRT 10.3, **mode 0 / 612 MHz, clocks locked**, @640,
batch 1, ms, measured one-process-at-a-time (fp32 is the reliable axis; **fp16 carries ±~20 %
TRT build variance** — indicative). **Accuracy is NOT apples-to-apples**: `baseline`/`anchor`
are COCO-pretrained + full recipe; grafts/dense are from scratch, pruned are pruned-from-baseline
+ 50-ep bare-AdamW recovery. Single-seed except the pretrained anchors — **de-noise owed before
any pick** (the prune ladder is visibly noisy: r30 @ 0.790 is an outlier for its size).

## Full frontier (sorted by accuracy)

| family | model | params | mAP | fp32 | fp16 | vs baseline fp32 |
|---|---|---|---|---|---|---|
| anchor | yolo11s | 9.7M | 0.882 | 21.70 | 14.93 | +70 % ✗ |
| **baseline** | **yolo11n** | 2.7M | **0.877** | **12.74** | **7.75** | — |
| dense | w30 | 3.9M | 0.856 | 15.27 | — | +20 % ✗ |
| **dense** | **w25** (ctrl_n) | 2.7M | **0.854** | 11.33 | 8.11 | **−11 %** |
| graft | v2topdown | ~2.4M | 0.846 | 18.15 | 12.58 | +42 % ✗ |
| dense | w22 | 2.3M | 0.845 | 11.55 | 6.95 | −9 % |
| graft | v3pan | ~2.5M | 0.842 | 18.37 | 12.76 | +44 % ✗ |
| graft | winner-v1 noneck | ~2.4M | 0.841 | 17.67 | 12.38 | +39 % ✗ |
| dense | w20 | 1.9M | 0.839 | 11.26 | 6.93 | −12 % |
| **prune** | **r20** (−41 %) | 1.6M | **0.838** | **9.52** | 5.91 | **−25 %** |
| dense | w18 | 1.6M | 0.834 | 10.01 | 6.48 | −21 % |
| prune | r15 (−39 %) | 1.6M | 0.834 | 9.54 | 5.93 | −25 % |
| prune | r10 (−31 %) | 1.9M | 0.830 | 9.82 | 6.13 | −23 % |
| prune | r35 (−59 %) | 1.1M | 0.826 | 8.36 | 5.38 | −34 % |
| dense | w15 | 1.2M | 0.815 | 9.53 | 6.30 | −25 % |
| dense | w13 | 1.0M | 0.813 | 9.56 | 6.45 | −25 % |
| prune | r45 (−66 %) | 0.9M | 0.809 | 7.94 | 7.18 | −38 % |
| prune | r55 (−76 %) | 0.65M | 0.798 | 7.66 | 5.07 | −40 % |
| prune | r30 (−58 %) | 1.1M | 0.790 | 8.28 | 5.34 | −35 % |

## The story
- **Every dense/pruned model beats the baseline on latency; every graft loses** (the depthwise
  OFA backbone is memory-bound → 17–18 ms despite fewer params). Accuracy is nearly flat across
  families (0.79–0.85 from-scratch), so **latency separates them.**
- **Pruning dominates scaling at matched accuracy.** At ~0.838: prune r20 (9.52 ms) vs dense w20
  (11.26 ms) → pruning is 15 % faster for the same mAP; likewise r15 (9.54) < w18 (10.01) at
  0.834. Reason: pruned nets inherit the baseline's pretrained weights; dense-scaled train from
  scratch. So **the pruned family owns the Pareto frontier below 0.85**, dense w25 owns the top.
- **Pareto-optimal (fastest per accuracy, all faster than baseline):**
  dense w25 (0.854 @ 11.33) › **prune r20 (0.838 @ 9.52)** › prune r35 (0.826 @ 8.36) ›
  prune r45 (0.809 @ 7.94) › prune r55 (0.798 @ 7.66). **Standout: prune r20 — 25 % faster at
  0.838**, only 0.016 below the best dense point.
- Gap to baseline's 0.877 is ~0.02–0.04 (from-scratch / weak recovery) → the target for
  distillation: **distil prune r20 (or a gentler rung) against the 0.877 teacher** is the
  clearest shot at a Pareto-dominant model.

## Folder
```
models/
  baseline/         yolo11n_pose_gate           (.pt + .onnx)
  anchor/           yolo11s_pose_640.onnx
  graft/            winner_v1_{noneck,v2topdown,v3pan}  (.pt + e2e .onnx)
  pruned_baseline/  prune_r{10,15,20,30,35,45,55}       (.pt + .onnx)   ← 7-rung ladder
  dense_scaled/     dense_w{13,15,18,20,22,25,30}       (.pt + .onnx)   ← 7-width curve
```
_Excludes dead ends (graft fallbacks; dense depth-duplicates). fp16 latencies are single clean
builds (±20 % variance); w30 fp16 skipped (dominated). Some fp16 builds are slow (autotuner) but
none genuinely fail on an idle board — the earlier "r15 fp16 FAIL / hangs" were GPU-contention
artifacts, since corrected._
