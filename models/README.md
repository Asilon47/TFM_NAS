# Models — the cross-family comparison set

Every architecture we've considered for the drone-gate pose task, organised by family. The
weight (`.pt`) and ONNX (`.onnx`) binaries are gitignored (regenerable from the Kaggle pulls /
the exporters); this manifest is the tracked record.

**Latencies** — Jetson Orin Nano, TensorRT 10.3, **mode 0 / 612 MHz, clocks locked**, @640,
batch 1, ms. **All measured in one clean session (2026-07-08).** fp32 is the reliable axis;
**fp16 carries ±~20 % build variance** (TRT's fp16 autotuner picks different kernels each build)
— treat fp16 as indicative, not exact. `FAIL` = TRT could not build an fp16 engine for that
model's channel dims.

**Accuracy is NOT apples-to-apples**: `baseline`/`anchor` are COCO-pretrained + full Ultralytics
recipe; every other family is **from scratch** (grafts: bare-AdamW; pruned: 50-ep bare-AdamW
recovery; dense: stock recipe). The from-scratch penalty is ~2.3 mAP (CP 3c.1's `ctrl_n`
control). Single-seed except the pretrained anchors; **de-noise is owed before any pick.**

| family | model | mAP | fp32 | fp16 | vs baseline (fp32) |
|---|---|---|---|---|---|
| **baseline** | `baseline/yolo11n_pose_gate` | **0.877** | 12.74 | 7.75 | — |
| anchor | `anchor/yolo11s_pose_640` | 0.882 | 21.70 | 14.93 | +70 % slower |
| **graft** | `graft/winner_v1_noneck` | 0.841 | 17.67 | 12.38 | +39 % slower |
| graft | `graft/winner_v1_v2topdown` | 0.846 | 18.15 | 12.58 | +42 % slower |
| graft | `graft/winner_v1_v3pan` | 0.842 | 18.37 | 12.76 | +44 % slower |
| **pruned** | `pruned_baseline/prune_r15` (−39 %) | 0.834 | **9.54** | FAIL | **−25 % faster** |
| pruned | `pruned_baseline/prune_r30` (−58 %) | 0.790 | **8.28** | 5.34 | **−35 % faster** |
| pruned | `pruned_baseline/prune_r45` (−66 %) | 0.809 | **7.94** | 7.18 | **−38 % faster** |
| **dense** | `dense_scaled/dense_w25_ctrl_n` | **0.854** | **11.33** | 8.11 | **−11 % faster** |
| dense | `dense_scaled/dense_w20` | 0.839 | **11.26** | 6.93 | **−12 % faster** |
| dense | `dense_scaled/dense_w15` | 0.815 | **9.53** | 6.30 | **−25 % faster** |

## The story in one read
- **Every dense/pruned model beats the baseline on latency; every graft loses.** The depthwise
  OFA backbone is memory-bound on the GPU (0.30 vs 0.60 TFLOP/s effective) → the graft is
  17–18 ms despite fewer params. Pruning/scaling the *dense* baseline keeps it tensor-core-fed.
- **Two standout candidates** (faster than baseline, best accuracy in their trade region):
  `dense w0.25` (11.33 ms, **0.854** — 11 % faster + top from-scratch accuracy) and
  `pruned r15` (9.54 ms, 0.834 — 25 % faster). Speed-vs-accuracy is the pick axis.
- **Accuracy is nearly flat** across families (0.79–0.85 from-scratch); latency separates them.
- fp16 quirks: `r15` won't fp16-build (odd pruned dims); fp16 latencies are noisy (build
  variance) — a real deployability signal for the odd-channel members.

---
_Not included (dead ends / redundant): the two graft fallbacks (`idx3`/`idx11`, never
full-trained); the dense depth-duplicates (`d25_w25`/`d33_w25` ≡ `w25`, `d50_w20` ≡ `w20`).
Round-2 Kaggle campaigns (extended prune ratios, dense wave-2 widths) will add points when they
land. **Measurement note:** the first pass suffered GPU contention (two bench batches + a
foreground run hit the single Jetson at once, inflating numbers to 18–35 ms); this table is the
clean single-process re-measurement — the baseline re-benched to 12.74 vs Stage-0's 12.75,
confirming the clocks were locked all along._
