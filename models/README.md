# Models — the cross-family comparison set

Every architecture we've considered for the drone-gate pose task, organised by family. The
weight (`.pt`) and ONNX (`.onnx`) binaries are gitignored (regenerable from the Kaggle pulls /
the exporters); this manifest is the tracked record.

**Latencies** are Jetson Orin Nano, TensorRT 10.3, **mode 0 / 612 MHz, clocks locked**, @640,
batch 1 (fp32 / fp16 ms). **Accuracy is NOT apples-to-apples**: `baseline`/`anchor` are
COCO-pretrained + full Ultralytics recipe; every other family is trained **from scratch**
(grafts: bare-AdamW; pruned: 50-ep bare-AdamW recovery; dense: stock recipe) — the from-scratch
penalty is ~2.3 mAP (CP 3c.1's `ctrl_n` control). Single-seed except the pretrained anchors;
**de-noise is owed before any pick**.

## baseline/ — the bar to beat
| file | what | mAP | fp32 / fp16 |
|---|---|---|---|
| `yolo11n_pose_gate.pt` + `yolo11n_pose_640.onnx` | deployed YOLO11n-pose, gate-trained | **0.877** | 12.75 / 7.58 |

## anchor/ — bigger-model reference (accuracy anchor B)
| file | what | mAP | fp32 / fp16 |
|---|---|---|---|
| `yolo11s_pose_640.onnx` | YOLO11s-pose (ONNX is stock; latency is arch-only) | 0.882 | 21.69 / — |

## graft/ — OFA-MBv3 backbone + YOLO11-pose head (the thesis approach)
Depthwise backbone → **memory-bound on the GPU → slower than the dense baseline** despite fewer
params (the Stage-0 finding).
| file | what | mAP | fp32 / fp16 |
|---|---|---|---|
| `winner_v1_noneck.*` | winner-v1 backbone, no neck | 0.841 | 17.69 / 12.37 |
| `winner_v1_v2topdown.*` | + zero-gated top-down neck | 0.846 | 18.13 / — |
| `winner_v1_v3pan.*` | + PAN neck | 0.842 | 18.38 / 12.75 |

## pruned_baseline/ — DepGraph structured pruning of the gate yolo11n (CP 6.2-B)
Stays dense → **tensor-core-friendly → faster than baseline.** Recovery is noisy (non-monotonic
mAP); latencies measured-only (off the LUT grid). fp16 build is per-model (r15's pruned dims
lack an fp16 SiLU kernel).
| file | prune | mAP | fp32 / fp16 |
|---|---|---|---|
| `prune_r15.*` | −39 % params | 0.834 | **9.50** / (fp16 build fails) |
| `prune_r30.*` | −58 % params | 0.790 | **8.28 / 5.65** |
| `prune_r45.*` | −66 % params | 0.809 | **7.94 / 5.13** |

## dense_scaled/ — YOLO11 width-scaling from scratch (CP 3c.1)
Depth is a dead knob below n (C3k2 repeat floor) → width-only. `w25` = yolo11n's own scale from
scratch = the recipe/pretrain control.
| file | width | mAP | fp32 / fp16 |
|---|---|---|---|
| `dense_w25_ctrl_n.*` | 0.25 (= n) | 0.854 | 11.34 / (measuring) |
| `dense_w20.*` | 0.20 | 0.839 | (measuring) |
| `dense_w15.*` | 0.15 | 0.815 | (measuring) |

---
_Not included (dead ends / redundant): the two graft fallbacks (`idx3`/`idx11`, never
full-trained); the dense depth-duplicates (`d25_w25`/`d33_w25` ≡ `w25`, `d50_w20` ≡ `w20`).
Round-2 Kaggle campaigns (extended prune ratios, dense wave-2 widths) will add points when they
land._
