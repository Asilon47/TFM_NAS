# Models — the cross-family comparison set

Every architecture considered for the drone-gate pose task, organised by family. Weights
(`.pt`) and ONNX (`.onnx`) are gitignored (regenerable from the Kaggle pulls / exporters); this
manifest is the tracked record.

**Latencies** — Jetson Orin Nano, TensorRT 10.3, **mode 0 / 612 MHz, clocks locked**, @640,
batch 1, ms, measured one-process-at-a-time. The long-standing "**fp16 carries ±~20 % TRT build
variance — indicative only**" caveat (procedure.md 2026-07-08) is **UNDER REVISION (2026-07-17)**:
it was never directly measured, it was *inferred* from outliers in a session that also documented
a contention incident and a contention-caused "r15 fp16 FAIL". The first controlled test —
v2_act292, 3 rebuilds with the timing cache wiped between each, idle board
(`bench_model --repeat 3 --fresh-cache`) — measured **0.34 %** (7.2087/7.2212/7.2330), not 20 %.
One graph is not all graphs, so the caveat is not retracted: a bimodal tactic pick on some other
graph could still swing (the standing suspect is **r45 fp16 = 7.18**, wedged between r35's 5.38
and r55's 5.07 despite sitting between them in size — re-run it with `--repeat 3 --fresh-cache`
to settle it). But treat ±20 % as an untested worst case, not a measured property. **Accuracy is NOT apples-to-apples**: `baseline`/`anchor`
are COCO-pretrained + full recipe; grafts/dense are from scratch, pruned are pruned-from-baseline
+ 50-ep bare-AdamW recovery. Single-seed except the pretrained anchors — **de-noise owed before
any pick** (the prune ladder is visibly noisy: r30 @ 0.790 is an outlier for its size).

## Full frontier (sorted by accuracy)

| family | model | params | mAP | fp32 | fp16 | vs baseline fp32 |
|---|---|---|---|---|---|---|
| anchor | yolo11s | 9.7M | 0.882 | 21.70 | 14.93 | +70 % ✗ |
| **baseline** | **yolo11n** | 2.7M | **0.877** | **12.74** | **7.75** | — |
| search | s39-40-38-38-14 | 2.8M | 0.871 | 15.27 | 8.84 | +20 % ✗ |
| search | s31-40-40-40-13 | 2.9M | 0.870 | 15.14 | 8.94 | +19 % ✗ |
| search | s40-38-39-36-13 | 2.6M | 0.868 | 14.98 | 8.67 | +18 % ✗ |
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
| graft-pruned | r40 (−64 %) | 1.1M | 0.816 | 11.81 | 8.41 | −7 % |
| dense | w15 | 1.2M | 0.815 | 9.53 | 6.30 | −25 % |
| dense | w13 | 1.0M | 0.813 | 9.56 | 6.45 | −25 % |
| graft-pruned | halp_10p4 +KD | 2.4M | 0.813 | 12.58 | 8.91 | −1 % |
| prune | r45 (−66 %) | 0.9M | 0.809 | 7.94 | 7.18 | −38 % |
| graft-pruned | halp_9p0 +KD | 1.8M | 0.802 | 11.37 | 8.14 | −11 % |
| prune | r55 (−76 %) | 0.65M | 0.798 | 7.66 | 5.07 | −40 % |
| prune | r30 (−58 %) | 1.1M | 0.790 | 8.28 | 5.34 | −35 % |
| **graft-pruned** | **r50_gtay** | 0.76M | **0.795** | 10.23 | **7.48** | −20 % |
| graft-pruned | v2_act292 +KD | 0.63M | 0.762 | 10.40 | **7.22** | −20 % |
| graft-pruned | r60_gtay | 0.44M | 0.777 | 8.85 | 6.36 | −31 % |
| graft-pruned | r60 (−84 %) | 0.49M | 0.759 | 9.01 | 6.58 | −29 % |

## v2_act292 measured (2026-07-17) — the activation spec works, at fp16 only

First real Nano numbers for v2_act292 (previously predicted-only). Mode-0 locked clocks,
`source=jetson_trt`; fp16 is the **median of 3 fresh-timing-cache builds**
(`bench_model --repeat 3 --fresh-cache`).

**Build variance on this graph is negligible**: 7.2087 / 7.2212 / 7.2330 → **spread 0.0243 ms
(0.34 %)**. That licenses comparing v2_act292 against r50_gtay's single-build 7.48 — the gap is
10.7× the spread. It does **not** licence a blanket "fp16 is fine": one graph is not all graphs
(see the header). r45's fp16 = 7.18 remains the outlier that could still be a real bimodal
tactic pick rather than the contention artifact it looks like.

**The rank flips with precision**, and the flip is the point:

| | fp32 | fp16 |
|---|---|---|
| v2_act292 vs r50_gtay | +0.171 ms (**+1.7 % slower**) | −0.261 ms (**−3.5 % faster**) |

The fp16 gap is **10.7× the build spread** — real. Why: the activation-latency oracle
(`ms ≈ 1.200 + 0.0205·act_MB`) that v2_act292's per-stage spec was allocated against is
**fp16-fitted**. It predicts what it was fitted on and degrades elsewhere:

| precision | spec predicted | measured | error |
|---|---|---|---|
| fp16 | 7.156 | **7.221** | **+0.9 %** |
| fp32 | 9.955 | **10.399** | **+4.5 %** |

The spec also hit its *stated* target — peak working set 30.19 vs r50_gtay's 33.26 MiB
(−9 %) at fp32, 17.92 MiB at fp16. It simply does not convert to fp32 latency, so read
HALP-lite allocations at fp16 or not at all.

**Champion: still open — it is a Pareto TRADE, not a domination.** v2_act292 (0.7625 @ 7.221)
is faster; r50_gtay (0.7947 @ 7.482) is more accurate; neither dominates, and both sit under
the baseline's 7.752 fp16. Fences: r50_gtay's 0.7947 is an **un-de-noised single seed** at
**+21 % params** (764 K vs 632 K), so the 3.2-pt accuracy gap is not a like-for-like result.
And the trade is bad in absolute terms — **−11.5 accuracy points to save 6.9 % latency**,
against a baseline already at 129 FPS on a 60 FPS bar. Latency was never the binding
constraint on this device.

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
- **The pruned graft (CP 6.2-G, 2026-07-11) recovers far better than predicted but stays
  strictly dominated**: r40 0.816 (−2.5 vs its 0.841 anchor at 64 % sparsity) and r60 0.759
  (−8.2 at 84 %) are mid-pack per-param, yet every dense/pruned point beats them per-ms — even
  84 % pruning can't buy back the memory-bound deficit. Floor pruner config (uniform/magnitude/
  one-shot/no-KD) → lower bounds; see `graft_pruned/README.md`.
- **The pruning-as-search program (2026-07-11/12) lifted the graft but not past the dense arm**:
  measured technique ordering global_taylor > uniform > HALP-lite (whose linear allocation
  model over-credited concentrated cuts +23–28 % — predicted-latency claims retired) >
  iterative > global_l2; KD +0.85 on the from-init graft, negative on the converged dense
  recovery (technique gains are training-state-dependent, measured both directions).
  **r50_gtay (0.795 @ 7.48 fp16) is the first graft point under the deployed baseline's fp16
  latency**; prune r20 still leads it by +4.3 pts at −1.6 ms. procedure.md 2026-07-11/12.

## Folder
```
models/
  baseline/         yolo11n_pose_gate           (.pt + .onnx)
  anchor/           yolo11s_pose_640.onnx
  graft/            winner_v1_{noneck,v2topdown,v3pan}  (.pt + e2e .onnx)
  pruned_baseline/  prune_r{10,15,20,30,35,45,55}       (.pt + .onnx)   ← 7-rung ladder
  dense_scaled/     dense_w{13,15,18,20,22,25,30}       (.pt + .onnx)   ← 7-width curve
  graft_pruned/     recover_graft_r{40,60}              (.pt + .onnx)   ← CP 6.2-G rungs
```
_Excludes dead ends (graft fallbacks; dense depth-duplicates). fp16 latencies are single clean
builds (build variance measured at 0.34 % on v2_act292, not the ±20 % long assumed — see above); w30 fp16 skipped (dominated). Some fp16 builds are slow (autotuner) but
none genuinely fail on an idle board — the earlier "r15 fp16 FAIL / hangs" were GPU-contention
artifacts, since corrected._
