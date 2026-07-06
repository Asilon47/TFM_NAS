# Stage R (ii) — Structured pruning + KD for YOLO-family models on Jetson-class GPUs

*ARS `deep-research` three-way-scan (low-oversight mode), 2026-07-06. Question: does the
Phase 6 (DepGraph pruning → winner-v2) + Phase 8 (KD) design match the evidence, and which KD
losses should CP 8.2 use? Feeds CP 6.1–6.3 and CP 8.2 design detail.*

## Shortlist (3 papers, WHY / HOW / WHAT)

### 1. DepGraph — Fang et al., CVPR 2023 ([arXiv:2301.12900](https://arxiv.org/abs/2301.12900))
- **WHY** — Structural pruning fails on real architectures because coupled layers (residuals,
  concats, heads) must be pruned *consistently*; manual per-arch case analysis doesn't scale.
- **HOW** — Build a dependency graph over parameters, automatically group coupled ones, prune
  and score at **group level** (their finding: even simple norm criteria work once grouping is
  right). Validated across CNNs/ViTs/GNNs/LSTMs.
- **WHAT** — Fully automatic, architecture-agnostic structural pruning with consistent groups;
  the maintained Torch-Pruning implementation ships a YOLOv8 pipeline. Our `GraftedPoseModel`
  (plain `nn.Sequential` of convs) is squarely inside its supported class.

### 2. YOLOv8 compression framework — [arXiv:2509.12918](https://arxiv.org/abs/2509.12918) (2025)
- **WHY** — Real-time aerial detection on edge devices: FLOPs cuts alone don't deliver;
  pruning must survive deployment through TensorRT with accuracy recovered.
- **HOW** — **Structured pruning + channel-wise-distillation recovery** on YOLOv8, then TRT —
  i.e., literally our Phase 6 + Phase 8 combination on our model family.
- **WHAT** — Reported pipeline: compression reaches ~45 FPS, TRT lifts it to **68 FPS with
  AP50 47.9 → 47.6** (−0.3) — pruning + KD-recovery + TRT compose without collapse. Treat the
  exact figures as their task/hardware, but the *shape* of the result (large measured speedup,
  sub-point AP cost after distillation-based recovery) is the Phase-6/8 hypothesis.

### 3. Channel-wise Distillation (CWD) — Shu et al., ICCV 2021 ([arXiv:2011.13256](https://arxiv.org/abs/2011.13256))
- **WHY** — Pixel/pair-wise feature mimicking is noisy for dense prediction (background
  dominates); what feature-KD form actually helps small dense-prediction students?
- **HOW** — Normalize each channel's activation map into a spatial probability distribution;
  minimize KL(teacher‖student) per channel — the salient regions dominate the loss.
- **WHAT** — CWD outperforms pixel-wise/pair-wise feature KD on dense tasks and measurably
  lifts lightweight detectors at zero inference cost; a natural fit for our P3/P4/P5 seam
  (adapter outputs vs teacher neck outputs).

## Cross-paper synthesis

- **Common WHY:** compression that survives *deployment* — consistency (DepGraph), measured
  latency (2509.12918), and recovery signal quality (CWD) are three faces of the same goal.
- **Divergent HOW:** graph-level grouping vs pipeline engineering vs loss design.
- **Strongest WHAT:** prune-with-groups → recover-with-distillation → measure-through-TRT is an
  evidenced, composable pipeline on exactly our model family; feature-KD for dense prediction
  should be channel-normalized (CWD), not raw L2 mimic.
- **Unresolved gap:** keypoint/pose-branch distillation on YOLO-pose has no canonical
  treatment (detection KD literature covers cls/box; kpts are usually plain regression mimic) —
  CP 8.2's kpt term will be our own design, worth stating in the thesis.

## Design implications for Phases 6 and 8

1. **CP 6.1 stands as planned** (DepGraph, group-L2, `ignored_layers` = head output convs +
   dfl). Supported directly by paper 1; the group-level criterion is the paper's own
   recommendation.
2. **`round_to=16` is justified by vendor docs, cite them:** TensorRT tensor-core convolutions
   want I/O channels aligned (FP16 kernels degrade / implicitly pad when in-channels aren't a
   multiple of 8; alignment granularities reach 16+ depending on kernel). 16 covers fp16 and
   int8 granularities with one knob.
   ([NVIDIA TensorRT performance docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html))
3. **Measured-only latency for pruned nets is the right rule** — the 45→68 FPS jump in paper 2
   happens *at the TRT step*, i.e., engine-level effects dominate; FLOPs/params deltas do not
   predict deployed latency for pruned channel patterns. (Matches our LUT-can't-cover-off-grid
   stance.)
4. **Option for CP 6.2 (flag at that CP, user decision):** paper 2 recovers *with
   distillation* rather than plain fine-tuning. We could add the CWD term to the Phase-6
   recovery fine-tune (teacher = unpruned winner-v1.5 — free, no Phase-8 teacher needed yet).
   Cheap to implement once CP 8.2's loss exists; keeps Phase 6/8 separation clean if declined.
5. **CP 8.2 loss menu, evidence-ranked:** (a) response-KD on the cls map (plain KL, the
   distill/README plan); (b) **box branch: Localization Distillation** — YOLO11's DFL already
   represents boxes as distributions, and LD ([arXiv:2102.12252](https://arxiv.org/abs/2102.12252))
   distills exactly such distributions with a temperature softmax — a near-zero-friction fit;
   (c) **feature mimic at P3/P4/P5: CWD form** if a feature term is added at all;
   (d) keypoints: regression mimic (no literature standard — our design).

## Fidelity notes (low-oversight scan)

Verified this session: DepGraph's scope/claims and group-level criterion; the 2509.12918
pipeline shape and its 45→68 FPS / 47.9→47.6 AP50 figures (abstract-level); CWD's mechanism and
its advantage over pixel/pair-wise feature KD; NVIDIA's channel-alignment guidance (fp16
tensor-core in-channels % 8, implicit padding otherwise). Not re-read at table level — pull
exact numbers before quoting in the thesis. LD (2102.12252) mechanism verified at
abstract level; its DFL fit to YOLO11 is our inference, validate when implementing CP 8.2.
