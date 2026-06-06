# Hardware-Aware NAS for the Jetson Orin Nano

## One-line summary

Sample a compact child architecture from a pretrained **Once-for-All (OFA)**
supernet, score it against a **Jetson-measured latency lookup table** plus task
accuracy, drive a **Bayesian Optimization (BO)** loop that proposes
architectures, and use **Net2Net** function-preserving transforms to warm-start
the weights of every proposal so each evaluation is cheap.

## Motivation

Hardware-aware NAS usually fails for one of two reasons:

1. **Latency proxy is wrong.** FLOPs and parameter counts don't predict Jetson
   latency — a 5×5 depthwise on 28×28×96 is often faster than a 3×3 regular
   conv with the same FLOPs because of memory-bandwidth effects.
2. **Each candidate is too expensive to evaluate.** Training every candidate
   from scratch is infeasible; proxy tasks under-rank the good ones.

This project attacks both problems head-on:

- **Lookup table measured on the target device (this repo's current output)**
  gives exact, deployment-faithful latency per block. No FLOPs proxy.
- **OFA + Net2Net** means we never train from scratch. Every candidate starts
  from weights that already solve the task; search becomes fine-tuning, not
  training.
- **BO** with a GP/TPE surrogate turns the discrete architecture space into an
  efficient query strategy — ~30–100 candidates usually beats thousands of
  random rollouts.

## The four pieces

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 1. Latency LUT       │   │ 2. OFA supernet      │   │ 3. BO search loop    │
│  (this repo)         │──▶│  pretrained on task  │──▶│  proposes arch α     │
│  block → latency_ms  │   │  produces child W(α) │   │  fits surrogate      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
                                      │                          │
                                      ▼                          ▼
                           ┌──────────────────────┐   ┌──────────────────────┐
                           │ 4. Net2Net warm-start│   │  Objective:          │
                           │  α' ← expand/deepen  │   │  acc − λ · latency   │
                           │  weights preserved   │   │  (from LUT + eval)   │
                           └──────────────────────┘   └──────────────────────┘
```

### 1. Latency LUT (status: **implemented** — see `lut/README.md`)

Precomputed `(block, cfg, input_shape) → {latency_ms, peak_mem_mib, params,
flops}` measured on the Jetson Orin Nano under TensorRT FP16, at a fixed
power mode. Covers MBConv essentials plus segmentation/detection specials
(FPN, ASPP, dilated conv, Upsample, PixelShuffle, heads). The schema is
append-only (`data/lut.jsonl`), so new blocks added later never invalidate
existing rows.

The LUT is called **once per candidate** during search — at tens of
microseconds per lookup, it replaces an on-device re-measurement (which would
cost 10–30 seconds per arch with engine builds). The BO loop can evaluate
thousands of candidates in seconds instead of hours.

### 2. Once-for-All supernet

OFA ([Cai et al., 2019](https://arxiv.org/abs/1908.09791)) trains a single
supernet once with progressive shrinking: every subnet (choice of depth,
kernel, expand ratio, width) inherits weights that were co-trained to work.
After training, **any** subnet can be sampled in O(1) and evaluated without
further training, often within 1–2% of its from-scratch accuracy.

Plan:

- **v1 — linear-only search space (explicit non-goal: no non-linear topology).**
  Fix the macro-topology to a linear stack (MBConv backbone → FPN → head).
  Search only over:
  - per-stage depth `d ∈ {2, 3, 4}`
  - per-block kernel `k ∈ {3, 5, 7}`
  - per-block expand ratio `e ∈ {3, 4, 6}`
  - per-stage width multiplier `w ∈ {0.75, 1.0, 1.25}`

  This is exactly the OFA "MobileNetV3-large" search space and has a
  well-validated pretrained supernet we can reuse. Search is cheap and the
  linearity keeps both the LUT lookup and the Net2Net step trivial.

- **v2 — non-linear topology (later).**
  Add branch/skip choices (e.g., FPN cross-levels, DeepLab ASPP insertion
  points, residual vs. dense connections). This requires either:
  - retraining (or fine-tuning) a supernet that spans these choices, **or**
  - dropping OFA for that subspace and using a DARTS-style continuous
    relaxation on the non-linear axes only.

  We defer this until v1 is validated end-to-end.

### 3. Bayesian Optimization search

The optimizer proposes architectures `α` from the discrete search space.
Each proposal is scored by the composite objective:

```
J(α) = accuracy(α) − λ · latency_ms(α) − μ · max(0, mem(α) − mem_budget)²
```

where:

- `accuracy(α)` comes from fine-tuning the OFA-extracted + Net2Net-warm-started
  weights for a short budget (e.g., 5 epochs) on the target task.
- `latency_ms(α)` is a **sum of LUT lookups** along `α`'s block list —
  additive latency is a good approximation for linear topologies, and the
  residual error (TRT fusions across block boundaries) is small enough (< 10%
  typical) that BO handles it as noise.
- `mem(α)` and `params(α)` are also summed from the LUT.
- `λ, μ` control the accuracy/latency/memory tradeoff.

**Surrogate choice:**

- **GP with a structured kernel over the categorical search space** (e.g.
  Hamming + Matérn) — the canonical BO pick for ≤ 20 discrete dimensions.
- **TPE (Tree-structured Parzen Estimator)** as a fallback for wider search
  spaces, via Optuna. Cheaper to fit, handles conditional parameters better.

Acquisition: Expected Improvement, or qEI for batch (evaluate 4 candidates
in parallel on the GPU).

### 4. Net2Net warm-starting

[Net2Net](https://arxiv.org/abs/1511.05641) (Chen et al., 2015) gives two
function-preserving operators:

- **Net2WiderNet** — widen a layer's channels while exactly preserving the
  network function (copy + split neurons and rescale).
- **Net2DeeperNet** — insert an identity-initialized layer that the optimizer
  can then specialize.

Both produce a new, larger network whose initial loss is *exactly* the same as
the old one's. Training resumes from "already-works" instead of "random noise."

Role in this project:

- When BO proposes a new `α` that differs from a recently-evaluated `α_prev`
  only by a widen or deepen step, we apply Net2Net instead of re-extracting
  from the OFA supernet. Fine-tuning converges in ~2× fewer epochs.
- When the proposal differs along a dimension Net2Net doesn't cover (kernel
  size change, SE on/off), we fall back to re-extracting from OFA.
- This gives us an architecture cache keyed on `α`: the BO loop does best
  when candidates are close in edit-distance, and Net2Net lets us exploit that
  locality.

### Why this combination?

| Component | Solves | Alternative considered | Why not |
|---|---|---|---|
| LUT on device | accurate latency | FLOPs proxy | Misranks memory-bound vs. compute-bound ops on Jetson |
| OFA supernet | evaluation cost | Training from scratch | 300+ GPU-hours per arch |
| BO | sample efficiency | Random / evolutionary | Needs 10× more candidates to find the same frontier |
| Net2Net | warm-start locality | Re-extract every time | Wastes OFA's already-trained weights in the local neighborhood |

No component is load-bearing alone — together they make on-laptop NAS against
a Jetson target realistic (days, not weeks).

---

## Repository status

- **`lut/` (Phase 0) — implemented.** Builds `data/lut.jsonl`. See
  `lut/README.md` for the guide.
- **`supernet/` (CP 1.1–1.3) — implemented.** Wraps OFA's reference supernet;
  given an arch descriptor, returns a PyTorch `nn.Module` with pretrained
  weights inherited. CP 1.4 (ImageNet sanity check) is next.
- **`search/` — planned.** BO loop: arch encoder ↔ surrogate, acquisition, and
  a worker that drives OFA → Net2Net → fine-tune → score.
- **`net2net/` — planned.** Operators + a graph diff that decides whether a
  BO proposal can be reached via Net2Net from the current best (and if so,
  applies the transform).
- **`eval/` — planned.** Short fine-tuning harness (target dataset, 5 epochs,
  fixed seed) plus a final "long-train" evaluator for the winning arch.

## Milestones

1. **M1 — LUT complete.** `lut.jsonl` has all catalog rows, plus
   `device_info.json` at the chosen power mode. ✅ infrastructure ready
2. **M2 — OFA subnet extraction works offline.** Given an arbitrary arch
   descriptor, produce a PyTorch model; confirm accuracy on a held-out split
   within 1% of OFA's published numbers.
3. **M3 — BO over linear search space.** At least 50 BO rounds, with a Pareto
   frontier of `(accuracy, latency_ms)` that strictly dominates a random-search
   baseline of the same budget.
4. **M4 — Net2Net caching cuts eval cost.** Mean fine-tune epochs per candidate
   drops by ≥ 30% versus re-extracting every time.
5. **M5 — Non-linear topology.** Add FPN-cross and ASPP-insertion choices;
   verify the LUT's additive-latency assumption still holds within 15%.

## Open design questions (tracked here, resolved per-milestone)

- **Target dataset.** Segmentation (Cityscapes / ADE20K) vs. detection (COCO).
  Decides the heads in the search space and the accuracy metric (mIoU vs. mAP).
- **Latency additivity error.** We assume `latency(α) ≈ Σ LUT[block]`. Need to
  measure a few full networks end-to-end on the Jetson and compare against the
  sum — if error > 15% on realistic architectures, switch to a learned
  correction term (GP on residuals) or move to a block-pair LUT.
- **Net2Net + BN.** Net2Net's function-preservation breaks BatchNorm's running
  statistics. Either freeze BN during the first warm-start epoch, or use the
  "re-estimate BN" trick (one pass of forward-only with train-mode BN stats).
- **Search-space size vs. OFA coverage.** OFA's pretrained supernet only covers
  a specific search space. Proposals outside it must fall back to
  train-from-scratch or a smaller from-OFA subspace + Net2Net expansion into
  the new axes.

## References

- **OFA**: Cai et al., *Once-for-All: Train One Network and Specialize it for
  Efficient Deployment* — ICLR 2020. https://arxiv.org/abs/1908.09791
- **Net2Net**: Chen et al., *Net2Net: Accelerating Learning via Knowledge
  Transfer* — ICLR 2016. https://arxiv.org/abs/1511.05641
- **Hardware-aware NAS with LUTs**: Wu et al., *FBNet* — CVPR 2019;
  Cai et al., *ProxylessNAS* — ICLR 2019.
- **BO for NAS**: Kandasamy et al., *Neural Architecture Search with Bayesian
  Optimization and Optimal Transport* — NeurIPS 2018.

## Non-goals (for the whole project, not just v1)

- Training any supernet from scratch (we use OFA's published weights).
- Cross-device transfer (the LUT is Jetson-specific; a new device means a new
  LUT sweep).
- Beating SOTA accuracy on standard benchmarks — we target Pareto-dominance
  over MobileNetV3 at Orin Nano's latency budget, not absolute SOTA.
- Real-time / online NAS. This is offline: search, commit, deploy.
