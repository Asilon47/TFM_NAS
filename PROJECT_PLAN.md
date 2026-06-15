# Hardware-Aware NAS — Execution Plan

A living, checkpoint-driven plan for turning the Jetson LUT into a
device-adaptive NAS pipeline built on a weight-sharing supernet with
supernet-expansion capability.

> **How to resume across sessions.** Start each new session by reading:
> 1. `PROJECT.md` — the vision (why this exists).
> 2. `PROJECT_PLAN.md` — this file (where we are).
> 3. `state/plan_state.yaml` (added in CP 1.1) — the last completed
>    checkpoint ID.
> 4. `procedure.md` — update at the end of each session with every thing done in great details and justification.
>
> Every checkpoint has **inputs**, **deliverables**, and a **definition of
> done** — so a phase's completeness is verifiable even weeks later.

---

## Zero-to-subnet pipeline at a glance

```
  Pretrained OFA Supernet                       Jetson LUT (DONE)
          │                                           │
          ▼                                           │
  [P1] OFA Wrapper ──▶ [P2] Subnet Extractor          │
          │                    │                      │
          │                    ▼                      │
          │             [P3] Device-aware search ◀────┘
          │                    │
          │                    ▼
          │             winner subnet α*
          │
          ▼
  [P4] Net2Net ops  ──▶ [P5] SOTA block injection (FusedMBConv / ConvNeXt / MobileViT)
                               │
                               ▼
                        [P6] Fine-tune expanded supernet
                               │
                               ▼
                        [P7] Search on expanded supernet
                               │
                               ▼
                        winner subnet α* (expanded)
                               │
                               ▼
                        [P8] Knowledge Distillation
                             (external SOTA teacher → student α*)
                               │
                               ▼
                        [P9] TRT export for deployment
```

---

## Phase 0 — Foundations (DONE)

- Jetson LUT built and validated. See `lut/README.md` + `PROJECT.md`.
- Catalog: MBConv essentials + seg/det specials.
- Schema is append-only; new blocks can be added later without invalidating
  rows.

**DoD:** `data/lut.jsonl` has ≥ 1 row per catalog entry at a locked
`power_mode`, and `data/device_info.json` exists for that mode.

---

## Phase 1 — Base Supernet Integration

**Goal:** Stand up an OFA-MBv3 supernet we can sample subnets from, offline,
with weights inherited from the published pretrained checkpoint.

### Checkpoints

- **CP 1.1 — Skeleton repo + state file** ✅
  - Create `supernet/`, `search/`, `eval/`, `net2net/`, `expand/`,
    `state/` subdirs.
  - `state/plan_state.yaml` with `current_checkpoint:` field for
    cross-session memory.
  - Pin deps: `torch` (GPU build, laptop has a dGPU), `ofa` (MIT-HAN-Lab
    reference impl), `numpy`, `pyyaml`.
  - **DoD:** `python -c "import ofa; print(ofa.__file__)"` works.

- **CP 1.2 — OFA checkpoint download + cache** ✅
  - `supernet/download_ofa.py`: pulls
    `ofa_mbv3_d234_e346_k357_w1.0` (or `w1.2`) into `~/.cache/ofa/`.
  - `supernet/README.md` documents which checkpoint is pinned
    (URL + SHA256). Future-you will thank present-you.
  - **DoD:** Checkpoint file exists on disk, hash matches the pin.

- **CP 1.3 — Subnet sampler** ✅
  - `supernet/sampler.py`: `sample(arch_dict) -> nn.Module`.
  - Accepts canonical OFA arch spec (`ks=[...]`, `e=[...]`, `d=[...]`).
  - **DoD:** `sampler.sample(random_arch)` forwards a `(1, 3, 224, 224)`
    tensor without error.

- **CP 1.4 — ImageNet sanity**
  - On a 2 k-image ImageNet-val subset, confirm a sampled subnet is within
    1.5 % top-1 of OFA's published accuracy for that arch.
  - **DoD:** One subnet, one published arch, one number matches.

### References
- OFA: Cai et al., ICLR 2020. https://arxiv.org/abs/1908.09791
- OFA code: https://github.com/mit-han-lab/once-for-all

### Risks
- OFA's repo is not actively maintained. Fork-lock a known-good commit.
- Weight loader expects a specific PyTorch version. Pin carefully.

---

## Phase 2 — Subnet Extraction & LUT-aware Scoring

**Goal:** Given a subnet, compute `(accuracy, latency, peak_mem, params,
flops)` **without** re-measuring on the Jetson during search.

### Checkpoints

- **CP 2.1 — Arch → block list translator** 🔲
  - `search/arch_to_blocks.py`: OFA arch_dict + macro-topology →
    ordered `(block_type, cfg, input_shape)` list (LUT keys).
  - Correctly propagate stride/resolution so each block's `input_shape` is
    right.
  - **DoD:** 10 random archs — every emitted tuple has a matching `row_key`
    in `data/lut.jsonl`.

- **CP 2.2 — LUT composite-cost function**
  - `search/cost.py`: `cost(arch) → {latency_ms, peak_mem_mib, params, flops}`.
  - Latency = Σ LUT[block].latency_ms.mean; `peak_mem_mib` = **max** over blocks
    (not the sum — schema.md); deployable memory adds resident weights via
    `search.cost.resident_mem_mib` (weights are precision-scaled).
  - Document the additivity assumption + known error range.
  - **DoD (depth-binned, not a single aggregate — peer-review R4.2):** measure
    full-net latency for ≥5 subnets **spanning the depth range**; report
    `(summed − measured)/measured` **binned by depth** via
    `search/validate_additivity.py`, plus the aggregate. **Pass = no depth bin
    exceeds 15 %.** A single averaged number would mask fusion error that grows
    with depth, so it is not the gate. If any bin breaches → CP 2.3.

- **CP 2.3 — Additivity correction (conditional)**
  - **Trigger (pre-registered — peer-review R4.2 / P1.8):** CP 2.2's by-depth
    report breaches the 15 % bar in **any** depth bin, **or** the residual trends
    upward with depth (the cross-block-fusion signature). Do not wait for the
    aggregate to miss — by then the deep regime is already mispriced.
  - Then: fit a residual GP, `δ = measured − summed` as a function of
    `(depth, total_flops, n_dw)`.
  - **DoD:** Corrected error < 10 % on held-out nets.

- **CP 2.4 — Eval harness (short fine-tune)**
  - `eval/shortft.py`: 5-epoch fine-tune on the target task
    (see open decision D1), fixed seed, fixed LR schedule.
  - Returns top-1 / mIoU / mAP depending on the task.
  - **DoD — reproducibility:** running twice on the same arch gives results
    within 0.5 %.
  - **DoD — proxy-rank fidelity (gates the whole search — peer-review R2.1 /
    P0.2):** fully train ≈8–12 architectures spanning the space; the 5-epoch
    proxy ranking must agree with the full-train ranking at **Kendall-τ ≥ 0.7**
    (also report Spearman). The 0.5 % check is *precision*, not *rank
    correctness*: if the proxy mis-ranks, BO climbs the wrong surface efficiently.
    Below threshold, repair the proxy (epochs / LR / resolution) **before**
    spending search compute. CUDA- and D1-dependent.

---

## Phase 3 — Device-Aware Search (search loop v1)

**Goal:** End-to-end search using the **unmodified** OFA supernet. Produces
a Pareto frontier of `(accuracy, latency_ms)` for the Jetson.

> **Statistical protocol (peer-review R2.2 / P0.3 — lock before spending search
> compute).** Every method comparison runs **≥5 seeds** for *both* BO and the
> same-budget random-search control. Report Pareto **hypervolume** (fixed
> reference point) with across-seed dispersion + an explicit
> **dominance-across-seeds** statement — never a single-run "the front
> dominates". Seed the GP with the random-search evaluations; for batch-EI use an
> explicit diversification (local penalisation / Kriging-believer) so a batch of 4
> doesn't collapse to near-duplicates (R2.3). The candidate budgets (open decision
> **D2**) must be justified against this protocol, not assumed.

### Checkpoints

- **CP 3.1 — Search space encoder**
  - `search/space.py`: encode/decode OFA archs ↔ flat vector for the
    surrogate. Track categorical vs. integer axes.
  - **DoD:** `decode(encode(arch)) == arch` for 100 random archs.

- **CP 3.2 — Evolutionary search baseline**
  - `search/evolution.py`: NSGA-II over `(accuracy, latency_ms)`.
  - 100 generations × 50 population; LUT cost + short-FT accuracy.
  - **DoD:** Frontier has ≥ 10 non-dominated points.

- **CP 3.3 — Bayesian Optimization**
  - `search/bo.py`: GP surrogate with a structured kernel
    (Hamming + Matérn), Expected Improvement acquisition.
  - **DoD:** over ≥5 seeds at a fixed budget, BO's Pareto **hypervolume**
    exceeds the same-budget random-search control's with non-overlapping
    across-seed dispersion (dominance-across-seeds, per the protocol above) —
    not a single-run `accuracy − λ·latency` comparison.

- **CP 3.4 — TPE fallback (Optuna)**
  - `search/tpe.py`: Optuna wrapper for conditional parameters.
  - Alternative when the GP scales poorly.
  - **DoD:** Same dominance test as CP 3.3.

- **CP 3.5 — Winner v1 exported**
  - Serialize best `α*` to `state/winner_v1/` (arch + weights + log).
  - **DoD:** Can reload in a clean Python session and reproduce cached
    accuracy within noise.

### References
- NSGA-II: Deb et al., 2002.
- GP + Hamming kernel for NAS (NASBOT): Kandasamy et al., NeurIPS 2018.
  https://arxiv.org/abs/1802.07191
- Optuna: https://optuna.org/

---

## Phase 4 — Net2Net Operator Library

**Goal:** A tested library of function-preserving transforms — the
substrate for both local BO warm-starts AND Phase 5's block injection.

> **Note on latency.** Net2Net transfers *weights*, not architecture. The
> target architecture (and therefore its LUT-predicted latency) is decided
> by BO or the expansion protocol. Net2Net never independently grows a
> model's latency — it only reduces the fine-tune cost of reaching a
> target the LUT-aware search has already approved.

### Checkpoints

- **CP 4.1 — Net2Wider**
  - `net2net/wider.py`: widen a Conv or Linear by integer factor.
  - Unit test: outputs on random input match pre/post-widen to 1e-5.
  - **DoD:** `pytest net2net/tests/test_wider.py` green.

- **CP 4.2 — Net2Deeper**
  - `net2net/deeper.py`: insert an identity-initialized Conv/Linear.
  - Unit test: forward output unchanged.
  - **DoD:** Same as CP 4.1.

- **CP 4.3 — BatchNorm handling**
  - Decide between: freeze BN during first warm-start epoch, OR
    "re-estimate BN" trick (one forward-only pass in train-mode BN).
  - Document the decision.
  - **DoD:** Deepen + BN re-estimation preserves the function within 1e-3.

- **CP 4.4 — Graph diff & applicability checker**
  - `net2net/diff.py`: `α_prev, α_new` → list of Net2Net ops or `None`
    (meaning Net2Net can't get you there).
  - **DoD:** Covers all widen/deepen deltas in the OFA space.

### References
- Net2Net: Chen et al., ICLR 2016. https://arxiv.org/abs/1511.05641

---

## Phase 5 — Supernet Expansion

**Goal:** Grow the base OFA supernet to include modern operator choices,
so Phase 7's search draws from a richer space.

### Why expansion doesn't force latency to rise

Three safeguards act in series:

1. **Elasticity ≠ forced cost.** The supernet is elastic (per-block choice
   of op, kernel, width, depth). Expansion adds *options*, not required
   capacity. A wider / richer supernet does not produce a wider or slower
   deployed model — search can still pick the narrow, fast subnet.
2. **LUT pre-screening (CP 5.0).** Before an op is injected, the LUT is
   queried at the exact `(in_c, out_c, resolution)` positions it would
   replace. If the candidate is Pareto-dominated by an incumbent op
   (strictly worse latency AND no parameter-efficiency edge) at every
   position, it is rejected before any fine-tune cost is paid.
3. **Search-time penalty.** Phase 7's `J(α) = accuracy − λ·latency − …`
   naturally deprioritises any injected op whose accuracy gain doesn't pay
   for its latency. Even if CP 5.0 misses a dominated op, the search
   won't pick it.

Expansion is cheap (code + an LUT query); the expensive step is fine-tuning
the expanded supernet (Phase 6). CP 5.0 exists specifically to avoid
paying that fine-tune cost for ops the LUT already rules out.

### CP 5.0 — LUT pre-screening of candidate ops

- `expand/screen.py`: for each candidate op type, enumerate the
  `(in_c, out_c, resolution)` positions it would occupy in the expanded
  supernet. Query the LUT at each position; build a Pareto plot of
  `(latency_ms, params)` against the incumbent MBConv choices.
- **Reject** a candidate if, at *every* position, it is Pareto-dominated
  by an incumbent op.
- **Keep** a candidate if it opens a new Pareto point anywhere the search
  is likely to care about (e.g., FusedMBConv tends to beat MBConv at early
  high-resolution stages where the expand ratio hurts latency).
- **DoD:** For each candidate family (FusedMBConv, ConvNeXt, MobileViT),
  emit a screening report: kept vs. rejected positions, with the
  dominating incumbent op cited per rejection.

### v1 — Within-family expansion (low risk, high reuse)

- **CP 5.1 — Widen the width-multiplier grid**
  - Use Net2Wider to seed `w=1.4, 1.6` supernets from `w=1.2`.
  - Sanity: at seed time, `w=1.4` matches `w=1.2` function; then fine-tune.
  - **LUT envelope check** before paying fine-tune cost: query the LUT at
    the widened `(in_c, out_c)` points. If the widest subnet's summed
    latency exceeds ~2× the Phase 3 winner's latency, no realistic search
    run will ever pick it — skip that width tier.
  - **DoD:** `w=1.4` expanded subnet reaches OFA's published `w=1.4`
    number within 2 %.

- **CP 5.2 — Extend kernel / expand grid**
  - Add `k=9`, `e=8` as optional choices; seed from `k=7, e=6`.
  - **DoD:** Supernet still samples + fine-tunes stably.

### v2 — Cross-family expansion (the ambitious part)

- **CP 5.3 — Pick 1–3 SOTA block types to inject**
  - Candidates:
    - **FusedMBConv** (EfficientNetV2): same family, very safe.
    - **ConvNeXt-style**: large-kernel depthwise + LN + GELU + 2-layer PW.
    - **MobileViT block**: local MB conv + tiny transformer.
  - Criteria: proven for image tasks, has a known-good PyTorch impl,
    LUT-measurable (add to `catalog/` first), **and survives CP 5.0
    screening**.
  - **DoD:** Each chosen block has a row in `data/lut.jsonl` and appears
    as "kept" (at ≥ 1 position) in its CP 5.0 screening report.

- **CP 5.4 — Insertion protocol**
  - Per OFA stage, decide **where** injected blocks can replace native
    MBConv choices.
  - Express as additional values in the arch_dict's per-stage "op" field.
  - **DoD:** `sampler.sample(expanded_arch)` forwards on (1,3,224,224).

- **CP 5.5 — Function-preserving initialization**
  - For each new op type, initialize so `new_op(x) ≈ native_op(x)` at
    injection time:
    - Net2Net-style identity decomposition where possible.
    - BN/layer re-scaling where the shape changes cleanly.
    - **Learned soft gate starting near zero** for ops with no
      function-preserving path (e.g., MobileViT self-attention).
  - **DoD:** Day-0 accuracy of the expanded supernet within 1 % of base.

### Risks
- **No free lunch for cross-family expansion.** MobileViT has attention;
  ConvNeXt uses LayerNorm. Net2Net only covers conv/linear widen/deepen.
  CP 5.5's soft-gate path is the riskiest step.
- **LUT coverage.** Every new op must be measured on the Jetson before
  search can use it. Budget time for `run_sweep.py` extensions.

### References
- EfficientNetV2 (FusedMBConv): Tan & Le, ICML 2021.
  https://arxiv.org/abs/2104.00298
- ConvNeXt: Liu et al., CVPR 2022. https://arxiv.org/abs/2201.03545
- MobileViT: Mehta & Rastegari, ICLR 2022. https://arxiv.org/abs/2110.02178
- BigNAS (related: single "universal" supernet): Yu et al., ECCV 2020.
  https://arxiv.org/abs/2003.11142
- NAT (Neural Architecture Transfer): Lu et al., TPAMI 2021.
  https://arxiv.org/abs/2005.05859

---

## Phase 6 — Expanded Supernet Fine-tuning

**Goal:** Re-co-adapt the shared weights after expansion so injected ops
fit in alongside native ones.

### Checkpoints

- **CP 6.1 — Sandwich-rule training loop**
  - Each step: sample {max-arch, min-arch, 2 random-archs}, forward each,
    accumulate grads, step.
  - **DoD:** Training loss decreases monotonically over 1 epoch on the
    target task.

- **CP 6.2 — Light progressive shrinking**
  - OFA's full PS is 180+ GPU-days. We don't have that.
  - Shrink only the newly added axes, keep depth fixed.
  - **DoD:** Accuracy gap between max and min arch closes by ≥ 50 %
    vs. day-0.

- **CP 6.3 — Expanded-supernet checkpoint**
  - `state/expanded_supernet_v1.pt` + manifest of included ops.
  - **DoD:** Load + sample + forward in a clean session.

### References
- Sandwich rule: BigNAS (above).
- AttentiveNAS: Wang et al., CVPR 2021.
  https://arxiv.org/abs/2011.09011

---

## Phase 7 — Search on the Expanded Supernet

**Goal:** Re-run Phase 3's search over the enriched space. Compare to
Phase 3's winner.

### Checkpoints

- **CP 7.1 — Search-space encoder update**
  - Extend `search/space.py` for the new op choices.
  - **DoD:** Round-trip test still passes.

- **CP 7.2 — Full search run**
  - 200 candidates via BO (or 300 via NSGA-II).
  - LUT cost + short-FT accuracy.
  - **DoD:** Expanded Pareto frontier **weakly dominates** Phase 3's
    frontier. At worst: matches; at best: strictly dominates for some
    latency budgets.

- **CP 7.3 — Winner α* (expanded)**
  - Long-train (full schedule, not 5 epochs) the top candidate(s).
  - **DoD:** Long-trained accuracy within 1.5 % of short-FT prediction.

---

## Phase 8 — Knowledge Distillation

**Goal:** Train the search winner α* (the student) to its maximum accuracy
against an external SOTA teacher, producing the final deployable weights. This
is the project's **one full-schedule training run** — every accuracy number
before this point is a 5-epoch *proxy* used only to *rank* candidates
(CP 2.4 / 3.2 / 7.2). KD against a strong teacher is the standard,
highest-accuracy-per-epoch way to do that final train (OFA, BigNAS, AttentiveNAS
all distill the final model).

> **Note on latency (mirrors Phase 4).** KD transfers *knowledge into weights*;
> it never touches the graph. α*'s architecture — and therefore its LUT-summed
> latency — is unchanged by distillation. The whole LUT contract and Phase 9's
> ≤ 15 % export bar carry over untouched; only accuracy moves.

> **Relationship to CP 7.3 (no redundancy).** CP 7.3's plain hard-label
> long-train is the **baseline**; Phase 8's KD run is the definitive, shipped
> train that must beat it. The two are a baseline/treatment pair, not duplicate
> work.

### Checkpoints

- **CP 8.1 — Teacher selection & caching**
  - Choose the external SOTA teacher that matches the target task (open
    decision **D1**): classification → e.g. ConvNeXt-L / EfficientNetV2-L
    (`timm`); segmentation → e.g. SegFormer-B5; detection → a strong detector.
  - `distill/teacher.py`: load it **frozen, eval-mode**; expose `teacher(x)`.
  - Pin URL + SHA256 in `distill/README.md` — same discipline as the OFA
    checkpoint pin in `supernet/download_ofa.py`.
  - **DoD:** the teacher reproduces its published metric on a small val subset
    within tolerance; its checkpoint hash is pinned.

- **CP 8.2 — Distillation loss & harness**
  - `distill/distill.py`: `L = α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T))
    + (1−α)·CE(z_s, y)` (per-pixel KL for seg; logit/feature mimicking for det).
  - Reuse `eval/`'s D1 data pipeline + metrics so splits/transforms match the
    short-FT harness. Full cosine LR schedule, fixed seed, temperature `T`,
    weight `α`.
  - **DoD:** a tiny overfit run (a few hundred images) drives KD loss down
    monotonically, with the KL and CE terms logged separately. (The loss/harness
    is CPU-testable on tiny tensors; the real run needs CUDA.)

- **CP 8.3 — Full distillation run on the winner**
  - Load α* from `state/winner_v1/` (Phase 3) or the expanded winner
    (Phase 7 CP 7.3); initialize the student from its searched / Net2Net
    warm-started weights (**not** random); train the full schedule with the
    teacher's soft targets.
  - **DoD:** the distilled student beats the CP 7.3 plain long-train baseline by
    a margin > noise (e.g. ≥ +0.3 % top-1 / mIoU / mAP) at the **same**
    architecture and latency.

- **CP 8.4 — Distilled-winner artifact**
  - Serialize distilled weights + training log + teacher pin to
    `state/winner_distilled/` (mirrors `state/winner_v1/`).
  - **DoD:** reload in a clean session and reproduce the distilled accuracy
    within noise. This artifact — not the plain winner — is the input to Phase 9.

### References
- Knowledge distillation: Hinton, Vinyals & Dean, *Distilling the Knowledge in
  a Neural Network*, NeurIPS-W 2014. https://arxiv.org/abs/1503.02531
- Dense-prediction KD (if D1 is seg/det): Liu et al., *Structured Knowledge
  Distillation for Dense Prediction*, CVPR 2019.
  https://arxiv.org/abs/1903.04197

### Risks
- **CUDA.** This is a real, full-length training run — it inherits the
  documented CUDA blocker (resolve before CP 2.4 / this phase). The loss and
  harness can be unit-tested on CPU; the run cannot.
- **D1 coupling.** The teacher must exist for the chosen task/dataset — a
  classification teacher won't transfer to seg/det. Pick the teacher *after* D1
  is resolved.

---

## Phase 9 — Deployment Packaging

**Goal:** Everything a deployment engineer needs to drop the **distilled**
winner on a Jetson.

### Checkpoints

- **CP 9.1 — ONNX → TensorRT engine**
  - Export the **distilled** winner (`state/winner_distilled/`, CP 8.4);
    reuse `lut/export/to_onnx.py` + `lut/bench/build_engine.py` patterns.
  - **DoD:** TRT engine deserializes and runs on the Jetson.

- **CP 9.2 — End-to-end latency validation**
  - Measure the deployed engine on the Jetson.
  - Compare to LUT-summed prediction (unchanged by KD — same architecture).
  - **DoD:** Error ≤ 15 % (same bar as CP 2.2).

- **CP 9.3 — Deployment bundle**
  - `deploy/<date>/engine.plan`, `model_card.md`, `device_info.json`,
    `arch.json`, `training_log.md`. The `model_card.md` records the
    distillation teacher + KD hyperparameters (T, α, schedule).
  - **DoD:** A colleague with zero context can run the engine from the
    bundle alone.

---

## Cross-cutting practices

- **Per-session resume note.** At the end of each session, append to
  `state/SESSION_LOG.md`:
  - What I finished (checkpoint ID).
  - What I'm about to start.
  - Any non-obvious state on disk (cached files, env vars).
- **Reproducibility.** Every training run pins seed, torch version, and
  OFA checkpoint hash in its log.
- **Never re-measure on Jetson during search.** The LUT is the contract.
  Only re-measure when validating CP 2.2 / CP 9.2 assumptions or adding a
  new block.
- **Idempotent scripts.** Same design as `run_sweep.py`: re-running picks
  up where it stopped.

## Open decisions (revisit at their phase)

- **D1 — Target dataset** (blocks CP 2.4 onward; also selects the Phase 8
  distillation teacher).
  Options: ImageNet (generic, well-aligned to OFA pretraining but doesn't
  match deployment workload); Cityscapes / ADE20K (seg, matches LUT's
  seg heads); COCO (det, matches LUT's det heads).
  The chosen dataset/task also determines which external SOTA teacher is
  available for CP 8.1 — a classification teacher won't transfer to seg/det.
  **Decide before CP 2.4.**

- **D2 — Search-budget target** (blocks CP 3.2 / 7.2).
  Default: 100 candidates for Phase 3, 200 for Phase 7.

- **D3 — Which SOTA blocks to inject** (blocks CP 5.3).

  Injection candidates fall into two Pareto-push directions, and the LUT
  screen (CP 5.0) treats them symmetrically:

  - **Accuracy-push** (open Pareto points at the high-latency end of the
    frontier): FusedMBConv, ConvNeXt-style, MobileViT. Worth it when the
    deployment budget has slack.
  - **Latency-push** (open Pareto points at the low-latency end):
    GhostConv, channel-shuffle / partial-conv, 1×1 bottlenecks without
    expand. Worth it when you're already bumping the MobileNetV3 floor.

  The OFA-MBv3 starting space *already* covers most of the
  latency-push region — MBConv with `e=3` and `w=0.75` is a thin
  DWSep-style bottleneck. CP 5.0 will likely reject "pure DWSep with no
  expand" as dominated by MBConv at `e=3`. Latency-push injection only
  pays off for ops OFA structurally cannot reach (e.g., GhostConv's
  cheap-linear half-channel trick; channel-split).

  **Injection order:**
  1. **FusedMBConv first** (accuracy-push at early high-res stages;
     cheapest CP 5.5 because same family).
  2. **GhostConv (or similar) only if** Phase 3's winner frontier sits at
     the low-latency wall — i.e., `α*` already picks `e=3`, `w=0.75`, and
     still isn't fast enough for the deployment budget.
  3. **ConvNeXt / MobileViT** deferred until Phase 3 shows the deployment
     budget leaves headroom for accuracy-push.

  Rule of thumb: run Phase 3 first, look at where `α*` lands on the
  latency axis, then pick the injection family that extends the frontier
  in *that* direction. Don't inject both directions speculatively —
  fine-tune budget (Phase 6) is the bottleneck, not supernet size.

- **D4 — λ, μ in `J(α)`** (blocks CP 3.3 onward).
  Calibrate via two anchor points: "MobileNetV3-large at X ms" and
  "EfficientNet-B0 at Y ms"; fit λ so both lie on the same iso-J contour.

- **D5 — Multi-device extension** (v3, out of scope for v1/v2).
  Current LUT is Jetson-only. Future: per-device LUTs + a
  device-conditioned GP.

## Timeline estimate

Rough, ~4 hours per session:

| Phase | Sessions | Dominant cost |
|---|---|---|
| 1 | 2–3 | OFA setup/debug |
| 2 | 2–3 | Dataset prep + additivity calibration |
| 3 | 3–4 | BO tuning, long-train winner |
| 4 | 2–3 | Net2Net unit tests |
| 5 | 4–6 | Cross-family injection, highest uncertainty |
| 6 | 2–4 | Fine-tune compute |
| 7 | 2–3 | Search + long-train |
| 8 | 2–3 | Distillation: teacher setup + full train |
| 9 | 1–2 | TRT export, bundle |
| **Total** | **20–31 sessions** | |

## Non-goals (whole project)

- Training an OFA supernet from scratch.
- Cross-device LUT transfer (until v3).
- Beating SOTA absolute accuracy — target Pareto dominance over
  MobileNetV3 / EfficientNet-B0 at Jetson latency.
- Online / real-time NAS. Offline: search, commit, deploy.
