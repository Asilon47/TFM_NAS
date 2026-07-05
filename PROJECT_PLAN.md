# Hardware-Aware NAS — Execution Plan

A living, checkpoint-driven plan for turning the Jetson LUT into a
device-adaptive NAS pipeline built on a weight-sharing supernet; after the
Phase-3 search, the winner is refined at the graft seam and compressed
(the 2026-07-05 pivot — supernet expansion descoped, see Phase 5 and
procedure.md "Plan pivot").

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
  [P4] Net2Net ops  ──▶ [P5] Graft-interface ablation (adapter init / nano-neck)
                               │
                               ▼
                        winner-v1.5 (accuracy lever)
                               │
                               ▼
                        [P6] Structured pruning (DepGraph) → winner-v2
                               │
                               ▼
                        [P7] Recipe-parity long-train (the honest baseline)
                               │
                               ▼
                        [P8] Knowledge Distillation
                             (bigger yolo11-pose teacher → student)
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
    `ofa_mbv3_d234_e346_k357_w1.0` (or `w1.2`) into
    `<project_root>/.cache/ofa/` (relocated 2026-06-16 from
    `~/.cache/ofa/` — see procedure.md "Cache relocation").
  - `supernet/README.md` documents which checkpoint is pinned
    (URL + SHA256). Future-you will thank present-you.
  - **DoD:** Checkpoint file exists on disk, hash matches the pin.

- **CP 1.3 — Subnet sampler** ✅
  - `supernet/sampler.py`: `sample(arch_dict) -> nn.Module`.
  - Accepts canonical OFA arch spec (`ks=[...]`, `e=[...]`, `d=[...]`).
  - **DoD:** `sampler.sample(random_arch)` forwards a `(1, 3, 224, 224)`
    tensor without error.

- **CP 1.4 — ImageNet sanity** ✅
  - On a 50k ImageNet-val run (Kaggle GPU), confirm the OFA weight load + BN
    recalibration are intact via rank fidelity across a spread of archs.
  - **DoD:** Spearman ρ ≥ 0.85 between OFA's accuracy predictor and measured
    top-1 across ≥ 20 archs (max/min corners + random interior).
    **RESULT: ρ = 0.919, p = 1.1e-08; max-arch = 77.3 % top-1 confirms weight
    load intact. CLOSED 2026-06-18.** See `procedure.md` "CP 1.4 CLOSED".

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

- **CP 2.1 — Arch → block list translator** ✅
  - `search/arch_to_blocks.py`: OFA arch_dict + macro-topology →
    ordered `(block_type, cfg, input_shape)` list (LUT keys).
  - Correctly propagate stride/resolution so each block's `input_shape` is
    right.
  - **DoD:** 10 random archs — every emitted tuple has a matching `row_key`
    in `data/lut.jsonl`.

- **CP 2.2 — LUT composite-cost function** ✅
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
  - **RESULT (33 subnets, depth 11–21, 2026-06-17):** all bins +6.8–9.2 %
    (aggregate +7.9 %), none breached 15 %; bias flat in depth (fusion = constant
    multiplicative discount, not depth-exploding). Predictor fidelity: Spearman
    ρ = 0.991, Kendall τ = 0.943. Calibration fit: measured ≈ 0.934·summed
    (MAPE 7.9 % → 1.0 %). **CLOSED 2026-06-17.** See `procedure.md` "CP 2.2 closed".

- **CP 2.3 — Additivity correction (conditional)** — **SKIPPED**
  - **SKIPPED 2026-06-17:** CP 2.2's depth-binned DoD passed in all bins
    (worst +9.2 %, aggregate +7.9 %, bias flat in depth). The trigger condition
    ("any depth bin > 15 %" or upward-with-depth residual) was **not met**.
    The pre-registered escalation path remains documented below in case a future
    re-sweep (e.g. at 640 resolution) triggers it.
  - **Trigger (pre-registered — peer-review R4.2 / P1.8):** CP 2.2's by-depth
    report breaches the 15 % bar in **any** depth bin, **or** the residual trends
    upward with depth (the cross-block-fusion signature). Do not wait for the
    aggregate to miss — by then the deep regime is already mispriced.
  - Then: fit a residual GP, `δ = measured − summed` as a function of
    `(depth, total_flops, n_dw)`.
  - **DoD:** Corrected error < 10 % on held-out nets.

- **CP 2.4 — Eval harness (short fine-tune)** ✅ CLOSED 2026-06-27 (reframe gate)
  - `eval/shortft.py`: short (≈5-epoch) fine-tune on the target task
    (D1 = gate-pose), fixed seed, fixed LR schedule. With the OFA backbone
    grafted under a YOLO11-pose head (`detect/pose_model.py`), this is a short
    Ultralytics pose fine-tune of the candidate.
  - Returns **pose mAP (OKS)** (D1 = gate-pose), reusing Ultralytics' pose
    validator (`detect/evaluate.py`) rather than re-implementing OKS.
  - **DoD — proxy-rank fidelity, REFRAMED (gates the whole search — peer-review
    R2.1 / P0.2):** fully train ≈8–12 architectures spanning the space; the
    proxy ranking must agree with the full-train ranking at **Spearman ρ ≥ 0.70
    AND top-1 regret ≤ 0.01** (`eval/shortft.py:rank_verdict`). The original
    **Kendall-τ-on-10 + reproducibility-Δ≤0.005** gate was **superseded**
    (2026-06-27): τ-on-10 has wide CIs at n=10 and punishes mid-rank
    disagreements the search ignores — it mis-measures (size descriptors fail τ
    yet have regret 0 / pick the true best). Kendall-τ, precision@k, and the
    run-to-run reproducibility-Δ ride along as reported *diagnostics*. Below
    threshold, repair the proxy (head warm-start / epochs / LR) **before**
    spending search compute. CUDA- and D1-dependent.
  - **Status (CLOSED 2026-06-27):** first GPU run failed both old DoDs
    (τ=0.20) → root cause = randomly-initialized Pose head (LP-FT distortion).
    Repair = warm-start + **freeze** a trained gate head, average over 3 seeds.
    Warm-head re-test (Colab, `data/cp24_warmstart.json`): **Spearman ρ=0.77,
    top-1 regret 0.0 → reframe gate PASS** (τ=0.60, repro-Δ=0.0145 as
    diagnostics). Cross-checked by the no-GPU zero-cost ranker (`eval/zerocost.py`:
    depth_sum ρ=0.843, regret 0). Carries into Phase 3: warm-head proxy = accuracy
    signal, zero-cost = free cold-start prefilter; J(α) λ/μ deferred to CP 3.3 (D4).
    See `procedure.md` "CP 2.4 CLOSED — warm-head re-test + reframe gate".

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
  - 200 generations × 150 population; **depth_sum (zero-cost) + LUT latency** — a CPU-free
    structural baseline (*not* short-FT; the accuracy axis is the structural prior, see D2).
    (pop=150 reaches the true global front; pop=50 under-converges regardless of n_gen —
    see procedure.md "CP 3.2 ... Convergence".)
  - **DoD:** Frontier has ≥ 10 non-dominated points.

- **CP 3.3 — Bayesian Optimization**
  - **Status (2026-06-28): buildable slice BUILT, still OPEN.** `search/bo.py` done
    (ParEGO + BoTorch `MixedSingleTaskGP` + `qLogEI`; CPU smoke beats random, DoD PASS);
    @640 catalog + Jetson `bench_model.py`/runbook + `kaggle/` push automation prepared.
    CLOSE awaits the Jetson @640 sweep + yolo11n-pose baseline, then the Kaggle 5-seed
    warm-head BO-vs-random hypervolume DoD. See `procedure.md` "CP 3.3 — buildable slice".
  - `search/bo.py`: GP surrogate with a structured kernel
    (Hamming + Matérn), Expected Improvement acquisition.
  - **Objective (D4, RESOLVED):** multi-objective `(acc_eff, latency_ms)` with a hard ceiling
    `latency ≤ T_max = min(baseline, fps_to_ms(60)=16.7 ms)`; soft μ² memory penalty folded into
    `acc_eff` (budget 512 MiB, ≡0 in v1); λ ParEGO-sampled, calibrated at selection via two-anchor
    iso-J. Formula locked in `search/objective.py`; λ/μ numbers set here (need the @640 latency scale).
  - **Budget (D2):** B=50/run, n_init=20 → ~400 warm-head fine-tunes across 5 seeds
    (incl. the same-budget random-search control; 1 seed/eval). Step 0 = one timed
    calibration eval on Colab (no per-eval wall-clock is recorded yet).
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
substrate for Phase 5's graft-interface variants (warm adapter init,
identity-gated fusion) and Phase 6's post-prune recovery (BN
re-estimation). (Originally also the substrate for supernet block
injection — descoped 2026-07-05.)

> **Note on latency.** Net2Net transfers *weights*, not architecture. The
> target architecture (and therefore its LUT-predicted latency) is decided
> by BO or the expansion protocol. Net2Net never independently grows a
> model's latency — it only reduces the fine-tune cost of reaching a
> target the LUT-aware search has already approved.

### Checkpoints

- **CP 4.1 — Net2Wider**
  - `net2net/wider.py`: widen a Conv or Linear by integer factor.
  - Unit test: outputs on random input match pre/post-widen to 1e-5.
  - **DoD:** `pytest tests/test_wider.py` green (tests live flat in
    `tests/` — the original `net2net/tests/` path predates that convention).

- **CP 4.2 — Net2Deeper**
  - `net2net/deeper.py`: insert an identity-initialized Conv/Linear.
  - Unit test: forward output unchanged.
  - **DoD:** Same as CP 4.1.

- **CP 4.3 — BatchNorm handling**
  - Decide between: freeze BN during first warm-start epoch, OR
    "re-estimate BN" trick (one forward-only pass in train-mode BN).
  - Document the decision.
  - **DoD:** Deepen + BN re-estimation preserves the function within 1e-3.

- **CP 4.4 — Graft-seam applicability (re-scoped 2026-07-05)**
  - Was: a graph diff over the OFA space for BO warm-starts — obsolete
    (Phase 3 closed without Net2Net warm-starts; no second search exists).
  - Now: `net2net/graft_init.py`: `identity_embed_conv1x1_(conv)` —
    initialize the ChannelAdapter's 1×1 convs (40→64, 112→128, 160→256) as
    the identity on the first `in_c` channels + Net2Wider-style replicated
    extras, replacing random init. Wired as
    `build_grafted_pose_model(adapter_init="net2wider")`.
  - **DoD:** the adapter output's first `in_c` channels equal its input
    exactly (unit test). Documented as an initialization *prior*, not
    end-to-end function preservation (the frozen donor head's expectations
    cannot be matched by any adapter init).

### References
- Net2Net: Chen et al., ICLR 2016. https://arxiv.org/abs/1511.05641

---

## Phase 5 — Graft-Interface Ablation → winner-v1.5 (was: Supernet Expansion — descoped 2026-07-05)

**Goal:** Close the accuracy seam the Phase-3 audit exposed. The deployed
graft is *neck-less*: three independently random-initialized 1×1 convs
bridge the backbone taps (40, 112, 160) straight into the Pose head with no
cross-scale fusion — while the yolo11n-pose baseline carries a full PAN-FPN.
CP 2.4 already proved this interface dominates the proxy signal (random
head → τ=0.20; warm+frozen head → ρ=0.77). Phase 5 ablates
function-preserving interface upgrades around the **fixed winner-v1
backbone** and selects **winner-v1.5** under the honest end-to-end ceiling.

> **Why this replaces Supernet Expansion** (user decision 2026-07-05,
> AskUserQuestion; procedure.md "Plan pivot"; D3 → descoped):
> (1) the old Phase 5/6 DoDs were ImageNet-framed — written before the D1
> pose pivot ("OFA's published w=1.4 number", 224² forwards, "day-0
> accuracy of the expanded supernet"); (2) Phase 6's expanded-supernet
> fine-tune was the compute bomb on free-tier GPUs ("fine-tune budget is
> the bottleneck, not supernet size" — the old D3 note; original OFA
> training ≈ 1,200 V100-hours); (3) decisive: the warm-head proxy's noise
> floor (σ ≈ 0.005–0.025) already exceeds the de-noised frontier's
> top-cluster gaps (~0.014), so an enriched space adds options the oracle
> cannot rank — CP 3.5's winner's curse was this saturation showing
> itself. Expansion stays future work; CP 5.0 below keeps the evidence
> trail without spending any training.

### Checkpoints

- **CP 5.0 — FusedMBConv LUT screen (OPTIONAL, evidence-only)**
  - The old CP 5.0 idea, kept as *descope evidence*: add a FusedMBConv
    builder to `catalog/` (append-only grid rows @640 at the early-stage
    positions it would occupy), one idempotent `run_sweep` session, then
    `expand/screen.py` emits the `(latency_ms, params)` Pareto report vs
    the incumbent MBConv rows.
  - **DoD:** screening report with kept/rejected positions — a future-work
    exhibit for the thesis, no training spent. Piggybacks on any Nano
    session; skippable without blocking anything.

- **CP 5.1 — Variant library (the code)**
  - `detect/neck.py`: `ZeroGatedTopDownNeck` — P5 →1×1→ upsample →
    add-into-P4, P4′ →1×1→ upsample → add-into-P3, each add scaled by a
    **zero-initialized learnable gate**, so at init the neck is exactly the
    identity (day-0 function preserved — same trick the old CP 5.5 planned
    for attention blocks). Optional bottom-up path for V3.
  - `GraftedPoseModel(neck=…)`: `neck=None` keeps the 3-module Sequential
    (existing state_dicts load unchanged); with a neck, `_predict_once`
    generalizes and `model[-1]` stays the Pose head (the `v8PoseLoss`
    contract).
  - `build_grafted_pose_model(adapter_init="net2wider")` wires CP 4.4.
  - **DoD:** unit tests — neck-at-init output == no-neck output exactly;
    loss/backward smoke; `model[-1] is head` with and without a neck.

- **CP 5.2 — Warm-head proxy ablation**
  - `eval/graft_ablate.py`: variants **V0** control (current graft) /
    **V1** net2wider adapter init / **V2** V1 + top-down neck / **V3**
    (optional — only if V2 beats V1 by >1σ) + bottom-up. Per
    (variant, seed ∈ {1,2,3}): the exact CP 3.5 warm-head protocol
    (5 epochs, frozen gate-head donor, imgsz 640, batch 16), resumable
    jsonl cache namespaced `graft_ablate_e5_r640`.
  - Kaggle: `MODE="graft_ablate"` in `kaggle/run.py` (clone of the denoise
    block; ~12 fine-tunes ≈ one session).
  - **DoD:** the cache holds every (variant, seed) mAP; report mean ± σ
    per variant.

- **CP 5.3 — End-to-end bench + full-FT + selection**
  - Bench V2/V3 graphs end-to-end on the Orin Nano (V1 shares V0's graph —
    no new bench); 100-epoch bare-AdamW full-FT of the **top-2** variants
    on the AGX — apples-to-apples with winner-v1's 0.841 reference.
  - Select **winner-v1.5** ceiling-first under the *measured end-to-end*
    T_max → `state/winner_v1_5/` (winner record + state_dict weights).
  - **DoD:** the winner record carries e2e latency, full-FT mAP, and the
    selection rationale; the pick is user-confirmed (CP 3.5 precedent).

### Risks
- **The neck spends latency headroom.** That is why the Stage-0 measured
  e2e margin (procedure.md "Plan pivot") gates the design, and why the
  fusion gates start at zero (a neck the data doesn't need can stay off).
- **Zero gates may stay near zero under a 5-epoch proxy.** Report gate
  magnitudes alongside the proxy mAPs; the full-FT of the top-2 is the
  decider, not the proxy alone.

### References
- Net2Net: Chen et al., ICLR 2016. https://arxiv.org/abs/1511.05641
- PANet (the fusion the baseline has and the graft lacks): Liu et al.,
  CVPR 2018. https://arxiv.org/abs/1803.01534
- Zero-init residual gating (ReZero): Bachlechner et al., UAI 2021.
  https://arxiv.org/abs/2003.04887

---

## Phase 6 — Structured Pruning → winner-v2 (was: Expanded Supernet Fine-tuning — descoped 2026-07-05)

**Goal:** The latency lever. Channel-level compression reaches off-grid
widths the OFA space cannot express, recovers whatever headroom Phase 5's
neck spent, and gives the thesis a compression ablation. TensorRT-aligned
throughout (pruned channel counts stay multiples of 16). Pruned networks
leave the LUT grid, so their latency claims are **measured-only**
(end-to-end Nano benches) — never LUT-summed.

### Checkpoints

- **CP 6.1 — DepGraph harness**
  - `torch-pruning>=1.4,<2` pinned in `requirements-nas.txt` (never in
    `requirements.txt` — that venv's `torch==2.3.1+cpu` pin must not be
    disturbed), plus the `kaggle/run.py` pip line and `jetson/Dockerfile`.
  - `prune/prune_graft.py`: DepGraph group pruning over the **trained**
    winner-v1.5 graft; `ignored_layers` = the semantic output convs (last
    conv of each `head.cv2/cv3/cv4` scale + `head.dfl`); group-L2
    importance; `round_to=16`. The head is **unfrozen during recovery** —
    pruning upstream of a frozen consumer corrupts its trained weights
    (DepGraph slices consumer in-channels when producer out-channels
    shrink).
  - **DoD (CPU smoke):** prune 20 % → forward OK, output shapes unchanged,
    params reduced, every pruned conv's channels % 16 == 0.

- **CP 6.2 — Sparsity ladder + recovery + measured curve**
  - ~15 / 30 / 45 % group sparsity; per point:
    `net2net.bn.reestimate_bn` → recovery fine-tune (Kaggle short /
    AGX long) → e2e ONNX export → Nano bench.
  - **DoD:** `data/pruning_curve.json` — 3 points × (recovered mAP,
    measured e2e latency), plus the table in procedure.md.

- **CP 6.3 — Operating point → winner-v2**
  - Pick the deployment point on the measured curve (user decision,
    CP 3.5 precedent) → `state/winner_v2/`.
  - **DoD:** winner-v2 record. If **no** pruned point Pareto-improves
    winner-v1.5, record that honestly and carry winner-v1.5 forward
    unpruned — a negative result here is a valid thesis finding.

### References
- DepGraph (Torch-Pruning): Fang et al., CVPR 2023.
  https://arxiv.org/abs/2301.12900

---

## Phase 7 — Recipe-Parity Training (was: Search on the Expanded Supernet — descoped 2026-07-05)

**Goal:** Make the final accuracy comparison honest. The winner-v1 full
fine-tune (0.841) used this repo's bare-AdamW loop while the 0.877 baseline
was trained with the full Ultralytics recipe (SGD + warmup + cosine + EMA +
augmentation schedule) — that gap is confounded by schedule, not just
architecture. Phase 7 long-trains the refined winner under a parity recipe;
its number is the plain-training reference Phase 8's KD must beat (the old
CP 7.3 ↔ CP 8.3 baseline/treatment pairing, preserved as CP 7.2 ↔ CP 8.3).

### Checkpoints

- **CP 7.1 — Parity trainer**
  - `eval/recipe_ft.py`: SGD (momentum 0.937, nesterov) with no-decay
    BN/bias param groups, 3-epoch warmup, cosine decay to `lrf`, EMA,
    `close_mosaic` for the final epochs; reuses `eval/shortft.py`'s
    loader/preprocess/eval plumbing. Checkpoints are **state_dict-only**
    (`GraftedPoseModel` is function-local — whole-model pickling crashes;
    the Ultralytics Trainer is deliberately NOT wrapped).
  - **DoD:** schedule functions unit-tested; a tiny CPU run steps without
    error.

- **CP 7.2 — Parity long-train (the reference number)**
  - Winner-v2 (plus optionally winner-v1.5 unpruned as control) on the
    AGX, full schedule, fixed seed.
  - **DoD:** the run completes with logged LR/loss/mAP curves; final pose
    mAP recorded with seed + recipe settings. This is the baseline that
    CP 8.3 must beat.

- **CP 7.3 — Honest gap report**
  - procedure.md table: proxy (0.610-class) → bare-AdamW full-FT
    (0.841-class) → parity full-FT → (Phase 8) KD, each against the 0.877
    baseline.
  - **DoD:** the table exists with every cell sourced to an artifact.

---

## Phase 8 — Knowledge Distillation

**Goal:** Train the refined winner (the student) to its maximum accuracy
against a bigger yolo11-pose teacher, producing the final deployable weights.
Together with Phase 7's parity train this is one of the project's two
full-schedule runs — the treatment to CP 7.2's baseline. Every earlier
accuracy number is a short-FT *proxy* used only to *rank* candidates
(CP 2.4 / 3.2 / 5.2) or a protocol-internal reference (CP 5.3's bare-AdamW
full-FT). KD against a strong teacher is the standard,
highest-accuracy-per-epoch way to do that final train (OFA, BigNAS, AttentiveNAS
all distill the final model).

> **Note on latency (mirrors Phase 4).** KD transfers *knowledge into weights*;
> it never touches the graph. α*'s architecture — and therefore its LUT-summed
> latency — is unchanged by distillation. The whole LUT contract and Phase 9's
> ≤ 15 % export bar carry over untouched; only accuracy moves.

> **Relationship to CP 7.2 (no redundancy).** CP 7.2's plain hard-label
> parity long-train is the **baseline**; Phase 8's KD run is the definitive,
> shipped train that must beat it. The two are a baseline/treatment pair —
> same init, same recipe, same seed; the only delta is the teacher term.

### Checkpoints

- **CP 8.1 — Teacher selection & caching**
  - Teacher matched to the target task (**D1 = gate-pose**, resolved): a bigger
    **yolo11-pose** trained on `dataset/`. Candidates: the in-repo
    **yolo11s-pose** (0.8819 measured, `runs/pose/experiments/gate_anchor_yolo11s`)
    vs a newly trained **yolo11m-pose** — measure its gate mAP *before*
    committing (anchor B showed thin headroom: +70 % latency bought +0.5 mAP;
    teacher latency is irrelevant, only its accuracy matters). Loss-shape
    reference: `yolo-ros2-inference/scripts/yolo_distillation.py`.
  - `distill/teacher.py`: load it **frozen, eval-mode**; expose `teacher(x)`.
  - Pin the in-repo checkpoint path + SHA256 in `distill/README.md` — same
    discipline as the OFA checkpoint pin in `supernet/download_ofa.py`.
  - **DoD:** the teacher reproduces its recorded gate mAP on a val subset
    within tolerance; its hash is pinned; the teacher choice is a user
    decision (AskUserQuestion).

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
  - Load the refined winner (`state/winner_v2/` — or `state/winner_v1_5/` if
    pruning didn't pay); initialize the student from the **same warm start as
    CP 7.2** (searched OFA backbone + trained interface, **not** random) and
    train the same parity schedule + seed, with the teacher's soft targets as
    the only delta.
  - **DoD:** the distilled student beats the CP 7.2 parity long-train baseline
    by a margin > noise (≥ +0.3 mAP) at the **same** architecture and latency.

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
    reuse the Stage-0 grafted exporter + `lut/bench/build_engine.py`
    patterns. Build **fp32 and fp16** engines (the deploy target is FP16).
  - **DoD:** both TRT engines deserialize and run on the Jetson.

- **CP 9.2 — End-to-end latency validation**
  - Measure the deployed engine on the Jetson.
  - Compare to the LUT-summed prediction **plus the measured pose
    stem/adapter/head offset** (`data/pose_stem_head_offset.json`, Stage 0;
    `cost(..., res=640, stem_head=…)`) — on the **unpruned** winner-v1.5
    architecture. Pruned / distilled-pruned engines are off the LUT grid,
    so their deployed figure is measured-only.
  - **DoD:** Error ≤ 15 % (same bar as CP 2.2) on the unpruned arch.

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

- **D1 — Target dataset — RESOLVED 2026-06-18 → gate-pose.**
  Target = `dataset/` (Ultralytics YOLO-pose: 1 class `gate`, 8 keypoints,
  synthetic A2RL drone-racing renders; see `dataset/SCHEMA.md`). The task is
  gate detection + 8-keypoint pose (metric: **pose mAP / OKS**), not
  classification. Decisions (AskUserQuestion): the OFA subnet is searched as a
  **backbone** under a YOLO11-pose head (`supernet/pose_backbone.py` +
  `detect/`), keeping the supernet/LUT/Net2Net/BO machinery and using the
  ImageNet pretrain as the backbone warm-start; the **baseline-to-beat** and the
  **Phase-8 teacher** become the deployed **yolo11n-pose** and a bigger
  **yolo11s/m/l-pose** (reusing `yolo-ros2-inference/scripts/yolo_distillation.py`),
  replacing MobileNetV3/ImageNet. **Consequence:** the LUT is keyed per-block at
  res 224; pose runs at 640, so the per-block shapes re-key — a second LUT sweep
  at the deployment resolution is owed (the append-only schema absorbs it). See
  `procedure.md` "D1 resolved — pose pivot".

- **D2 — Search-budget target** — **RESOLVED 2026-06-27 → B=50.**
  Under the "cheap NSGA-II + expensive BO" design the binding budget is the BO **per-run
  budget B**; NSGA-II is CPU-free (`depth_sum` + LUT) and not budget-constrained.
  **CP 3.3: B=50, n_init=20** → total `5·(2B−n_init)` = **400** warm-head 5-epoch proxy
  fine-tunes across 5 seeds — already including the same-budget random-search control,
  whose evals double as the GP's shared init. 1 seed per eval (the GP absorbs proxy noise
  via its nugget term; 3-seed averaging is reserved for the CP 3.5 winner check). The old
  "100 candidates" default was infeasible: read as B=100 the 5-seed protocol costs ~900
  Colab fine-tunes. (The "Phase-7 budget re-decided at CP 7.2" clause is obsolete since the
  2026-07-05 pivot — Phase 7 is now recipe-parity training; there is no second search.)
  See `procedure.md` "D2 RESOLVED".

- **D3 — Which SOTA blocks to inject** — **RESOLVED 2026-07-05 → descoped**
  (user decision, AskUserQuestion; see `procedure.md` "Plan pivot").
  Supernet expansion (the old Phases 5–7) left the plan: the old DoDs were
  ImageNet-framed (pre-dated the D1 pose pivot), the Phase-6
  expanded-supernet fine-tune was infeasible on free-tier GPUs, and the
  warm-head proxy's noise floor (σ ≈ 0.005–0.025) exceeds the de-noised
  frontier's top-cluster gaps (~0.014) — an enriched space is unrankable by
  the oracle that would have to search it. The injection analysis that
  lived here (FusedMBConv first at early high-res stages; GhostConv only at
  the low-latency wall; ConvNeXt/MobileViT deferred) is preserved in git
  history; its own rule of thumb — "run Phase 3 first, look at where α*
  lands, don't inject speculatively; the fine-tune budget is the
  bottleneck" — is exactly what, evaluated on the Phase-3 outcome
  (saturated accuracy, comfortable latency), recommended against paying
  Phase 6. FusedMBConv survives as the optional, evidence-only CP 5.0 LUT
  screen (future-work exhibit, no training).

- **D4 — λ, μ in `J(α)`** — **RESOLVED 2026-06-27 → Pareto + hard latency ceiling.** Search is
  multi-objective `(acc_eff, latency)` bounded by `latency ≤ T_max = min(baseline, 60 FPS→16.7 ms)`;
  the soft μ² memory penalty folds into `acc_eff` (budget 512 MiB; ≡0 across v1). λ is ParEGO-sampled
  while searching and calibrated for the deploy winner via the two-anchor iso-J fit
  ("MobileNetV3-large at X ms" vs "EfficientNet-B0 at Y ms": `λ = Δacc/Δlat`), reported as a
  sensitivity sweep. Formula in `search/objective.py`; λ/μ *numbers* land at CP 3.3 (need the @640
  latency scale). See procedure.md "D4 RESOLVED".

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
| 5 | 2–3 | Graft ablation: proxy runs + Nano benches |
| 6 | 2–3 | Pruning ladder + recovery fine-tunes |
| 7 | 1–2 | Parity long-train (AGX wall-clock) |
| 8 | 2–3 | Distillation: teacher setup + full train |
| 9 | 1–2 | TRT export, bundle |
| **Total** | **17–26 sessions** | |

## Non-goals (whole project)

- Training an OFA supernet from scratch.
- Cross-device LUT transfer (until v3).
- Beating SOTA absolute accuracy — target Pareto dominance over
  MobileNetV3 / EfficientNet-B0 at Jetson latency.
- Online / real-time NAS. Offline: search, commit, deploy.
