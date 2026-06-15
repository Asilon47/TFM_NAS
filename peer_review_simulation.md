# Simulated peer-review panel — TFM: hardware-aware NAS for the Jetson Orin Nano

> **What this is.** A simulated multi-perspective peer review of this thesis,
> produced with the `academic-paper-reviewer` skill. One model enacts five
> disciplined, independent reviewer personas (each writes as if it has not seen the
> others), followed by an editorial synthesis and a prioritised revision roadmap.
> **It is a critique, not an edit:** no thesis source file was modified, and no
> number, citation, or `\gls` term was altered. This document lives outside the
> LaTeX source tree, so it never reaches the PDF.
>
> **Date:** 2026-06-13 · **Calibration (set by the author):** review *frame* =
> **Both** (TFM-tribunal verdict + a publishability addendum); *pending results* =
> **vet the design + pilot** (judge whether the method and experiment are sound and
> will yield defensible results; treat scaffolded sections as a plan to critique,
> not as failures).

---

## Phase 0 — Field analysis

| Dimension | Result |
|---|---|
| Primary discipline | Hardware-aware Neural Architecture Search / efficient on-device deep learning |
| Secondary | GPU & embedded-systems performance engineering · automatic control & robotics (deployment) · ML reproducibility / MLOps |
| Research paradigm | Empirical–experimental + *constructive* (builds a pipeline artefact, then measures it) |
| Methodology type | Design-and-build + on-device latency measurement + Bayesian-optimisation search; controls = random search (same budget) and NSGA-II |
| Venue tier (if mapped) | Methodology/SOTA ≈ solid Q2 edge-/efficient-ML workshop or journal; **not submittable as a paper yet** (5 of 6 objectives have no data) |
| Maturity | **Split** — framing, SOTA and methodology *pre-submission/polished*; Results *scaffolded* (pilot data only; every gap flagged `TODO`) |

**Recommended comparator venues** (for the publishability addendum): *MLSys*;
*IEEE TECS / ACM TECS* (embedded); a *TinyML / edge-AI* workshop. As a degree work
it is an ETSII *Máster en Automática y Robótica* TFM, judged by a tribunal.

**The defining fact for this review.** The thesis cleanly separates a finished
*design/argument* layer (Ch. 1–4) from a pending *data* layer (Ch. 5), and says so
openly ("No value here is invented"). Reviewing it well means judging whether the
experiment is the *right* experiment — not penalising absent numbers.

---

## Phase 1 — Five independent reviews

### Reviewer 1 of 5 — Editor-in-Chief
*Dual hat: ETSII tribunal chair (Automática y Robótica) + associate editor at an efficient-ML venue. Judges significance, fit, and the contribution claim.*

**Summary.** The manuscript proposes and partially builds a hardware-aware NAS
pipeline for one fixed device (Jetson Orin Nano) assembling five mature components
— a device-measured latency LUT, a pretrained OFA supernet, Net2Net warm-starting,
Bayesian-optimisation search, and a final knowledge-distillation train. Framing
(Ch. 1–2), objectives (3), and methodology (4) are polished; results (5) are a
scaffold with one real pilot (three LUT rows), the rest flagged `TODO`.

**On significance and fit.** The problem is real and well-motivated: §1.2
(`sec:intro_problem`) names the two failure modes — a device-independent cost proxy
and the cost of evaluation — and the pipeline answers each. My central concern is
the **contribution claim**, stated disarmingly in §2.8 (`sec:positioning`) and §1.3:
*"None of these components is new on its own… the assembly is the contribution."* I
accept integration as a contribution — but only once it is shown to *work and to
beat the obvious alternative*. Today the assembly is a promise; the abstract/§6
claim ("an evaluation can be made cheap **and** faithful") is a hypothesis with no
result behind it.

**Strengths on record.** (1) The State of the Art (Ch. 2) is strong and current and
does not hide from the field's uncomfortable literature. (2) The reproducibility
discipline (§4.10 `sec:method_repro`; SHA-pinned OFA checkpoint, append-only LUT)
is above the norm. (3) Integrity is exemplary — §5 opens "No value here is
invented," and every missing number is a visible placeholder.

**Concerns at my altitude.**
- **The pivotal scope decision D1 is still open** — target dataset/task, teacher,
  and the deployment latency budget (ms) are unresolved (`TODO` at §1.4, §4.5,
  §4.7, §5.7). Until D1 closes you cannot even state which accuracy metric the
  thesis reports (top-1 vs mIoU vs mAP appears conditionally). A timeline risk.
- **Degree fit.** For an *Automática y Robótica* TFM the robotics connection is
  asserted but no concrete robotic task/latency budget grounds it.

**Recommendation.** *Major revision (TFM frame: on track — sound, defensible design;
execute the empirical program).* For any venue, not yet submittable.

---

### Reviewer 2 of 5 — Peer Reviewer 1 (Methodology: NAS search + Bayesian optimisation)
*NAS-search researcher attuned to the NAS reproducibility crisis. Asks: is the search sound, and can the data support the conclusions?*

**Method under review.** A GP surrogate with a Hamming∘Matérn kernel and Expected
Improvement (batch of four) over a scalarised objective
`J(α) = acc(α) − λ·ℓ(α) − μ·max(0, m(α) − m_budget)²` (§4.3 `eq:objective`), with
TPE/Optuna as fallback, NSGA-II and a same-budget random search as baselines (§4.6
`sec:method_search`).

**What is right, and rare.** You adopt the same-budget random-search control as a
first-class baseline and motivate it from the field's own reckoning — §2.3 cites
`li2020random`, `yang2020`, the NAS-Bench line, then states "a NAS result means
little without a same-budget baseline." Most submissions fail exactly here.

**Major concerns.**
1. **Proxy-rank fidelity is the load-bearing assumption and it is not tested.**
   Every in-loop accuracy is a few-epoch fine-tune (§4.4, §4.7). Your only gate,
   CP 2.4, checks that two proxy runs *agree within 0.5 %* — that is **precision,
   not rank correctness**. The search climbs the *proxy's* front; if proxy rank ≠
   true rank, BO optimises the wrong surface efficiently. You cite the very
   literature documenting this (`li2020random`). *Fix:* fully train ≈8–12
   architectures spanning the space, report Kendall-τ / Spearman vs the proxy, and
   gate the search on an acceptable τ. Add it beside CP 2.4.
2. **"BO dominates random" needs statistics, not an anecdote.** Objective 3 / M3
   reads "the front dominates that baseline at the same budget." Random search has
   high variance; one run each proves nothing. *Fix:* ≥5 seeds per method; report
   Pareto **hypervolume** with dispersion + a dominance-across-seeds statement.

**Medium concerns.**
3. **GP capacity vs budget.** 100 candidates (open decision D2) is thin for a GP to
   learn a structured categorical kernel; batch-EI of 4 needs an explicit
   diversification (local penalisation / Kriging-believer) or the picks collapse to
   near-duplicates. *Fix:* state the batch mechanism; justify 100; seed the GP with
   the random-search evaluations.
4. **Linear scalarisation cannot reach concave Pareto regions.** Sweeping λ traces
   the front only where it is convex. *Fix:* acknowledge it, state the λ grid,
   consider augmented Tchebycheff if the front looks concave.

**Minor.** μ, λ co-calibrated against "two anchor models" (D4) — under-specified;
the front shape depends on it. Encoder round-trip unit-tested (§4.6) — good.

**Recommendation.** *Major revision.* The search design is reasonable and the
baseline discipline is correct; settle the two majors **before** spending search
compute, or the eventual numbers won't support the claims.

---

### Reviewer 3 of 5 — Peer Reviewer 2 (Domain: hardware-aware NAS literature & contribution)
*Senior HW-NAS researcher (OFA, FBNet, ProxylessNAS, nn-Meter). Asks: is the literature complete, and is the contribution genuine?*

**Coverage.** Comprehensive and current. The three-part NAS decomposition
(`elsken2019…`), the search-strategy families, the cheap-estimation ladder (Table
`tab_speedup`), the morphism vs weight-sharing split, the hardware-aware line
(MnasNet → ProxylessNAS → FBNet → nn-Meter → BRP-NAS → HW-NAS-Bench), and the KD
lineage are all present and correctly attributed. The §2.5 (`sec:hwnas`) discussion
of **TensorRT fusion eroding per-operation sums**, anchored on nn-Meter, shows real
domain command.

**Major concern — make the contribution delta legible and quantified.**
- **vs OFA.** OFA already yields device-specialised subnets training-free via an
  accuracy predictor + evolutionary search; you replace that with BO + short
  fine-tunes (§2.6.3 `sec:supernet`). The cost claim — "a few real evaluations
  against a predictor whose training costs thousands of subnet evaluations" — is
  **asserted, not quantified**.
- **vs FBNet/ProxylessNAS.** These already do measured-LUT, latency-aware search;
  your delta is the LUT + off-grid reach via Net2Net + BO sample-efficiency on one
  device.

*Fix:* a comparison table in §2.8 — rows {latency model, per-candidate eval cost,
off-grid reach, search sample-efficiency, device-faithfulness}, columns {OFA, FBNet,
ProxylessNAS, this work} — plus a GPU-hours budget table for "this pipeline" vs
"OFA predictor+evo." Converts an assertion into a contribution.

**Medium concerns.**
- **Closest missing relative: MCUNet / TinyNAS** (Lin et al.) — NAS for a single
  fixed device under a hard memory budget — is the nearest neighbour and is absent;
  **HAT** (Hardware-Aware Transformers) is the predictor-on-device analogue.
  *Fix:* add MCUNet to §2.5 and distinguish (they co-design a TinyEngine; you use
  stock TensorRT on a larger device).
- **Unnamed external baselines.** Objective 3's scope promises "Pareto-dominance
  over strong mobile baselines," but only random search is committed. *Fix:* name
  MobileNetV3 / EfficientNet-lite / an OFA-specialised subnet and overlay them on
  Fig `fig_res_pareto`.

**Minor (a strength).** The activation subtlety in §2.4.2 — ReLU idempotent so
Net2DeeperNet is exact in early MBv3 stages, h-swish not in the deep stages — is
correctly handled and keeps the morphism-exactness claim from being overstated.

**Recommendation.** *Major revision.* Literature is not the problem; the
contribution *framing* is. Quantify the delta vs OFA and name the external
baselines.

---

### Reviewer 4 of 5 — Peer Reviewer 3 (Cross-disciplinary: GPU performance / systems measurement)
*TensorRT/CUDA performance engineer. Interrogates the measurement physics the ML reviewers wave through — the paper's most novel and most fragile layer.*

**Where this work is better than its field.** The protocol (§4.2.1
`sec:method_protocol`) is systems-grade: CUDA-event timing on the execution stream,
queue depth one, fifty warm-ups, and a **dual stop condition (≥200 samples AND
≥0.5 s) motivated by an observed 9 % drift** that you found and fixed. The shared
timing cache pinning tactics, the clock locking with a preflight that *aborts* on an
unlocked governor (§4.2.2), and full per-row provenance are correct and worth
foregrounding.

**Major concerns.**
1. **The 32 ns quantisation argument is elegant but over-claimed.** §5.2's first
   observation shows the medians are exact multiples of the 31.25 MHz globaltimer
   tick (170432, 75328, 40960 ns). That proves the numbers are **read from the
   GPU's own counter** — it refutes a host-side origin. It does **not** prove the
   values are *correct*. *Fix:* say plainly that quantisation establishes **timer
   provenance, not value correctness**; correctness rests on the plausibility check
   and the (pending) end-to-end additivity validation.
2. **Cross-block fusion is the real threat, and the pilot cannot see it.** Three
   rows of one isolated convolution test the path, not the additive model.
   TensorRT fuses conv+BN+activation and can fuse *across* block seams; blocks timed
   in isolation **miss the fusion savings**, so a summed LUT tends to *over*-predict
   full-network latency. A conv's fastest *tactic* in isolation may differ inside a
   full engine even with a shared cache. The 15 % bar (CP 2.2 / CP 8.2) may be
   optimistic as depth grows. *Fix:* in `tab_lut_validation` report **error vs
   network depth**, not a single aggregate; pre-register the CP 2.3 residual-
   correction trigger.
3. **Precision transfer is unstated as a validity threat.** The LUT is fp32 + TF32
   (runtime default, `nvidia2024trt`), but real deployments use FP16 (~2×) or INT8
   (~4×). Block latency **rankings are not precision-invariant**, so a TF32-optimal
   architecture need not be optimal at the deployment precision. §4.2's "a re-sweep
   away" is true for *rebuilding the table* but not for the *search result*. *Fix:*
   add to §5.10 limitations; soften "faithful to one fixed device" → "…at the
   searched precision"; ideally run the additivity spot-check once at FP16.

**Medium.**
4. **The peak-memory rule is an estimate, not a guaranteed bound.** "m = weights +
   largest single-block working set" (§4.2) can *under*-count with seam double-
   buffering or *over*-count if scratch is reused. *Fix:* validate composed m
   against a measured peak on the same five subnets used for latency.
5. **State the smallest block's regime.** Table `tab_lut_pilot`'s smallest row
   reaches ~2.4 GB/s effective BW and a small fraction of compute peak — it is
   **launch-bound**, which *is* your thesis. Saying so strengthens the case.

**Minor.** Quote the timer resolution as an uncertainty floor (±16 ns); negligible
at 41 µs but it belongs in the error budget.

**Recommendation.** *Major revision.* The rigor is real; the gaps are one over-claim
to soften and two unstated validity threats — fusion and precision — that the
additivity validation must be designed to expose rather than average away.

---

### Reviewer 5 of 5 — Devil's Advocate
*Charge: find the strongest case against the core thesis and the contribution.*

**Strongest counter-argument.** The load-bearing claim is that an evaluation can be
*simultaneously* cheap and faithful to a fixed device, and that assembling five
mature parts is a contribution. The strongest case against is that the assembly's
value is unproven and its *faithfulness is bought by narrowing the problem to the
regime where the easy answers already work.* Faithfulness comes from a measured LUT
— but the LUT is faithful only on **linear chains**, at **one precision (TF32)**,
for **blocks timed in isolation**. Chains at TF32 are precisely where an additive,
FLOPs-adjacent model is least likely to break; the hard cases the field cares about
— branched topologies (FPN, ASPP), fused engines, FP16/INT8 deployment — are scoped
out (§5.10, §7). So "faithful to one fixed device" may reduce to "faithful to the
easy corner of one device." Meanwhile cheapness rests on a proxy fine-tune whose
ranking fidelity is *asserted, not measured*, against a literature the thesis itself
cites showing proxies and random search routinely match full NAS (`li2020random`,
`yang2020`). If the proxy mis-ranks, the impressive BO machinery optimises noise —
cheaply. And the only committed external comparison is random search; the one that
would actually threaten the contribution — **OFA's own predictor-plus-evolution
recipe** — is set aside by argument (§2.6.3), not by experiment. With every
empirical objective and both validation checks still `TODO`, the contribution
*today* is a well-engineered, honest **protocol**, not a result.

**Issue list.**
- **CRITICAL — Empirical claims unvalidated.** Objectives 3/4/5, the additivity
  check (`tab_lut_validation`), and deployment (§5.9) carry no data; the
  abstract/conclusion claims are hypotheses. *(Forbids "Accept"; under the
  vet-the-design TFM frame it sets the verdict to "execute the design," not
  "reject.")*
- **MAJOR — Proxy-rank fidelity untested** (CP 2.4 checks reproducibility, not
  rank). §4.4 / §4.8.
- **MAJOR — Faithfulness scoped to the easy regime** (chains, TF32, isolated
  blocks). §4.2 / §2.5 / §5.10 / §7.
- **MAJOR — The demanding control (OFA predictor+evo) is avoided.** §2.6.3, Obj. 3.
- **MAJOR — D1 unresolved** (dataset/task/teacher/latency-budget/metric). Project
  risk, not a writing gap. §1.4 / §4.5 / §4.7 / §5.7.
- **MINOR — "Strong mobile baselines" promised but unnamed.** Obj. 3.
- **MINOR — Linear-scalarisation Pareto incompleteness unacknowledged.** §2.3 / §4.3.

**Ignored alternative explanations / paths.**
- The pilot's "small blocks cost more than FLOPs predict" may *partly* be fixed
  launch overhead that a **fused full network amortises** — the effect motivating
  measurement could shrink at the network level. Check whether the gap survives
  fusion.
- A **zero-cost proxy** (`mellor2021`, `abdelfattah2021` — cited and dismissed on
  principle in §2.5) + the measured LUT might reach a comparable front far more
  cheaply. Dismissing it without one empirical comparison leaves a cheap baseline
  untested.

**Missing stakeholder perspectives.**
- The **deployer** running FP16/INT8 — the searched result may not transfer.
- The **robotics** reader (degree context) — no concrete robotic task grounds the
  work.

**Observations (non-defects).** No fabricated data (§5 intro); measurement rigor is
real; the "no single piece is novel" honesty is correct and disarming. None of the
above is fatal to the *design* — all of it is fatal to claiming the design *works*
before the runs exist.

---

## Phase 2 — Editorial synthesis & decision

### Consensus vs disagreement

| Finding | EIC | R1 | R2 | R3 | DA | Status |
|---|:--:|:--:|:--:|:--:|:--:|---|
| SOTA breadth + random-search discipline + measurement rigor + no-fabrication integrity | ✓ | ✓ | ✓ | ✓ | ✓ | **Consensus strength** |
| Proxy-rank fidelity untested (the #1 design gap) | | ✓ | | | ✓ | **Consensus** |
| Faithfulness scoped to easy regime (precision + fusion under-flagged) | | | | ✓ | ✓ | **Consensus** |
| Contribution delta vs OFA/FBNet must be explicit & quantified | ✓ | | ✓ | | ✓ | **Consensus** |
| D1 (dataset/teacher/budget) unresolved = structural risk | ✓ | | | | ✓ | **Consensus** |
| How damning is the pending data? | stage-expected | stage-expected | stage-expected | stage-expected | **CRITICAL** | **Tension → split verdict** |
| Is integration-as-contribution enough? | yes (TFM) / borderline (venue) | — | only if quantified | — | no (yet) | **Tension → arbitration** |

**Arbitration.** The pending-data tension resolves by *frame*: CRITICAL for
**publishability**, expected for the **TFM at this stage** — so the decision is
reported twice rather than blended. Integration-as-contribution is sufficient for
the degree *if the runs land and the OFA-recipe comparison is at least discussed*;
for a venue it needs that comparison plus an ablation. The Devil's Advocate CRITICAL
forbids any "Accept."

### Decision

**(i) TFM tribunal frame — MAJOR REVISION, read as "on track."** The design is
sound and defensible and the writing (Ch. 1–4) is strong; nothing in the
methodology is broken. The thesis is **not defendable today** (no results) but is on
a defensible track. Close the P0 design gaps *before* spending search compute, then
execute.

**(ii) Publishability addendum — not yet submittable to any venue.** Path to
submittable: complete Objectives 3/4/5 with multi-seed statistics + proxy-rank
validation + named external baselines + an OFA-recipe comparison + precision/fusion
robustness, with a transfer discussion.

### Revision roadmap (prioritised — design-before-runs first)

**P0 — close before spending compute (cheap now, ruinous later):**
1. **Resolve D1** — fix dataset/task, teacher, latency budget (ms), accuracy metric.
2. **Proxy-rank validation** — fully train ≈8–12 architectures, report Kendall-τ vs
   the proxy, gate the search on an acceptable τ. (New DoD beside CP 2.4.)
3. **Statistical protocol for Obj. 3** — ≥5 seeds for BO *and* random; Pareto
   hypervolume with dispersion + dominance-across-seeds statement.
4. **Specify batch-EI diversification** and justify the 100-candidate budget (D2);
   seed the GP with the random-search evaluations.

**P1 — tighten claims/framing now (no compute needed):**
5. **Contribution-delta table** (OFA / FBNet / ProxylessNAS / this work) in §2.8 +
   a GPU-hours budget table vs OFA's predictor+evo.
6. **Name the external baselines** (Obj. 3) and overlay them on Fig `fig_res_pareto`.
7. **Add to §5.10 limitations:** precision transfer, cross-block fusion,
   memory-rule validity; soften "faithful to one fixed device" → "…at the searched
   precision and topology."
8. **Design the additivity check to expose, not hide, the threat:**
   `tab_lut_validation` reports error **vs depth**; pre-register the CP 2.3 trigger.

**P2 — for publishability, later:**
9. OFA-predictor+evolution comparison; a zero-cost-proxy cheap baseline; an FP16
   additivity spot-check; MCUNet/HAT in related work; an ablation isolating
   Net2Net's contribution.

**Quick wins (minor):** state the 32 ns result as *timer-provenance, not accuracy*;
name the smallest pilot block as launch-bound; acknowledge linear scalarisation
can't reach concave Pareto regions; quote the ±16 ns timer floor.

---

*End of simulated review. This document is advisory: a rehearsal of the questions a
real tribunal or program committee is likely to ask, calibrated to a mid-project
TFM whose design is finished and whose data is pending.*
