# Stage R (iii) — Supernet expansion: compute cost + ranking-fidelity evidence (descope defense)

*ARS `deep-research` three-way-scan (low-oversight mode), 2026-07-06. Purpose: the thesis must
defend descoping the old Phases 5–7 (supernet expansion + re-search; D3 → descoped 2026-07-05)
with literature, not just our own artifacts. Two independent grounds: (a) the compute cost of
(re)training expanded shared weights, (b) weight-sharing ranking fidelity, which degrades with
space size — the "enriched space is unrankable" argument.*

## Shortlist (3 papers, WHY / HOW / WHAT)

### 1. Once-for-All — Cai et al., ICLR 2020 ([arXiv:1908.09791](https://arxiv.org/abs/1908.09791))
- **WHY** — Train once, specialize everywhere: amortize NAS training across deployments.
- **HOW** — Progressive shrinking (PS): train the full net, then progressively bring smaller
  ks/e/d subnets into the shared weights, phase by phase.
- **WHAT** — The cost anchor: the OFA training run is ≈ **1,200 V100 GPU-hours** (restated by
  follow-ups, e.g. DϵpS, ECCV 2024). That is the price of the *ranking-faithful shared
  weights our whole pipeline free-rides on* — and the price class a post-injection
  re-co-adaptation buys back into.

### 2. CompOFA — Sahni et al., ICLR 2021 ([arXiv:2104.12642](https://arxiv.org/abs/2104.12642)) *(cheap-variant line; DϵpS, ECCV 2024, same lesson)*
- **WHY** — OFA's cost is the bottleneck; can constraining the space cut it?
- **HOW** — Couple the elastic dimensions (compound width/depth) so the space shrinks by
  orders of magnitude; train the smaller family.
- **WHAT** — Same accuracy at **half** the training budget — i.e., even the *efficiency line*
  of this literature saves cost by **shrinking** the space, the exact opposite of expansion;
  and "half of 1,200 V100-h" still dwarfs a free-tier Kaggle/Colab + one AGX budget.

### 3. Evaluating the Search Phase of NAS — Yu et al., ICLR 2020 ([arXiv:1902.08142](https://arxiv.org/abs/1902.08142))
- **WHY** — Does weight sharing actually rank candidates by their true standalone quality?
- **HOW** — Compare weight-sharing rankings against ground-truth (fully trained) rankings
  under controlled protocols.
- **WHAT** — Weight sharing **degrades candidate ranking to the point of not reflecting true
  performance**, explaining why several NAS methods matched random search. Follow-up
  weight-sharing appraisals report the size effect directly: on NAS-Bench-201, supernet rank
  correlation only becomes respectable once the space is *downscaled* ~64× (to a few hundred
  archs) — **the larger the space, the worse the shared-weight oracle ranks it.**

## Cross-paper synthesis

- **Common WHY:** shared weights are a cost-amortization device whose value stands or falls
  with ranking fidelity.
- **Divergent HOW:** buy fidelity with massive training (OFA), with space *reduction*
  (CompOFA/DϵpS), or audit it and find it wanting (Yu et al.).
- **Strongest WHAT (for the descope):** expansion attacks from both sides at once — it adds
  training cost we cannot pay (papers 1–2) *and* grows the space in the direction that erodes
  the oracle's ranking ability (paper 3). Our own CP 3.5 measurements put numbers on the
  second effect locally: per-arch fresh-seed σ 0.005–0.025 vs top-cluster gaps ~0.014, i.e.,
  the existing space is already at the oracle's resolution limit.
- **Unresolved gap:** none of these measure ranking fidelity for *injected-and-lightly-adapted*
  operators specifically (our old CP 5.5/6.2 plan) — but that cuts against expansion too: we
  would have had to re-run a CP 2.4-style fidelity gate per injected family before trusting
  any of it.

## Implications for the thesis (defense of the 2026-07-05 pivot)

1. **Cost ground:** cite OFA's ≈1,200 V100-h and the plan's own "OFA's full PS is 180+
   GPU-days. We don't have that" note; even halving-line methods (CompOFA/DϵpS) remain
   orders beyond the free-tier budget that already interrupted CP 3.3 once.
2. **Fidelity ground:** cite Yu et al. + the space-size effect, then present CP 3.5's
   winner's-curse data (σ vs top-gap) as the *in-situ* replication of the same phenomenon —
   the oracle was already saturated before any enrichment.
3. **Consistency ground:** D3's own recorded rule of thumb ("run Phase 3 first, look where α*
   lands, don't inject speculatively — the fine-tune budget is the bottleneck") evaluates, on
   Phase-3's outcome, to *no injection*. The descope follows the plan's own logic.
4. Keep CP 5.0-lite (FusedMBConv LUT screen) as the constructive future-work exhibit: the
   screen costs one idempotent sweep and shows exactly what an expansion would have had to
   beat, without paying Phase 6.

## Fidelity notes (low-oversight scan)

Verified this session: OFA's 1,200-V100-hour figure (paper + DϵpS restatement); CompOFA's
half-budget claim; Yu et al.'s ranking-degradation finding and its random-search-parity
framing; the NAS-Bench-201 downscale-~64× rank-correlation observation (reported in the
weight-sharing appraisal literature — re-verify the exact source table before quoting a number
in the thesis). DϵpS (ECCV 2024) skimmed at abstract level.
