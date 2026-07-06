# Stage R (i) — Graft-interface / neck design for transplanted backbones

*ARS `deep-research` three-way-scan (low-oversight mode), 2026-07-06. Question: how much does a
neck buy on a lightweight detector, what is the minimal viable fusion, and how should the
CP 5.1 nano-neck (V1/V2/V3) be designed? Feeds Phase 5 (PROJECT_PLAN.md) design detail only —
the ablation itself remains the decider.*

## Shortlist (3 papers, WHY / HOW / WHAT)

### 1. YOLOF — Chen et al., CVPR 2021 ([arXiv:2103.09460](https://arxiv.org/abs/2103.09460))
- **WHY** — Is FPN's benefit the multi-scale *fusion*, or the divide-and-conquer assignment of
  objects to per-scale outputs?
- **HOW** — Controlled encoder ablation on RetinaNet-class detectors: MiMo (= FPN) vs SiMo
  (single C5 input, **no fusion**, multi-scale outputs) vs MiSo/SiSo; then a single-level
  detector (dilated encoder + uniform matching) built on the finding.
- **WHAT** — SiMo lands **within <1 mAP of MiMo**: fusion contributes little once multi-scale
  *outputs* exist; the big loss only appears when the multi-level outputs are removed (SiSo).
  FPN's value ≈ divide-and-conquer, not fusion.

### 2. EfficientDet / BiFPN — Tan et al., CVPR 2020 ([arXiv:1911.09070](https://arxiv.org/abs/1911.09070))
- **WHY** — What fusion topology and weighting give the best accuracy/cost on
  efficiency-constrained detectors?
- **HOW** — Swap necks on a fixed detector: top-down FPN vs PANet vs NAS-FPN vs BiFPN;
  per-input-edge **scalar** fusion weights (softmax vs "fast normalized" variants).
- **WHAT** — Top-down-only FPN is the *weakest* topology ("inherently limited by the one-way
  flow"); adding the bottom-up path (PANet) beats it, BiFPN matches repeated FPN+PANet at much
  lower cost; **scalar per-edge weights** suffice, and fast-normalized ≈ softmax accuracy while
  running 28–31 % faster. In their pipeline-level ablation, better fusion was worth ~4 AP over
  plain FPN (backbone held fixed) — on models far bigger than ours.

### 3. ViTDet — Li et al., ECCV 2022 ([arXiv:2203.16527](https://arxiv.org/abs/2203.16527))
- **WHY** — Does a *plain* (non-hierarchical, transplanted-from-pretraining) backbone need an
  FPN at all?
- **HOW** — Build detectors on plain ViT with (a) no pyramid, (b) a **simple feature pyramid**
  (parallel per-scale convs, no top-down/lateral fusion), (c) full FPN variants; compare.
- **WHAT** — Any pyramid beats none (up to **+3.4 AP**), but the simple no-fusion pyramid
  matches the FPN designs (~+3.2 either way): with a strong backbone, multi-scale *sampling*
  is sufficient and cross-scale fusion is not needed.

## Cross-paper synthesis

- **Common WHY:** decompose "the neck" into its two jobs — producing per-scale outputs vs
  mixing information across scales — and price each.
- **Divergent HOW:** YOLOF ablates *inputs* to the encoder, EfficientDet ablates *topologies +
  weights*, ViTDet ablates the pyramid's very existence on a non-hierarchical backbone.
- **Strongest WHAT (for us):** fusion's marginal value is real but small (≲1–2 AP) once
  per-scale outputs exist, *and it shrinks as the backbone gets stronger/better-matched*; when
  fusion is used, **scalar per-edge weights are the evidenced form** (BiFPN), and top-down-only
  is the weakest-but-cheapest topology (one-way flow).
- **Unresolved gap:** none of these study a **frozen consumer head** — every paper trains the
  head jointly. Our setting (donor-trained, frozen YOLO11-pose head; CP 2.4's LP-FT-style
  finding that the interface dominates the proxy signal; cf. Kumar et al., ICLR 2022,
  [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)) has no direct literature answer — the
  V0/V1/V2 ablation is genuinely novel evidence, not a reproduction.

## Design implications for CP 5.1 (V1/V2/V3)

1. **How much can V2 (top-down-only) recover?** Expect *modest* gains: top-down-only is the
   weakest fusion topology (BiFPN), and fusion per se is worth ≲1 mAP-class once P3/P4/P5
   outputs exist (YOLOF, ViTDet). The honest hypothesis: our accuracy gap lives at least as
   much in the *interface statistics* (random adapters into a frozen head) as in missing
   fusion. The V0→V1 (init-only) vs V1→V2 (fusion) split of the ablation measures exactly this
   decomposition — keep it.
2. **Zero-init gates: supported.** ReZero ([arXiv:2003.04887](https://arxiv.org/abs/2003.04887))
   gates each residual with a **single zero-initialized scalar**, starts at the identity, and
   trains stably/faster at depth. Our day-0-function-preserving neck is the same mechanism.
3. **Gate granularity: one scalar per fusion edge** (P5→P4 and P4→P3 each get one α; V3's
   bottom-up edges likewise). This is what both ReZero and BiFPN's fusion weights use;
   per-channel gates have no evidence behind them here and add TRT pointwise overhead.
   → resolves the open CP 5.1 design question in favor of scalars (ablation can revisit).
4. **Adapter init under a frozen head (V1): no direct external evidence either way.** Nearest
   support is LP-FT's feature-distortion argument + our own CP 2.4 repair. V1 stays in the
   ablation as a cheap, genuinely open hypothesis; do not presume it wins.
5. **Risk framing for the thesis:** if V2 ≈ V1 ≈ V0 on the proxy, that is *consistent with*
   YOLOF/ViTDet (fusion rarely binding) and still publishable as a negative result with
   literature backing — pre-register that interpretation now.

## Fidelity notes (low-oversight scan)

Verified via web sources this session: YOLOF's SiMo-within-<1-mAP-of-MiMo claim; BiFPN's
topology ranking, scalar fast-fusion ≈ softmax (28–31 % faster), and the ~+4 AP
fusion-pipeline delta; ViTDet's +3.4/+3.2 pyramid numbers; ReZero's scalar-zero-init mechanism.
Exact per-table numbers (e.g., FPN-vs-PANet row-level APs) were **not** re-read from the PDFs —
pull them before quoting in the thesis text. Escalate to `lit-review` mode if the chapter needs
full coverage (e.g., GiraffeDet, RepGFPN, NAS-FPN lineage).
