# `distill/` — Knowledge-Distillation Harness (Phase 8)

**Status:** 🔲 Stub. Phase 8 of `PROJECT_PLAN.md`. No implementation yet — this
directory reserves the module and records the design so the plan's module map
has an honest target.

Phase 8 takes the search winner α\* (the **student**) and trains it to its
maximum accuracy against a strong external **teacher**, producing the final
deployable weights. This is the project's one full-schedule training run; every
accuracy number before it is a 5-epoch *proxy* used only to rank candidates.

> **Latency is unchanged.** KD transfers knowledge into *weights*, never the
> graph. α\*'s architecture — and therefore its LUT-summed latency — is identical
> before and after distillation. The LUT contract and Phase 9's ≤ 15 % export
> bar are untouched; only accuracy moves.

## Planned files

| File | Role | Checkpoint |
|---|---|---|
| `teacher.py` | Load the external SOTA teacher (frozen, eval-mode); expose `teacher(x)`. | CP 8.1 |
| `distill.py` | KD loss + full-schedule training harness; reuses `eval/`'s D1 data pipeline + metrics. | CP 8.2–8.3 |

## Distillation loss (CP 8.2)

```
L = α · T² · KL( softmax(z_student / T) ‖ softmax(z_teacher / T) )
  + (1 − α) · CE( z_student, y )
```

- `T` — temperature; `α` — KD/CE mix; full cosine LR schedule, fixed seed.
- Segmentation: per-pixel KL. Detection: logit / feature mimicking.

## Teacher pin (CP 8.1) — TBD

The concrete teacher depends on **D1** (target dataset/task), still open:

| D1 task | Candidate teacher |
|---|---|
| Classification (ImageNet) | ConvNeXt-L / EfficientNetV2-L (`timm`) |
| Segmentation (Cityscapes / ADE20K) | SegFormer-B5 |
| Detection (COCO) | a strong detector |

Once chosen, pin its URL + SHA256 here — same discipline as the OFA checkpoint
pin in `supernet/download_ofa.py`. **A classification teacher will not transfer
to seg/det** — pick the teacher after D1 is resolved.

## Dependency

The full distillation run needs **CUDA** (the documented blocker; resolve before
CP 2.4 / this phase). The loss and harness are unit-testable on CPU with tiny
tensors; the real run is GPU-only. Runs in the `.venv-nas` environment.

## References

- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*,
  NeurIPS-W 2014. https://arxiv.org/abs/1503.02531
- (seg/det) Liu et al., *Structured Knowledge Distillation for Dense
  Prediction*, CVPR 2019. https://arxiv.org/abs/1903.04197
