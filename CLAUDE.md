# CLAUDE.md — AI Session Context

This is a hardware-aware Neural Architecture Search (NAS) thesis project. Read
this file at session start to avoid re-deriving context from scratch.

---

## Project in one paragraph

The goal is to find compact architectures for the **Jetson Orin Nano** (8 GB,
FP16 TensorRT) that Pareto-dominate MobileNetV3 on the accuracy/latency
frontier, without training from scratch. The strategy: sample subnets from a
pretrained **OFA supernet**, score them with a **Jetson-measured latency LUT**
plus a short fine-tune accuracy, and let **Bayesian Optimization** guide the
search. **Net2Net** transforms warm-start weights when BO proposes a nearby
architecture, so each evaluation costs ~5 epochs instead of full training.

---

## How to resume a session

1. Read `state/plan_state.yaml` → tells you the current checkpoint.
2. Read the relevant section of `procedure.md` → tells you what was done and
   why at each completed checkpoint.
3. Read `PROJECT_PLAN.md` → the next checkpoint's inputs, deliverables, and DoD.
4. Activate the right venv (see below) and verify the DoD commands still pass.

---

## Current state (as of last update: CP 1.3)

- **Phase 0 (LUT):** Complete. `data/lut.jsonl` has all catalog rows.
- **CP 1.1:** Done — skeleton packages + state file.
- **CP 1.2:** Done — OFA w1.0 checkpoint downloaded + SHA256 pinned.
- **CP 1.3:** Done — `supernet/sampler.py` works; random subnet forwards
  `(1, 3, 224, 224) → (1, 1000)` without error.
- **CP 1.4 (next):** ImageNet sanity — confirm a sampled subnet is within
  1.5% top-1 of OFA's published number on a 2k-image ImageNet-val subset.
- **Phases 2–8:** Planned (see `PROJECT_PLAN.md`).

### Open design decisions (do not resolve unilaterally)

| ID | Decision | Blocks |
|---|---|---|
| D1 | Target dataset (ImageNet vs. Cityscapes vs. COCO) | CP 2.4 onward |
| D2 | Search budget (default: 100 candidates Phase 3, 200 Phase 7) | CP 3.2 / 7.2 |
| D3 | Which SOTA blocks to inject (FusedMBConv, ConvNeXt, MobileViT) | CP 5.3 |
| D4 | λ, μ in `J(α) = acc − λ·latency − μ·max(0, mem−budget)²` | CP 3.3 |
| D5 | Multi-device extension (out of scope for v1/v2) | v3 |

---

## Module structure

```
catalog/      Block registry (shared by lut/ and all NAS phases)
lut/          Phase 0: Jetson LUT pipeline (DONE)
  export/     PyTorch → ONNX
  bench/      Jetson-side TRT engine build + benchmarking (runs in Docker)
  orchestrate/ Laptop-side sweep loop + SSH orchestration
  docs/       lut.jsonl + device_info.json schema
supernet/     Phase 1: OFA-MBv3-w1.0 wrapper + subnet sampler (CP 1.3 done)
search/       Phase 2–3: search loop stub
eval/         Phase 2–3: fine-tune harness stub
net2net/      Phase 4: Net2Net operators stub
expand/       Phase 5–6: supernet expansion stub
state/        Checkpoint tracking (plan_state.yaml)
data/         lut.jsonl + device_info.json (gitignored)
scripts/      Setup scripts (setup_laptop.sh, setup_laptop_nas.sh, etc.)
```

---

## Two environments — always activate the right one

| Venv | Activate | Use for |
|---|---|---|
| `.venv` | `source .venv/bin/activate` | LUT pipeline (CPU torch, fabric for SSH) |
| `.venv-nas` | `source .venv-nas/bin/activate` | NAS pipeline (GPU torch, ofa, torchvision) |

**Never** mix them. The LUT pipeline uses `torch==2.3.1+cpu`; the NAS pipeline
uses `torch>=2.3,<2.12` (GPU build). Installing one into the other's venv
breaks things.

To rebuild `.venv-nas`: `bash scripts/setup_laptop_nas.sh` — **do not**
`pip install -r requirements-nas.txt` directly (it pulls the wrong torchvision
CUDA variant from PyPI).

---

## Key conventions

- **LUT schema is append-only.** Each row is keyed by
  `sha1(block + cfg + input_shape)`. Adding new blocks or widening grids
  never invalidates existing rows. Never edit `data/lut.jsonl` by hand.
- **Idempotent sweep.** Re-running `python -m lut.orchestrate.run_sweep` skips
  already-measured rows. Safe to Ctrl-C and resume.
- **No PyTorch on the Jetson.** The Jetson runs only TensorRT (inside Docker).
  Keeping PyTorch off it preserves its 8 GB RAM for accurate memory measurements.
- **FP16 only.** INT8 is a non-goal for v1 (it can be added as a new
  `precision` column without schema changes).
- **OFA checkpoint is pinned.** SHA256 lives in `supernet/download_ofa.py`.
  Do not change it without recording the decision in `procedure.md`.
- **Checkpoint discipline.** Every completed checkpoint must: (a) have its
  entry in `procedure.md` with full rationale, (b) advance `state/plan_state.yaml`.

---

## Things NOT to do

- `pip install -r requirements-nas.txt` directly → use `setup_laptop_nas.sh`.
- Edit `data/lut.jsonl` by hand.
- Commit anything in `data/` (it's gitignored for a reason — 50+ MB).
- Resolve open decisions D1–D5 without a user conversation.
- Name a local Python package `ofa/` — it shadows the pip-installed OFA library.
  The wrapper is in `supernet/` for this reason.
