# Hardware-Aware NAS for Jetson Orin Nano

Master's thesis project. Samples compact child architectures from a pretrained
**Once-for-All (OFA)** supernet, scores them against a **Jetson-measured latency
lookup table** plus task accuracy, drives a **Bayesian Optimization** loop, and
uses **Net2Net** transforms to warm-start weights so every evaluation is cheap.

For the full motivation and design: [`PROJECT.md`](PROJECT.md).

---

## Current status

| Milestone | Checkpoint | Status |
|---|---|---|
| Latency LUT | Phase 0 | ✅ Complete |
| OFA supernet skeleton | CP 1.1 | ✅ Done |
| OFA checkpoint download + hash pin | CP 1.2 | ✅ Done |
| Subnet sampler | CP 1.3 | ✅ Done |
| ImageNet sanity check | CP 1.4 | 🔲 Next |
| Subnet extraction + LUT-aware scoring | Phase 2 | 🔲 Planned |
| Device-aware search v1 | Phase 3 | 🔲 Planned |
| Net2Net operator library | Phase 4 | 🔲 Planned |
| Supernet expansion | Phase 5 | 🔲 Planned |
| Expanded supernet fine-tune | Phase 6 | 🔲 Planned |
| Search on expanded supernet | Phase 7 | 🔲 Planned |
| Knowledge distillation | Phase 8 | 🔲 Planned |
| Deployment packaging | Phase 9 | 🔲 Planned |

Exact checkpoint: `state/plan_state.yaml`.

---

## Module map

| Module | Purpose | Status |
|---|---|---|
| `catalog/` | Block registry — 13+ block types + sweep grids; shared by LUT and NAS | ✅ Done |
| `lut/` | Jetson latency LUT pipeline (export → Jetson → TRT bench → jsonl) | ✅ Done |
| `supernet/` | OFA-MBv3-w1.0 supernet wrapper + subnet sampler | ✅ CP 1.3 done |
| `search/` | BO/NSGA-II search loop, arch encoder, cost function | 🔲 Phase 2–3 |
| `eval/` | Short fine-tune harness + long-train baseline evaluator | 🔲 Phase 2–3 |
| `net2net/` | Function-preserving widen/deepen operators + graph diff | 🔲 Phase 4 |
| `expand/` | Cross-family block injection + LUT pre-screen | 🔲 Phase 5–6 |
| `distill/` | KD harness — external teacher + distillation-loss final train of the winner | 🔲 Phase 8 |

---

## Quick start

**To build or extend the Jetson LUT:**

```bash
bash scripts/setup_laptop.sh
source .venv/bin/activate
# See lut/README.md for the full workflow
python -m lut.orchestrate.run_sweep
```

**To work on the NAS pipeline:**

```bash
bash scripts/setup_laptop_nas.sh   # creates .venv-nas/ with GPU torch + ofa
source .venv-nas/bin/activate
python -m supernet.sampler         # CP 1.3 smoke test: sample a random subnet
```

**To generate a synthetic LUT (no Jetson needed):**

```bash
source .venv-nas/bin/activate
python -m lut.orchestrate.gen_dummy_lut
```

---

## Key documents

| Document | What it is |
|---|---|
| [`PROJECT.md`](PROJECT.md) | Vision — why this project exists, the four-piece design |
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Phase plan — all 9 phases, checkpoints, DoDs, open decisions |
| [`procedure.md`](procedure.md) | Checkpoint journal — every decision, command, and justification |
| [`state/plan_state.yaml`](state/plan_state.yaml) | Machine-readable resume state |
| [`lut/README.md`](lut/README.md) | LUT pipeline guide — hardware setup, sweep, schema |
| [`supernet/README.md`](supernet/README.md) | OFA wrapper guide — deps, checkpoint pin, usage |
| [`lut/docs/schema.md`](lut/docs/schema.md) | LUT + device_info JSON schema |
| [`CLAUDE.md`](CLAUDE.md) | AI session context (read this to resume quickly) |
