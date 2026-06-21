# CLAUDE.md — AI Session Context

This is a hardware-aware Neural Architecture Search (NAS) thesis project. Read
this file at session start to avoid re-deriving context from scratch.

---

## Project in one paragraph

The goal is to find a compact architecture for the **Jetson Orin Nano** (8 GB,
FP16 TensorRT) that Pareto-dominates the deployed **YOLO11n-pose** on the
accuracy/latency frontier for **drone-racing gate detection + 8-keypoint pose**
(the target task, D1 = `dataset/`), without training from scratch. The strategy:
sample subnets from a pretrained **OFA supernet** and use each as a **backbone**
under a **YOLO11-pose head**, score them with a **Jetson-measured latency LUT**
plus a short pose fine-tune (**pose mAP / OKS**), and let **Bayesian
Optimization** guide the search. **Net2Net** transforms warm-start weights when
BO proposes a nearby architecture, so each evaluation costs ~5 epochs instead of
full training. Once the search settles, the winning architecture is
**distilled** against a bigger **YOLO11-pose** teacher for its final deployable
weights (Phase 8).

---

## How to resume a session

1. Read `state/plan_state.yaml` → tells you the current checkpoint.
2. Read the relevant section of `procedure.md` → tells you what was done and
   why at each completed checkpoint.
3. Read `PROJECT_PLAN.md` → the next checkpoint's inputs, deliverables, and DoD.
4. Activate the right venv (see below) and verify the DoD commands still pass.

---

## Current state (as of last update: CP 2.4 FAILED → head-warm-start repair built, GPU re-test owed, 2026-06-21)

- **Phase 0 (LUT):** COMPLETE. `data/lut.jsonl` holds all 2710 *measured* rows
  (`source=jetson_trt`, fp32, TRT 10.3.0, clocks locked); the original dummy lives
  in `data/lut.jsonl.dummy.bak`. Re-run/extend: `python -m lut.orchestrate.run_sweep`
  (idempotent; a device-state preflight re-probes the Jetson and refuses to
  measure with unlocked clocks / wrong power mode — `--skip-preflight` to
  bypass). `scripts/setup_jetson.sh` before, `scripts/teardown_jetson.sh`
  after.
- **CP 1.1:** Done — skeleton packages + state file.
- **CP 1.2:** Done — OFA w1.0 checkpoint downloaded + SHA256 pinned.
- **CP 1.3:** Done — `supernet/sampler.py` works; random subnet forwards
  `(1, 3, 224, 224) → (1, 1000)` without error.
- **CP 1.4:** CLOSED (2026-06-18) — re-framed from an absolute-bar top-1 to a
  **rank-fidelity** gate: Spearman ρ=0.919 (≥0.85) over 20 archs on full ImageNet
  val (Kaggle GPU). Confirms the OFA weight load + BN recalibration are intact.
  See `procedure.md` "CP 1.4 CLOSED".
- **CP 2.1:** Done — `search/arch_to_blocks.py` translates an OFA arch_dict to
  the ordered `(mbconv, cfg, input_shape)` LUT-keyed list. Added
  `catalog/ofa_mbv3.py` (shared OFA-MBv3 topology + `reachable_mbconv_configs()`
  → 91 configs) and unioned those into the catalog grid (mbconv 2016→2107);
  dummy LUT regenerated (2710 rows). DoD passes (10 random + 1 real arch).
- **Hardening pass (2026-06-11):** not a checkpoint. Test suite (`tests/`,
  102 tests incl. golden row_key hashes), dev tooling (`pyproject.toml`,
  `scripts/check.sh`, `requirements-dev.txt`, GitHub Actions CI),
  `lut/loader.py` + precision-aware `completed_keys()` (CP 2.2 groundwork),
  `catalog/flops.py` + `catalog/contracts.py`, boundary validation
  (`validate_arch_dict`, config checks), credentials scrubbed from git
  history (history rewritten + force-pushed — stale clones must re-clone).
  One TODO(user) remains: the `load_device_info` failure policy in
  `lut/orchestrate/run_sweep.py`. See `procedure.md` "Hardening pass".
- **Measurement audit (2026-06-12):** not a checkpoint. Audited the LUT
  collection path after the first real rows landed; methodology confirmed
  sound (CUDA-event timing, queue depth 1, warmup, locked clocks). Added:
  sweep-start **preflight** (re-probes device, verifies `clocks_locked` +
  power mode; policy in `run_sweep.preflight_verdict` — TODO(user) owns it),
  **min timed window** (`sweep.min_window_s`, default 0.5 s — kills the ~9%
  p50 drift seen on ~40 µs blocks), **TRT timing cache** (persistent at
  `{remote_workdir}/cache`), `config.local.yaml` overlay (gitignored real
  endpoint; `config.yaml` is a placeholder template). Decisions: fp32 rows
  deliberately allow TF32 (deployment-realistic; see `lut/docs/schema.md`);
  `peak_mem_mib` = TRT scratch + IO (excludes weights; do NOT sum across
  blocks). Rows now stamp `source`/`clocks_locked`. See `procedure.md`
  "Measurement audit".
- **CP 2.2:** CLOSED (2026-06-17) — `search/cost.py` composes a subnet's predicted
  cost from per-block LUT rows (SUM latency/params/flops, MAX peak_mem). The
  depth-binned additivity **DoD PASSES** on-device (33 whole subnets; every bin
  +6.8…9.2 %, aggregate +7.9 %, under the 15 % bar, bias flat in depth) → the
  conditional **CP 2.3** (residual correction) was **NOT triggered — skipped**.
  Predictor fidelity Spearman ρ=0.991 / Kendall τ-b=0.943; opt-in calibration
  (`measured ≈ 0.934·summed`) cuts MAPE 7.9 → 1.0 %. See `procedure.md` "CP 2.2 closed".
- **D1 RESOLVED → gate-pose (2026-06-18) — the pose pivot.** Target = `dataset/`
  (Ultralytics YOLO-pose: 1 class `gate`, 8 keypoints, 2842 train / 140 val
  synthetic A2RL renders; see `dataset/SCHEMA.md`). Decisions: (1) the OFA subnet
  becomes a **backbone** under a **YOLO11-pose head** — keeps supernet / LUT /
  Net2Net / BO, ImageNet pretrain = warm-start; (2) baseline-to-beat + Phase-8
  teacher = the deployed **yolo11n-pose** / a bigger **yolo11-pose** (reuse
  `yolo-ros2-inference/scripts/yolo_distillation.py`); (3) accuracy metric top-1 →
  **pose mAP (OKS)**, reusing Ultralytics' validator. Prototype built +
  CPU-verified in `.venv-nas`: `supernet/pose_backbone.py` (OFA subnet → P3/P4/P5
  at the invariant **(40, 112, 160)** ch, strides 8/16/32) and `detect/` (channel
  adapter + a real Ultralytics `Pose` head → boxes/scores/kpts). Added
  `ultralytics>=8.3`; fixed the moved-repo `setup_laptop_nas.sh`. **CONSEQUENCE
  owed:** LUT rows are keyed per-block at res 224; pose runs @640 → blocks re-key →
  a **2nd LUT sweep at the deploy resolution** is owed (append-only schema absorbs
  it; `cost.py` offset generalizes to stem + pose head). See `procedure.md`
  "D1 resolved — pose pivot".
- **CP 2.4 (eval harness — first GPU run FAILED both DoDs 2026-06-21; head-warm-start repair built):**
  The CPU slice (built 2026-06-18) is the trainable graft `detect.pose_model.GraftedPoseModel`
  + `detect.evaluate.pose_map_model` + the `eval/shortft.py` harness with the two DoD gates
  `rank_fidelity`/`reproducible`, plus the one-command driver `eval/proxy_rank.py`. **The Kaggle
  GPU run failed both gates** (`data/cp24_proxy_rank.json`): proxy-rank **Kendall-τ = 0.20**
  (gate ≥ 0.7) and reproducibility **Δ = 0.0149** (≤ 0.005). **Investigation (read-only, no GPU —
  correlated the 10 archs vs zero-cost LUT descriptors):** `full_map` tracks size strongly — depth
  τ=0.767, Jetson latency τ=0.733 (both *pass* the gate; ordered even inside the cluster) — so the
  ground truth is **real, not flat**; the 5-epoch proxy correlates with **nothing** (τ=0.20; −0.08
  w/o the min corner; τ=0.07 vs FLOPs). Root cause = the **randomly-initialized** Pose head
  (`eval/shortft.py` trains it from scratch in 5 epochs → head-init luck; idx8 = best backbone but
  worst proxy). **Decision (AskUserQuestion → "fix the head first"):** built the **head warm-start +
  freeze** repair — `detect.pose_model.warm_start_head` / `freeze_module` / `_donor_head_state`,
  `build_grafted_pose_model(head_weights=, freeze_head=)`, `short_finetune` trains only trainable
  params, and `eval/proxy_rank.py` gains `--head-weights/--freeze-head/--reset-proxy` (242 tests
  green). **GPU re-test owed (Kaggle, on a *copy* of the results file):** `python -m eval.proxy_rank
  --reset-proxy --head-weights <gate-yolo11n-pose.pt> --freeze-head --no-full --device cuda --imgsz
  640 --batch 16 --out .../cp24_warmstart.json` → re-correlates the warm proxy vs the existing full
  maps. **Pass = τ≥0.7 & Δ≤0.005 → CP 2.4 closes;** miss → run the (already-built) `--diagnose-full`
  noise floor. `current_checkpoint` stays 2.4. See `procedure.md` "CP 2.4 — repair: head warm-start".
- **Phases 3–9:** Planned (see `PROJECT_PLAN.md`).

### Known blockers

- **CUDA / `.venv-nas` (laptop).** `torch.cuda.is_available()` is False here, `nvidia-smi`
  isn't on PATH, and **`.venv-nas` is not currently built** (only `.venv` exists) — so every
  OFA/ultralytics integration (the fine-tune, the warm-head re-test) runs on **Kaggle /
  the Jetson**, not locally. The CP 2.4 GPU run is **done and failed** (see Current state); the
  next GPU step is the **head-warm-start re-test** (`--reset-proxy --head-weights … --freeze-head`,
  proxy-only). Pure logic (`full_noise_verdict`, the DoD gates) is unit-tested in `.venv`/CI, and
  both `run_full_diagnostic`'s and `run_protocol`'s (incl. `--reset-proxy` + warm-start threading)
  resume/guard/verdict paths are covered by **stubbed-fine-tune** tests, so the orchestration is
  verified without a GPU. The
  owed **640 LUT re-sweep** + **baseline yolo11n-pose anchor** stay Jetson-gated (anchor's *mAP
  half* runs on CPU via `detect.evaluate.pose_map`).

### Lowest-friction next build

The CP 2.4 proxy failed (τ=0.20) because of a **random Pose head**; the **head-warm-start repair is
built** (see CP 2.4 bullet). The next step is the **warm-head re-test** (one command, Kaggle; needs
the deployed **gate** `yolo11n-pose.pt` as `--head-weights`, and a *copy* of the prior results):
```
cp data/cp24_proxy_rank.json /kaggle/working/cp24_warmstart.json   # never touch the original
python -m eval.proxy_rank --reset-proxy --head-weights <gate-yolo11n-pose.pt> --freeze-head \
    --no-full --device cuda --imgsz 640 --batch 16 --out /kaggle/working/cp24_warmstart.json
```
`--reset-proxy` nulls the proxy maps (keeps the expensive seed-0 full maps), recomputes the proxy
with the warm+frozen head, then re-correlates. Read `…cp24_warmstart.json.verdict.json`:
- **τ ≥ 0.7 & Δ ≤ 0.005** → CP 2.4 **closes** (advance state + `procedure.md`).
- **miss** → run the (already-built) `python -m eval.proxy_rank --diagnose-full --indices 7,4,8
  --full-epochs 100 --device cuda` to decide repair-more vs **reframe** (D4 → ask the user).

Donor note: the **gate** checkpoint (nc=1, 8-kpt) makes the whole head transfer + freeze cleanly;
with only generic COCO `yolo11n-pose.pt` (17-kpt) the keypoint branch reinitializes → run **without**
`--freeze-head`. CPU-runnable in parallel: anchor the **baseline yolo11n-pose** mAP via
`detect.evaluate.pose_map` (latency half stays Jetson-gated).

### Open design decisions (do not resolve unilaterally)

| ID | Decision | Blocks |
|---|---|---|
| ~~D1~~ | **RESOLVED 2026-06-18 → gate-pose** (`dataset/`; OFA backbone + YOLO11-pose head) | — |
| D2 | Search budget (default: 100 candidates Phase 3, 200 Phase 7) | CP 3.2 / 7.2 |
| D3 | Which SOTA blocks to inject (FusedMBConv, ConvNeXt, MobileViT) | CP 5.3 |
| D4 | λ, μ in `J(α) = acc − λ·latency − μ·max(0, mem−budget)²` | CP 3.3 |
| D5 | Multi-device extension (out of scope for v1/v2) | v3 |

---

## Module structure

```
catalog/      Block registry (shared by lut/ and all NAS phases)
              + flops.py (shared FLOPs counter), contracts.py (TypedDicts)
lut/          Phase 0: Jetson LUT pipeline (DONE)
  loader.py   Validated LUT reading (precision filter) — CP 2.2's input surface
  export/     PyTorch → ONNX
  bench/      Jetson-side TRT engine build + benchmarking (runs in Docker)
  orchestrate/ Laptop-side sweep loop + SSH orchestration
  docs/       lut.jsonl + device_info.json schema
supernet/     Phase 1: OFA-MBv3-w1.0 wrapper + sampler; pose_backbone.py
              (OFA subnet → P3/P4/P5 taps for the pose head)
search/       Phase 2: arch_to_blocks + cost.py (LUT composite cost); search loop (Phase 3)
detect/       D1 pose pivot: OFA-backbone → YOLO11-pose-head graft (adapter.py,
              pose_model.py: graft + warm_start_head/freeze_module) + pose-mAP eval (evaluate.py)
eval/         Eval harness: imagenet_sanity.py (CP 1.4); shortft.py + proxy_rank.py (CP 2.4 —
              fine-tune + DoD driver + --diagnose-full noise floor + --reset-proxy warm-head re-test)
net2net/      Phase 4: Net2Net operators stub
expand/       Phase 5–6: supernet expansion stub
distill/      Phase 8: KD harness (teacher = bigger yolo11-pose) stub
state/        Checkpoint tracking (plan_state.yaml)
data/         lut.jsonl + device_info.json (gitignored)
dataset/      D1 target: gate-pose data (gitignored; SCHEMA.md tracked)
tests/        Contract + regression tests (run via scripts/check.sh)
scripts/      Setup scripts + check.sh (ruff + mypy + pytest)
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

## Tests & tooling

- `bash scripts/check.sh` = ruff + mypy + pytest (uses `.venv`); append
  `-m "not slow"` for the ~3 s fast lane. CI (`.github/workflows/ci.yml`)
  runs the same on every push.
- **The golden hashes in `tests/test_row_key.py` ARE the LUT contract.** A
  failing golden means the change re-keys every measured Jetson row. Never
  update them without recording the decision in `procedure.md`.
- ofa-dependent tests auto-skip outside `.venv-nas`; LUT-file tests skip when
  `data/lut.jsonl` is absent (e.g. CI), and the catalog-coverage gate skips
  (with a coverage count) while real collection is still partial.
- Invoke venv tools as `python -m <tool>`, never via `bin/` entry points —
  the venvs' script shebangs went stale when the repo directory moved.
  `check.sh` also unsets `PYTHONPATH` (ROS's setup.bash leaks pytest plugins
  into venvs on this machine).

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
- **Commit after every code change.** Make a git commit after each logical code
  change (descriptive message, one concern per commit). This is a standing
  authorization — don't wait to be asked. Do **not** add Claude as a co-author
  or include any `Co-Authored-By:` / "Generated with Claude Code" trailer in the
  commit message.

---

## Things NOT to do

- `pip install -r requirements-nas.txt` directly → use `setup_laptop_nas.sh`.
- Edit `data/lut.jsonl` by hand.
- Update the golden hashes in `tests/test_row_key.py` without a decision
  recorded in `procedure.md` — they pin the LUT key contract.
- Commit anything in `data/` (it's gitignored for a reason — 50+ MB).
- Resolve open decisions D2–D5 without a user conversation (D1 is resolved).
- Name a local Python package `ofa/` — it shadows the pip-installed OFA library.
  The wrapper is in `supernet/` for this reason.
- Add Claude as a commit co-author, or include `Co-Authored-By:` /
  "Generated with Claude Code" trailers in commit messages.
