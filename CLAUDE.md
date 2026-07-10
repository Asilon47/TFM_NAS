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
Optimization** guide the search. A warm-started, **frozen gate-trained head**
makes each evaluation a ~5-epoch fine-tune instead of full training (**Net2Net**
transforms now serve the post-search graft refinement, Phase 5). Once the search
settles, the winner is refined at the graft seam, pruned, and **distilled**
against a bigger **YOLO11-pose** teacher for its final deployable weights
(Phase 8).

---

## How to resume a session

1. Read `state/plan_state.yaml` → tells you the current checkpoint.
2. Read the relevant section of `procedure.md` → tells you what was done and
   why at each completed checkpoint.
3. Read `PROJECT_PLAN.md` → the next checkpoint's inputs, deliverables, and DoD.
4. Activate the right venv (see below) and verify the DoD commands still pass.

---

## Current state (as of last update: STAGE 0 + CP 5.2 + Phase 3b DONE — all measurement/search contingencies closed; next = user framing decision + CP 5.3, 2026-07-07)

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
- **CP 2.4 (eval harness — CLOSED 2026-06-27 on the reframed gate):** The CPU slice (graft
  `detect.pose_model.GraftedPoseModel` + `detect.evaluate.pose_map_model` + `eval/shortft.py`
  harness + the `eval/proxy_rank.py` driver) plus the head warm-start+freeze repair
  (`warm_start_head`/`freeze_module`, `build_grafted_pose_model(head_weights=, freeze_head=)`). The
  first GPU run failed both old DoDs (Kendall-τ=0.20) — root cause = the **randomly-initialized Pose
  head** (LP-FT distortion; idx8 = best backbone, worst proxy). The **warm-head re-test** (Colab,
  `data/cp24_warmstart.json`: warm-start + **freeze** the trained gate head, **3-seed** avg) fixed
  it: **Spearman ρ=0.77, top-1 regret 0.0 → PASS** (Kendall-τ=0.60, repro-Δ=0.0145 are now
  diagnostics). **DoD reframed (D4-adjacent, user-approved): Spearman ρ≥0.70 AND top-1 regret≤0.01**
  (`eval.shortft.rank_verdict`), superseding τ-on-10 + Δ≤0.005 (which mis-measures — size descriptors
  fail τ yet pick the true best). Cross-checked **no-GPU** by `eval/zerocost.py` (depth_sum ρ=0.843,
  regret 0). The close was CPU-only: `assemble_verdict` now gates on `rank_verdict`, new
  `reverdict()`/`--reverdict` re-stamps the verdict offline (266 tests green). Carries into Phase 3:
  warm-head proxy = accuracy signal, zero-cost = free cold-start prefilter; **J(α) formulation resolved
  (D4 → Pareto + hard latency ceiling; `search/objective.py`); λ/μ *numbers* calibrated at CP 3.3**. See
  `procedure.md` "CP 2.4 CLOSED — warm-head re-test + reframe gate" + "D4 RESOLVED".
- **CP 3.1:** CLOSED (2026-06-27) — `search/space.py`: OFA arch_dict ↔ flat surrogate vector;
  `decode(encode(arch))==arch` 100/100. Tracks categorical ks/e vs ordinal depth + `canonical`
  masks depth-inactive don't-cares.
- **CP 3.2:** CLOSED (2026-06-27) — `search/evolution.py`: pymoo NSGA-II over `(depth_sum, latency_ms)`,
  11-point depth-staircase frontier (≥10 DoD). The reusable structural baseline + BO warm-start seeds
  (`data/phase3_nsga2_frontier.json`).
- **CP 3.3 (BO):** CLOSED — `search/bo.py` (ParEGO + BoTorch `MixedSingleTaskGP` + `qLogEI`; latency
  LUT-exact, only accuracy GP-modeled; hard ceiling pre-filters). The @640 LUT (2710→2801 rows) + the
  yolo11n-pose anchor landed (`data/baseline_anchor.json`: **12.755 ms** → `T_max=12.75`); 5-seed
  warm-head DoD on Kaggle: **BO HV 3.441±0.022 vs cold random 2.088±0.211**, wins 5/5. Scope caveat
  (recorded at CP 3.4): warm-random ≈ BO/TPE, so the claim is "warm-started pipeline > cold random",
  not "the acquisition is the driver". See `procedure.md` "CP 3.3".
- **CP 3.4 (TPE):** CLOSED (2026-07-04) — `search/tpe.py` (Optuna MOTPE) reproduces BO within 0.8 %
  (HV 3.414±0.023); the warm-random ablation (3.403) is the recorded threat-to-validity. Anchor B:
  yolo11s-pose **0.8819 @ 21.69 ms** ⇒ the task is accuracy-saturated (+70 % latency → +0.5 mAP).
  See `procedure.md` "CP 3.4 CLOSED".
- **CP 3.5 (winner-v1):** CLOSED (2026-07-04) — **PHASE 3 COMPLETE.** The reproduce-DoD caught a
  single-seed **winner's curse** (α* cached 0.650 → fresh 3-seed 0.610, rejected); the top-12 were
  re-scored at seeds {1,2,3} (`search/denoise.py`) and the winner re-picked on de-noised means: the
  knee **d=[2,2,4,3,3] @ 11.208 ms** (backbone LUT sum), proxy mAP **0.6101±0.0049** — the "12 %
  faster than yolo11n" headline carries the Stage-0 caveat below. Tie-band sensitivity is
  reproducible via `python -m search.denoise_report`. Side experiment (`e2bfc17`): 100-epoch
  bare-AdamW full-FT of winner-v1 → **0.841** vs baseline 0.877 (single seed, recipe-confounded).
  See `procedure.md` "CP 3.5 CLOSED".
- **PLAN PIVOT (2026-07-05):** Phases 5–7 **re-scoped** (user decision; **D3 RESOLVED → descoped**).
  Out: supernet expansion + re-search (ImageNet-framed DoDs pre-dating the pose pivot; Phase-6
  supernet fine-tune free-tier-infeasible; proxy noise σ 0.005–0.025 exceeds the frontier's
  top-cluster gaps ~0.014 → an enriched space is unrankable by the oracle). In: **Phase 5 =
  graft-interface ablation → winner-v1.5** (the graft is neck-less with random-init 1×1 adapters —
  the accuracy seam; Net2Wider adapter init + zero-gated top-down "nano-neck"), **Phase 6 = DepGraph
  structured pruning → winner-v2** (measured-only latencies), **Phase 7 = recipe-parity training**
  (the 0.841-vs-0.877 gap is recipe-confounded; CP 7.2 is the baseline CP 8.3 must beat). Phase 4
  kept (CP 4.4 → graft-seam adapter init). **OWED FIRST (Stage 0):** the grafted winner was never
  benched end-to-end — 11.208 ms is backbone-blocks-only vs the baseline's full-network 12.755 ms;
  the pose stem/adapter/head offset is unmeasured (`data/stem_head_offset.json` is the *classifier*
  head @224). Export grafted + backbone-only ONNX → `bench_model` on the Nano (mode 0) →
  `data/pose_stem_head_offset.json` → additively re-stamp `winner.json` with the honest e2e speedup;
  fallbacks `[2,2,4,3,2]` / `[2,4,3,4,4]` benched in the same session. See `procedure.md` "Plan pivot".
- **Phases 4–9:** re-scoped plan in `PROJECT_PLAN.md` (rewritten 2026-07-05; the old expansion text
  lives in git history).

### Known blockers

- **CUDA / `.venv-nas` (laptop).** `torch.cuda.is_available()` is False (long-standing).
  **`.venv-nas` is BUILT (2026-07-05) as the CPU variant** — torch 2.11.0+cpu via
  `TORCH_CUDA_INDEX=cpu bash scripts/setup_laptop_nas.sh`; the disk could not fit the cu128
  stack (root fs was 100% full) and the laptop's CUDA is broken anyway, so nothing is lost:
  Stage 0's ONNX export runs locally on CPU. To restore the GPU variant later: free ~10 GB,
  then plain `bash scripts/setup_laptop_nas.sh`. GPU fine-tunes run remotely:
  **Kaggle (quota restored 2026-07-05)** is primary, Colab free T4 the backup, and the **AGX Orin**
  runs long trains via the `jetson/` Docker kit. The Orin Nano 8 GB stays measurement-only
  (TensorRT in Docker; mode 0 + locked clocks via `scripts/setup_jetson.sh` — since the JetPack
  update the board *idles* in the 25 W/918 MHz Super regime, so never bench without the setup
  script; every repo latency is a 612 MHz mode-0 number).
- **Stage 0 LANDED (2026-07-07) — the "12 % faster" claim is RETIRED, permanently.** Measured
  same-session on the Nano (mode 0): baseline 12.75 ms fp32 / **7.58 ms fp16**; winner-v1 e2e
  **17.69 ms fp32 / 12.37 ms fp16** (+38.7 % / +63 % SLOWER). Causes, quantified: the @224
  additivity calibration inverts at 640 (measured backbone = **1.236×** the LUT sum;
  DRAM-bound), and the pose adapter+head offset is **3.84 ms**
  (`data/pose_stem_head_offset.json`). No LUT-frontier arch beats the baseline e2e. The good
  number: **V3 (PAN nano-neck) fp16 = 12.75 ms = 78 FPS — meets the 60 FPS deployment bar**
  with 24 % headroom. LUT-summed latencies are **ranking-only** in every future claim; honest
  numbers live in `winner.json`'s `e2e` block + procedure.md "STAGE 0 COMPLETE".

### Lowest-friction next build

**ALL THREE DENSE-FAMILY CAMPAIGNS ARE IN (2026-07-08, user decision A+B1+B2). Results
pulled + recorded (procedure.md "CP 6.2-B" / "CP 3c.1"); the SOLE remaining blocker to the
cross-family verdict is ONE Nano bench session.** Accuracy (all from-scratch, single-seed):

| family (from-scratch) | mAP50-95 | artifact |
|---|---|---|
| dense w0.25 = ctrl_n (yolo11n arch) | 0.854 | `data/kaggle_out_asilarnous47/dense_scaling/` |
| graft v2topdown / v3pan (100ep) | 0.846 / 0.842 | `data/cp33_kaggle_out/full_finetune_v*` |
| dense w0.20 | 0.839 | dense_scaling |
| pruned-baseline r15 (−39 % params) | 0.834 | `data/kaggle_out_asilarnous/prune_baseline/` |
| dense w0.15 | 0.815 | dense_scaling |

(baseline pretrained anchor 0.877 → COCO-pretrain+recipe worth only ~2.3pts; CP 3c.1.) All four
families cluster **0.79–0.85 from-scratch → LATENCY, not accuracy, is the separator.**

1. **NEXT NANO SESSION benches EVERYTHING** (mode 0 / 612 MHz — `scripts/setup_jetson.sh`):
   winner-v1.5 e2e (both necks), the **3 pruned-baseline ONNX** (`prune_base_r{15,30,45}_640.onnx`),
   the **6 dense-wave ONNX** (`dense_*_640.onnx`), + deferred riders (SE ablation, 512-res sweep,
   FusedMBConv LUT screen). `lut.orchestrate.bench_model --imgsz 640` per model. This
   is the CP 3c.3 cross-family figure's x-axis. **(OFA-R50 screen DONE 2026-07-10 → INFEASIBLE:
   smallest R50 backbone 16.17ms fp32 > whole 12.75ms baseline; the "different supernet?" gate is
   closed by measurement — `models/screen_r50/`, procedure.md "OFA-ResNet50 screen".)**
   **(PRUNE-THE-GRAFT screen DONE 2026-07-10 → pruning the winner graft cuts latency, but in fp16
   only r=0.60 = 84% param sparsity beats the 7.75ms baseline (6.58ms); fp32 needs r≥0.40 (64%).
   From a 0.61 proxy → accuracy-dominated by the dense arm. The "prune the OFA?" gate is closed too
   — `models/screen_prune_graft/`, procedure.md "Prune-the-graft latency screen".)**
2. **THEN de-noise before ANY pick** (single-seed everywhere): the 3 distinct dense widths AND
   the prune rungs (CP 6.2-B is **non-monotonic** r45>r30 ⇒ recovery-noise-bound) at seeds
   {1,2,3} — CP 3.5 discipline. User owns winner-v1.5 / CP 6.3 point / CP 3c.2 wave-2 / framing.
- **Kaggle prune infra fixed 2026-07-08:** `prune/yolo_tp_prep.py` (splits C2f chunks, keeps
  C2PSA dense, trace@128) makes torch-pruning tractable on ANY yolo11/dense donor; and
  `load_baseline_model` resets the donor checkpoint's CPU-cached criterion + dict args (else a
  cuda/cpu loss mismatch). Both have regression tests. yolo11 depth mult is **degenerate below n**
  (C3k2 n-floor) — dense scaling is width-only; drop depth as a wave-2 knob.
Donor for every warm-head proxy: `runs/pose/experiments/gate_baseline/weights/best.pt` (nc=1/8-kpt →
whole head transfers + freezes cleanly; [[cp24-donor-must-be-trained]]).

### Open design decisions (do not resolve unilaterally)

| ID | Decision | Blocks |
|---|---|---|
| ~~D1~~ | **RESOLVED 2026-06-18 → gate-pose** (`dataset/`; OFA backbone + YOLO11-pose head) | — |
| ~~D2~~ | **RESOLVED 2026-06-27 → B=50** (CP 3.3 BO per-run budget; `5·(2B−n_init)`=400 warm-head fine-tunes / 5 seeds; NSGA-II free). Phase-7 budget → CP 7.2 | — |
| ~~D3~~ | **RESOLVED 2026-07-05 → descoped** (supernet expansion left the plan with the winner-refinement pivot; FusedMBConv survives only as the optional, evidence-only CP 5.0 LUT screen). See `procedure.md` "Plan pivot" | — |
| ~~D4~~ | **RESOLVED 2026-06-27 → Pareto + hard latency ceiling** (multi-objective `(acc_eff, latency)`, `latency ≤ T_max=min(baseline, 60 FPS→16.7 ms)`; soft μ² folded into `acc_eff`, budget 512 MiB; λ ParEGO-sampled). Formula in `search/objective.py`. **REFINED at CP 3.5 (2026-07-02) → ceiling-first**: winner selection is now λ-free (`select_winner.ceiling_first_winner` = max acc under `T_max`); the two-anchor iso-J λ is a *secant/linearising* estimate (λ≈0.001–0.002 acc/ms here), so it is demoted to a **robustness check** (`winner_is_lambda_stable`), not the decision. α* needs neither anchor. See `procedure.md` "CP 3.5 refinement". | — |
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
              + dense_family.py (Phase 3c: yolo11-pose scaling wave — the dense-family arm)
detect/       D1 pose pivot: OFA-backbone → YOLO11-pose-head graft (adapter.py,
              pose_model.py: graft + warm_start_head/freeze_module) + pose-mAP eval (evaluate.py)
eval/         Eval harness: imagenet_sanity.py (CP 1.4); shortft.py + proxy_rank.py (CP 2.4 —
              fine-tune + DoD driver + --diagnose-full noise floor + --reset-proxy warm-head re-test)
net2net/      Phase 4: Net2Net operators (wider/deeper/bn/graft_init — Stage 1)
expand/       Descoped expansion stub (2026-07-05) — will hold only the optional CP 5.0 screen
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
| `.venv-nas` | `source .venv-nas/bin/activate` | NAS pipeline (torch, ofa, torchvision, ultralytics — since 2026-07-05 the **CPU build**, see Known blockers) |

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
- Resolve open decision D5 without a user conversation (D1–D4 resolved).
- Name a local Python package `ofa/` — it shadows the pip-installed OFA library.
  The wrapper is in `supernet/` for this reason.
- Add Claude as a commit co-author, or include `Co-Authored-By:` /
  "Generated with Claude Code" trailers in commit messages.
