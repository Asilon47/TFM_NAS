# CPU cross-device rank check — design

**Date:** 2026-07-15
**Status:** awaiting review
**Owner decision this serves:** is the Orin's architecture ranking a property of the Orin, or a law?

---

## 1. Purpose

Every latency in this repo is a Jetson Orin Nano / TensorRT number. The central negative
result of CP 6.2-G is that **every OFA graft is strictly dominated** — even pruned to 84 %
sparsity — and the diagnosis is that grafts are **memory-bound**: measured backbone latency is
1.236× the LUT sum at 640, DRAM-bandwidth-limited.

That diagnosis has never been tested against a second memory system. This bench answers one
question:

> Does the Orin's ordering of these architectures survive on an x86 CPU with a different
> compute/bandwidth balance — and does the graft penalty grow as bandwidth pressure rises?

Two outcomes, both publishable:

- **Ranking holds, penalty grows with thread count** → memory-boundness is a property of the
  *architectures*, not of the Orin. The CP 6.2-G rejection generalises and hardens.
- **Ranking inverts at low thread counts** → the graft penalty is an Orin artifact. The
  rejection is device-specific and must be caveated as such in the thesis.

**Non-goals.** Not a second deployment target (D5 stays out of scope). Not an accuracy
measurement — accuracy is already recorded per model and is device-invariant. Not an absolute
"x86 runs at X ms" claim.

## 2. Hardware under test

Intel Core Ultra 9 185H, 31 GiB RAM, on AC.

| Property | Value | Consequence |
|---|---|---|
| P-cores | 6 physical, CPUs 0–11 (SMT) | Caps the clean sweep at 6 threads |
| HT sibling pairs | `{0,5} {1,2} {3,4} {6,7} {8,9} {10,11}` | `taskset -c 0-5` would hit **3** physical cores, all HT-contended |
| One-per-physical-P-core set | `{0, 1, 3, 6, 8, 10}` | The affinity list the sweep draws from, in order |
| E-cores | CPUs 12–19 @ 3.8 GHz | Slower cores — adding them confounds thread-count scaling |
| LP-E-cores | CPUs 20–21 @ 2.5 GHz | Same |
| Governor | `powersave` (intel_pstate), turbo on | Recorded, not changed — see §5 |

## 3. Architecture

Three units, mirroring the existing `run_bench` / `run_sweep` / `measure_additivity` split. The
measured device happens to be this laptop, so the "device-side timing" role lands locally.

### 3.1 `lut/bench/cpu_ort.py` — timing core

```
time_model(onnx_path, threads, affinity, iters, warmup) -> LatencyStats
```

Pure and dumb. Knows nothing about the sweep, the frontier, or the Jetson. Creates an ORT
session with `intra_op_num_threads=threads`, `inter_op_num_threads=1`, applies affinity via
`os.sched_setaffinity`, warms up, times `session.run` per iteration with
`time.perf_counter_ns`, returns `{mean, std, p50, p95, n}`.

Input tensor: fixed `np.random.default_rng(0)` NCHW float32 `(1,3,640,640)`, reused across all
iterations and models. Conv nets have no data-dependent control flow, and these ONNX exports
are network-only (no NMS), so a constant input is sound and removes a noise source.

Testable in milliseconds against a tiny synthetic ONNX — no 25-minute run required.

### 3.2 `lut/orchestrate/cpu_bench.py` — driver

Owns methodology. Discovers the model set from the pair map (§4), builds the config matrix,
runs the interleaved schedule (§5), writes one JSON row per `(model, config)` to
`data/cpu/<name>__<config>.json`. Idempotent and resumable — skips rows already on disk, same
contract as `run_sweep`.

### 3.3 `lut/orchestrate/cpu_rank_report.py` — analysis

Joins `data/cpu/*.json` against `data/e2e/*.json` on the pair map. Emits
`data/cpu/rank_report.json` + a printed table:

- Spearman ρ and Kendall τ-b, CPU-vs-Jetson, **per thread count**
- Per-family mean rank delta (which families move, and which direction)
- The headline curve: **graft penalty vs thread count**, defined concretely below

**Defining the graft penalty.** "Penalty at matched params" needs to be arithmetic, not a
gesture. For each config, fit an ordinary least-squares line of `latency_ms` on `params` across
a **reference set = `dense_scaled` (7) + `prune` (7) + `baseline` (1) = 15 models**, spanning
**0.9–3.9 M params**. The penalty for a graft model is its **residual** against that line — how
many ms it costs beyond what its parameter count predicts. The reported curve is the mean graft
residual per config, `t1 → t2 → t4 → t6`.

**Why the anchor is excluded from the fit:** `yolo11s` at 9.7 M params is 2.5× the next-largest
reference model. As an OLS leverage point it would dominate the slope, and every graft residual
would then be an artifact of one distant model. The grafts span ~1.1–2.5 M, comfortably *inside*
the 0.9–3.9 M reference range, so this is interpolation. The anchor is still benched and still
enters the Spearman — it is only kept out of the regression.

`dense_nas` is likewise reported but excluded from the reference fit: it is a distinct design
process, and the reference set should be the two families whose latency/params relationship the
project already trusts.

This makes the two outcomes numerically distinguishable: a residual that **grows** with thread
count means grafts lose *more* as bandwidth pressure rises (memory-bound, generalises); a
residual near **zero at `t1`** that only appears under pressure means the penalty is about the
memory system, not the architecture's FLOPs — and a residual that is **flat and positive
everywhere** would instead suggest a runtime/kernel-coverage effect (threat #4), not bandwidth.

`params` comes from the ONNX initialisers, counted once at load — not from `models/README.md`,
whose values are rounded and partly approximate (`~2.4M`).

## 4. The pair map — reviewable data, not a heuristic

Jetson row names do not match ONNX filenames and **cannot be derived**. An auto-matcher would
mispair models and still produce a plausible-looking ρ. The map is declared explicitly and
reviewed:

| Jetson row (`data/e2e/`) | ONNX (`models/`) | family |
|---|---|---|
| `baseline_recheck_640` | `baseline/yolo11n_pose_640.onnx` | baseline |
| `yolo11s_pose_640` | `anchor/yolo11s_pose_640.onnx` | anchor |
| `dense_ctrl_n_640` | `dense_scaled/dense_w25_640.onnx` | dense |
| `dense_d33_w20_640` | `dense_scaled/dense_w20_640.onnx` | dense |
| `dense_d50_w15_640` | `dense_scaled/dense_w15_640.onnx` | dense |
| `dense_w13_640` | `dense_scaled/dense_w13_640.onnx` | dense |
| `dense_w18_640` | `dense_scaled/dense_w18_640.onnx` | dense |
| `dense_w22_640` | `dense_scaled/dense_w22_640.onnx` | dense |
| `dense_w30_640` | `dense_scaled/dense_w30_640.onnx` | dense |
| `densenas_s31_640` | `dense_nas/dense_s31-40-40-40-13_o100_640.onnx` | dense_nas |
| `densenas_s39_640` | `dense_nas/dense_s39-40-38-38-14_o100_640.onnx` | dense_nas |
| `densenas_s40_640` | `dense_nas/dense_s40-38-39-36-13_o100_640.onnx` | dense_nas |
| `prune_base_r10_640` | `pruned_baseline/prune_r10_640.onnx` | prune |
| `prune_base_r15_640` | `pruned_baseline/prune_r15_640.onnx` | prune |
| `prune_base_r20_640` | `pruned_baseline/prune_r20_640.onnx` | prune |
| `prune_base_r30_640` | `pruned_baseline/prune_r30_640.onnx` | prune |
| `prune_base_r35_640` | `pruned_baseline/prune_r35_640.onnx` | prune |
| `prune_base_r45_640` | `pruned_baseline/prune_r45_640.onnx` | prune |
| `prune_base_r55_640` | `pruned_baseline/prune_r55_640.onnx` | prune |
| `winner_v1_e2e_640` | `graft/winner_v1_noneck_e2e_640.onnx` | graft |
| `winner_v1_v2topdown_e2e_640` | `graft/winner_v1_v2topdown_e2e_640.onnx` | graft |
| `winner_v1_v3pan_e2e_640` | `graft/winner_v1_v3pan_e2e_640.onnx` | graft |
| `graft_prune_r20_e2e_640` | `screen_prune_graft/graft_prune_r20_e2e_640.onnx` | graft_pruned |
| `graft_prune_r40_e2e_640` | `screen_prune_graft/graft_prune_r40_e2e_640.onnx` | graft_pruned |
| `graft_prune_r60_e2e_640` | `screen_prune_graft/graft_prune_r60_e2e_640.onnx` | graft_pruned |
| `graft_r50_gtay_640` | `graft_pruned/recover_graft_r50_gtay_640.onnx` | graft_pruned |
| `graft_r60_gtay_640` | `graft_pruned/recover_graft_r60_gtay_640.onnx` | graft_pruned |
| `graft_halp_9p0_640` | `graft_pruned/recover_graft_halp_fp32_9p0_640.onnx` | graft_pruned |
| `graft_halp_10p4_640` | `graft_pruned/recover_graft_halp_fp32_10p4_640.onnx` | graft_pruned |

**28 pairs.** Validated at load: every ONNX must exist and every Jetson row must parse, else
hard fail — a silently short pair list would weaken ρ without announcing itself.

**Deliberate exclusions.**

- `*_backbone_640` rows (`addv_idx*`, `addv_{min,max}_corner`, `winner_v1_backbone`) — backbone
  scope, not comparable to e2e networks. Mixing scopes is the category error behind the retired
  "12 % faster" claim.
- `fallback_idx3_e2e_640`, `fallback_idx11_e2e_640` — no local ONNX.
- `models/graft_pruned/recover_graft_r40_640.onnx` and `recover_graft_r60_640.onnx` — **name
  collision risk**: `screen_prune_graft/graft_prune_r{40,60}_e2e_640.onnx` are the artifacts
  whose Jetson latencies (11.81 / 9.00) match `models/README.md`'s CP 6.2-G rungs, so those are
  the mapped ones. The `recover_graft_*` pair is left out rather than guessed at.

## 5. Methodology

### 5.1 Config matrix

fp32 only, 5 configs per model:

| config | threads | affinity | role |
|---|---|---|---|
| `t1` | 1 | `{0}` | compute-bound end of the sweep |
| `t2` | 2 | `{0,1}` | clean sweep |
| `t4` | 4 | `{0,1,3,6}` | clean sweep |
| `t6` | 6 | `{0,1,3,6,8,10}` | bandwidth-pressured end, P-cores only |
| `all22` | default pool | unpinned | practical "what this PC does" number |

**Why fp32 only.** ORT's CPU fp16 is emulated — an fp16 column would measure emulation
overhead, not the models. It also matches `models/README.md` treating fp32 as the reliable axis
(Jetson fp16 carries ±20 % TRT build variance).

**Why `t1..t6` are P-core-only.** Core type must stay constant or thread count is not an honest
variable: a 6→8 thread step adds *slower* cores, so latency would change for a reason unrelated
to bandwidth pressure — confounding precisely the effect under test.

**`all22` is confounded by design** (E-cores are slower *and* add pressure). It is reported as a
practical number and **excluded from the clean scaling claim**.

### 5.2 Round-robin interleaving — the load-bearing choice

Not model-at-a-time. The schedule is `R` rounds; each round times **every** model once, rotating
the starting offset per round.

A laptop heats up over a ~25-minute run — the x86 analog of the unlocked-clock failure
`setup_jetson.sh` exists to prevent, and there is no equivalent lock available. Under
model-at-a-time, thermal drift becomes a systematic bias that tracks *bench order*; since the
natural iteration order is by family, that bias would correlate with architecture family and
could **manufacture the very effect the experiment tests for**. Interleaving spreads drift
across all models as approximately common-mode noise.

Per `(model, config)` cell: warmup 5 iterations, then `R × k` timed iterations pooled into the
row's stats. **Defaults `R = 10` rounds × `k = 6` iterations → `n = 60` per cell**, both
`--rounds` / `--iters` overridable. `n = 60` is well short of the Jetson rows' `n = 200`, which
is accepted deliberately: CPU per-iteration cost is ~10–50× the Jetson's, and a rank check needs
a stable p50 rather than a tight absolute CI.

**Budget:** 28 models × 5 configs = **140 cells**. At an assumed 30–150 ms/iteration (fp32 @640,
1–6 threads) and 65 iterations/cell, the run is ~20–40 minutes. The driver prints a measured ETA
after round 1 and the run is resumable, so a bad estimate costs nothing.

### 5.3 Canary drift detection

`baseline_recheck_640` is re-timed every round. If its round-mean p50 drifts > 5 % from round 1,
every row in the run is stamped `thermal_drift_detected: true`. The run is not aborted — a
drifted run is still readable, it just must not be quoted as clean.

### 5.4 Governor

Left at `powersave`, recorded per row. Rationale: on `intel_pstate`, `powersave` still ramps to
turbo under sustained load (unlike the old `ondemand`); the residual difference is ramp latency,
which the per-cell warmup absorbs. More decisively — a clock offset applying to *every* model is
near-monotone across the latency column, and **Spearman is invariant to monotone transforms**.
Governor state matters for absolute claims, not for a rank check. Revisit only if absolute x86
numbers are ever wanted.

## 6. Row schema

Mirrors `data/e2e/*.json` so downstream tooling reads both. Jetson provenance fields are
replaced by x86 ones.

```json
{
  "name": "prune_base_r20_640",
  "config": "t4",
  "precision": "fp32",
  "imgsz": 640,
  "latency_ms": {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "n": 60},
  "threads": 4,
  "affinity": [0, 1, 3, 6],
  "governor": "powersave",
  "cpu_mhz_mean": 0.0,
  "loadavg_1m": 0.0,
  "on_ac": true,
  "thermal_drift_detected": false,
  "ort_version": "1.27.0",
  "cpu_model": "Intel(R) Core(TM) Ultra 9 185H",
  "source": "x86_ort",
  "timestamp": "2026-07-15T00:00:00Z"
}
```

`source: "x86_ort"` keeps these rows unmistakable against `jetson_trt` — no CPU number can ever
be mistaken for a device latency in a later claim.

## 7. Error handling

| Failure | Behaviour |
|---|---|
| ONNX missing / unparseable at startup | Hard fail before any timing — never bench a short set silently |
| Session creation fails for one model | Record `status: "fail"` + reason, continue; report lists it |
| Row already on disk | Skip (resumable) |
| Canary drift > 5 % | Stamp all rows, continue, report loudly |
| Jetson pair missing at report time | Exclude from ρ, list explicitly in report |

## 8. Testing

Unit tests in `tests/test_cpu_bench.py`, fast lane (no real bench):

- **Affinity resolver** — `physical_p_cores()` returns `{0,1,3,6,8,10}` from the sibling map, and
  specifically *not* `0-5`. This pins the gotcha that would otherwise silently measure HT
  contention as scaling.
- **Interleave scheduler** — every model appears exactly once per round; offset rotates; canary
  present each round.
- **Row schema** — emitted row validates against the `data/e2e` shape.
- **Pair map** — every ONNX path exists; no duplicate ONNX mapped to two rows.
- **Report math** — Spearman on a known-order fixture returns 1.0.

Timing core tested against a synthetic 2-layer ONNX, not the real frontier.

## 9. Threats to validity (to be recorded with the result)

1. **Clocks cannot be locked** as on the Jetson. Mitigated by interleaving + canary + AC power;
   not eliminated.
2. **`all22` is confounded** (core type × pressure) — excluded from the scaling claim.
3. **Single machine, single run** — a rank inversion near the noise floor would need repeat runs
   at fresh rounds before being quoted. Same discipline as the CP 3.5 winner's-curse finding.
4. **ORT ≠ TensorRT.** Graph optimisation and kernel selection differ; some of any rank delta is
   *runtime*, not *device*. This bench cannot separate those two. A clean separation would need
   ORT-on-Jetson or TRT-on-x86 — out of scope, and named as a limitation.
5. **Accuracy is not apples-to-apples** across families (per `models/README.md`) — irrelevant
   here, since only latency ranks are compared, but must not be forgotten if the report is later
   read as a Pareto statement.
6. **The penalty metric assumes latency is ~linear in params** over the reference range. It is
   a convenience, not a law — params ignores activation volume, which is exactly the quantity a
   memory-bound argument cares about. The residual is therefore a *screening* statistic; if the
   headline curve is the result, the report should also show the raw latency-vs-params scatter
   per config so the linearity can be judged by eye rather than trusted.

## 10. Deliverables

- `lut/bench/cpu_ort.py`, `lut/orchestrate/cpu_bench.py`, `lut/orchestrate/cpu_rank_report.py`
- `tests/test_cpu_bench.py`
- `data/cpu/*.json` (gitignored, like all of `data/`) + `data/cpu/rank_report.json`
- A `procedure.md` entry recording the result and its threats
