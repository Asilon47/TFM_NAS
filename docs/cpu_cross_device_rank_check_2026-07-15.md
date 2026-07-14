# CPU cross-device rank check — result (2026-07-15)

**Question:** is the Orin's architecture ranking — and the CP 6.2-G verdict that grafts are
memory-bound and strictly dominated — a property of the *architectures*, or of the *Orin*?

**Answer: it generalises.** The Jetson's ordering reproduces on an x86 CPU (Spearman ρ = 0.88–0.94),
and the graft penalty **grows with bandwidth pressure** on completely different silicon: 5.9 % of
predicted latency at 1 thread → 14.4 % at 6 threads. The `dense`/`prune` families show no such
growth. The memory-boundness diagnosis is not an Orin artifact.

**What this does *not* say:** nothing here is a deployment number. Every latency below is
`x86_ort` and is never comparable to the `jetson_trt` figures in `models/README.md`. This is a
ranking experiment.

---

## Method

| | |
|---|---|
| Machine | Intel Core Ultra 9 185H, 31 GiB, on AC, governor `powersave` (intel_pstate; measured boosting to 4.5 GHz under load) |
| Runtime | onnxruntime 1.27.0, CPUExecutionProvider, **fp32 only** (ORT CPU fp16 is emulated) |
| Models | 29 e2e networks @640 with a measured Jetson fp32 pair (`lut/orchestrate/cpu_pairs.py`) |
| Configs | `t1/t2/t4/t6` pinned one-thread-per-**physical** P-core `{0,1,3,6,8,10}`, plus `all22` (default pool, unpinned) |
| Sampling | 10 rounds × 6 iterations = **n=60**/cell, round-robin interleaved with rotated order |
| Canary | `baseline_recheck_640` re-timed every round; >5 % p50 drift stamps the rows |

Only 6 physical P-cores exist, so the clean sweep stops at 6: an 8-thread step would add *slower*
E-cores, confounding thread count with core type. `all22` is reported but excluded from the
scaling claim for exactly that reason.

The sibling pairs on this part are `{0,5} {1,2} {3,4} {6,7} {8,9} {10,11}` — the obvious
`taskset -c 0-5` would occupy **three** physical cores with every hyperthread contended and
measure HT contention as thread scaling.

## Results

Latency scaling is sane, which is the first thing to check: yolo11n falls 77.1 → 29.4 ms from 1
to 6 threads, and `all22` (37.3 ms) is *slower* than `t6` — the predicted E-core confound,
visible in the data.

| config | ρ vs Jetson | τ-b | yolo11n (ms) | graft penalty (ms) | graft penalty (% of predicted) |
|---|---|---|---|---|---|
| `t1` | 0.888 | 0.714 | 77.1 | 1.04 | **5.9 %** |
| `t2` | 0.912 | 0.749 | 49.3 | 2.76 | 10.0 % |
| `t4` | 0.934 | 0.788 | 32.8 | 2.41 | 13.2 % |
| `t6` | 0.939 | 0.803 | 29.4 | 2.94 | **14.4 %** |
| `all22` | 0.940 | 0.793 | 37.3 | 4.07 | 15.3 % *(confounded)* |

n = 29 for every row.

**Penalty definition.** For each config, OLS-fit `latency ~ params` over a reference set of
`dense_scaled` (7) + `prune` (7) + `baseline` (1) = 15 models spanning 0.9–3.9 M params. A
model's penalty is its **residual** — ms beyond what its parameter count predicts. `yolo11s`
(9.7 M) is excluded from the fit as a leverage point that would otherwise dominate the slope; it
still enters the correlation. Grafts span ~1.1–2.5 M, inside the reference range, so this is
interpolation.

### The discriminating result: residual slope by family

| family | `t1` residual | `t6` residual | Δ |
|---|---|---|---|
| `dense_nas` | 48.9 % | 37.5 % | **−11.5** |
| `graft_pruned` | 7.0 % | 16.6 % | **+9.6** |
| `graft` | 3.2 % | 9.5 % | **+6.2** |
| `dense` | 5.4 % | 7.0 % | +1.6 |
| `prune` | −4.8 % | −6.0 % | −1.2 |
| `baseline` | −1.7 % | −6.9 % | −5.1 |
| `anchor` | −8.7 % | −11.7 % | −3.0 |

The **sign of Δ separates two different failure modes**:

- **Grafts (`graft`, `graft_pruned`) start near the reference line and climb under pressure.**
  At 1 thread — compute-bound, memory latency hidden behind arithmetic — a graft costs only
  3.2 % over its size-predicted latency. Add cores until DRAM is the constraint and it costs
  9.5 %; pruned grafts climb hardest, 7.0 → 16.6 %. That is the memory-bound signature, on a
  memory system with nothing in common with the Orin's.
- **`dense_nas` is the mirror image**: a large 48.9 % penalty at 1 thread that *shrinks* to
  37.5 % with parallelism — a compute-heavy penalty that parallelises away.

So `dense_nas` and the grafts are both "slow", for opposite reasons. A single-config bench would
have reported one number and hidden the distinction.

Note also that ρ **rises** with thread count (0.888 → 0.939): the more bandwidth-bound the x86
becomes, the more its ordering agrees with the Orin's. That is consistent with the Orin being
bandwidth-bound at 640, and is independent corroboration of the CP 6.2-G diagnosis.

### Reproducibility

`t1` and `t6` were re-measured end-to-end in a second independent run:

| | ρ run1 | ρ run2 | penalty run1 | penalty run2 | yolo11n run1 | yolo11n run2 |
|---|---|---|---|---|---|---|
| `t1` | 0.888 | 0.884 | 5.9 % | 6.1 % | 77.1 | 77.4 |
| `t6` | 0.939 | 0.936 | 14.4 % | 13.6 % | 29.4 | 30.4 |

The t1→t6 penalty roughly doubles in both runs. Run-to-run spread (~0.8 pt) is far smaller than
the effect (~8 pt).

## Methodology finding: ORT spin-wait corrupts interleaved benching

**The first complete run was wrong, and wrong in the shape of the hypothesis.** It reported the
graft penalty rising 1.15 → 6.12 ms with thread count — which looked like a clean confirmation.
It was an artifact.

ORT's intra-op workers **spin-wait** after `run()` returns rather than sleeping, to shave wake-up
latency for a server calling *one* session in a loop. This bench rotates 29 live sessions, so
every model's run collided with the previous models' pools still burning the same pinned cores.
Measured at `t6`, 29 sessions held, interleaved access:

| | t6 baseline p50 |
|---|---|
| spinning ON | **564.5 ms** |
| spinning OFF | **30.4 ms** |
| isolated reference (no interleaving) | 20.3 ms |

An **18.6× artifact that grows with thread count** — indistinguishable in shape from the
bandwidth effect under test. The tell was the raw scaling column: yolo11n appeared to get
*slower* with more cores (77 → 117 ms), which is physically impossible and is what exposed it.

Two traps worth recording:

1. **Spinning and interleaving interact; neither alone shows the fault.** A first test of
   spinning came back negative because it timed one session back-to-back — the spin-wait was
   always helping the session that owned it. The confound was in the test.
2. Disabling spinning also removed most of the thermal drift, because the spin storm was what
   was cooking the CPU.

The fix is `session.intra_op.allow_spinning = 0`, pinned by `ALLOW_SPINNING` in
`lut/orchestrate/cpu_ort.py` and guarded by `tests/test_cpu_ort.py::test_spinning_stays_disabled`.

## Threats to validity

1. **Thermal drift.** `t2/t4/t6/all22` are stamped `thermal_drift_detected` (canary p50 moved
   >5 % from round 0); only `t1` is clean. Clocks cannot be locked as on the Jetson — this
   laptop reaches ~99 °C under sustained load regardless of start temperature. The rank claim
   survives because **interleaving makes drift common-mode**: every model appears exactly once
   per round, so each model's pooled p50 averages over the same thermal states, and a
   config-wide inflation is absorbed by the OLS intercept rather than by any one family. The
   **absolute** millisecond figures above are therefore not clean numbers and must not be quoted
   as such.
2. **ORT ≠ TensorRT.** Graph optimisation and kernel selection differ, so part of any rank delta
   is *runtime*, not *device*. This bench cannot separate them; a clean separation needs
   ORT-on-Jetson or TRT-on-x86. This is the hardest limit on how far the conclusion can be
   pushed.
3. **`all22` is confounded by construction** (E-cores are slower *and* add pressure) — reported,
   excluded from the scaling claim.
4. **The penalty metric assumes latency is ~linear in params**, and params ignores activation
   volume — which is precisely what a memory-bound argument is about. The residual is a
   *screening* statistic; the per-config scatter is in `data/cpu/rank_report.json` for anyone who
   wants to judge the linearity rather than trust it.
5. **Single machine.** One x86 part, one memory system. "Reproduces on a second device" is not
   "reproduces on all devices".

## Artifacts

- `data/cpu/*.json` — 145 rows (29 models × 5 configs), `source: "x86_ort"`, gitignored
- `data/cpu/rank_report.json` — ρ/τ, reference fits, per-model residuals, per-config scatter
- Regenerate: `python -m lut.orchestrate.cpu_bench --configs <cfg>` then
  `python -m lut.orchestrate.cpu_rank_report` (`.venv`)
- Spec: `docs/superpowers/specs/2026-07-15-cpu-cross-device-rank-check-design.md`
