# CPU latency — the frontier measured on x86 (2026-07-15)

The `models/README.md` frontier, re-measured on this laptop's CPU. **These are not deployment
numbers**: every latency here is `x86_ort` and is never comparable to the `jetson_trt` column.
The point is what the *comparison between architectures* looks like on a different memory system.

**Machine:** Intel Core Ultra 9 185H (6 P-cores + 8 E-cores + 2 LP-E, 22 threads), 31 GiB, on AC.
**Runtime:** onnxruntime 1.27.0, CPUExecutionProvider, **fp32 only** (ORT CPU fp16 is emulated,
so an fp16 column would measure emulation, not the models). @640, batch 1, p50 of n=60.

**Configs:** `t1/t2/t4/t6` = 1/2/4/6 threads pinned one-per-**physical** P-core `{0,1,3,6,8,10}`;
`all22` = ORT's default pool, unpinned. Sibling pairs here are `{0,5} {1,2} {3,4} {6,7} {8,9}
{10,11}`, so `taskset -c 0-5` would occupy three cores with every hyperthread contended.

---

## Full frontier (sorted by accuracy, same order as `models/README.md`)

| family | model | params | mAP | Jetson fp32 | CPU t1 | CPU t2 | CPU t4 | CPU t6 | CPU all22 |
|---|---|---|---|---|---|---|---|---|---|
| anchor | yolo11s | 9.7M | 0.882 | 21.70 | 218.3 | 131.9 | 83.7 | 70.0 | 75.0 |
| **baseline** | **yolo11n** | 2.7M | **0.877** | **12.74** | **77.1** | **49.3** | **32.8** | **29.4** | **37.3** |
| search | s39-40-38-38-14 | 2.8M | 0.871 | 15.27 | 113.4 | 72.5 | 47.1 | 43.1 | 44.6 |
| search | s31-40-40-40-13 | 2.9M | 0.870 | 15.14 | 115.3 | 73.2 | 47.7 | 42.4 | 44.9 |
| search | s40-38-39-36-13 | 2.6M | 0.868 | 14.98 | 111.9 | 70.8 | 46.3 | 41.9 | 45.8 |
| dense | w30 | 3.9M | 0.856 | 15.26 | 104.1 | 66.6 | 44.0 | 39.6 | 43.5 |
| dense | w25 (ctrl_n) | 2.7M | 0.854 | 11.33 | 71.6 | 47.5 | 32.3 | 29.4 | 38.4 |
| graft | v2topdown | ~2.4M | 0.846 | 18.15 | 84.7 | 58.2 | 38.0 | 35.8 | 43.0 |
| dense | w22 | 2.3M | 0.845 | 11.54 | 69.6 | 46.9 | 31.9 | 30.1 | 37.4 |
| graft | v3pan | ~2.5M | 0.842 | 18.37 | 88.0 | 60.8 | 39.6 | 36.9 | 42.5 |
| graft | winner-v1 noneck | ~2.4M | 0.841 | 17.67 | 84.4 | 57.9 | 37.8 | 35.4 | 42.3 |
| dense | w20 | 1.9M | 0.839 | 11.26 | 61.2 | 42.5 | 28.7 | 27.7 | 36.6 |
| **prune** | **r20** | 1.6M | **0.838** | **9.52** | **45.9** | **32.3** | **22.1** | **21.0** | **28.2** |
| dense | w18 | 1.6M | 0.834 | 10.01 | 51.4 | 36.1 | 24.8 | 24.0 | 31.8 |
| prune | r15 | 1.6M | 0.834 | 9.53 | 46.4 | 32.5 | 22.3 | 21.4 | 27.4 |
| prune | r10 | 1.9M | 0.830 | 9.82 | 49.4 | 34.4 | 23.5 | 22.1 | 28.3 |
| prune | r35 | 1.1M | 0.826 | 8.36 | 33.7 | 24.8 | 17.7 | 17.7 | 24.3 |
| graft-pruned | r40 | 1.1M | 0.816 | 11.81 | 41.0 | 31.1 | 23.0 | 22.9 | 33.7 |
| dense | w15 | 1.2M | 0.815 | 9.52 | 44.2 | 31.5 | 22.3 | 22.6 | 28.5 |
| dense | w13 | 1.0M | 0.813 | 9.56 | 42.4 | 30.3 | 21.8 | 22.3 | 28.8 |
| graft-pruned | halp_10p4 | 2.4M | 0.813 | 12.58 | 51.4 | 37.6 | 27.0 | 26.8 | 34.1 |
| prune | r45 | 0.9M | 0.809 | 7.94 | 30.9 | 23.2 | 16.9 | 17.2 | 21.9 |
| graft-pruned | halp_9p0 | 1.8M | 0.802 | 11.37 | 43.6 | 32.8 | 24.5 | 24.5 | 33.8 |
| prune | r55 | 0.65M | 0.798 | 7.66 | 27.9 | 21.2 | 15.4 | 16.0 | 20.3 |
| graft-pruned | r50_gtay | 0.76M | 0.795 | 10.23 | 34.0 | 26.9 | 20.3 | 20.9 | 27.0 |
| prune | r30 | 1.1M | 0.790 | 8.28 | 33.4 | 24.6 | 17.9 | 17.3 | 22.6 |
| graft-pruned | r60_gtay | 0.44M | 0.777 | 8.84 | 26.9 | 21.8 | 17.9 | 18.8 | 25.5 |
| graft-pruned | r60 | 0.49M | 0.759 | 9.00 | 27.0 | 22.2 | 17.9 | 18.4 | 25.1 |
| graft-pruned | r20 (screen) | — | — | 14.27 | 58.7 | 42.6 | 28.9 | 28.5 | 38.3 |

`params`/`mAP` are carried from `models/README.md` (accuracy is device-invariant and is **not
apples-to-apples across families** — see that file's caveat).

## Same table as "vs baseline", which is where it gets interesting

| family | model | mAP | Jetson fp32 | vs base | CPU t6 | vs base | CPU all22 | vs base |
|---|---|---|---|---|---|---|---|---|
| anchor | yolo11s | 0.882 | 21.70 | +70 % | 70.0 | +138 % | 75.0 | +101 % |
| baseline | yolo11n | 0.877 | 12.74 | — | 29.4 | — | 37.3 | — |
| search | s39-40-38-38-14 | 0.871 | 15.27 | +20 % | 43.1 | +47 % | 44.6 | +20 % |
| search | s31-40-40-40-13 | 0.870 | 15.14 | +19 % | 42.4 | +44 % | 44.9 | +20 % |
| search | s40-38-39-36-13 | 0.868 | 14.98 | +18 % | 41.9 | +43 % | 45.8 | +23 % |
| dense | w30 | 0.856 | 15.26 | +20 % | 39.6 | +35 % | 43.5 | +17 % |
| dense | w25 (ctrl_n) | 0.854 | 11.33 | **−11 %** | 29.4 | **+0 %** | 38.4 | **+3 %** |
| graft | v2topdown | 0.846 | 18.15 | **+42 %** | 35.8 | **+22 %** | 43.0 | **+15 %** |
| dense | w22 | 0.845 | 11.54 | −9 % | 30.1 | +2 % | 37.4 | +0 % |
| graft | v3pan | 0.842 | 18.37 | **+44 %** | 36.9 | **+26 %** | 42.5 | **+14 %** |
| graft | winner-v1 noneck | 0.841 | 17.67 | **+39 %** | 35.4 | **+21 %** | 42.3 | **+13 %** |
| dense | w20 | 0.839 | 11.26 | −12 % | 27.7 | −6 % | 36.6 | −2 % |
| prune | r20 | 0.838 | 9.52 | −25 % | 21.0 | −28 % | 28.2 | −24 % |
| dense | w18 | 0.834 | 10.01 | −21 % | 24.0 | −18 % | 31.8 | −15 % |
| prune | r15 | 0.834 | 9.53 | −25 % | 21.4 | −27 % | 27.4 | −27 % |
| prune | r10 | 0.830 | 9.82 | −23 % | 22.1 | −25 % | 28.3 | −24 % |
| prune | r35 | 0.826 | 8.36 | −34 % | 17.7 | −40 % | 24.3 | −35 % |
| graft-pruned | r40 | 0.816 | 11.81 | **−7 %** | 22.9 | **−22 %** | 33.7 | **−10 %** |
| dense | w15 | 0.815 | 9.52 | −25 % | 22.6 | −23 % | 28.5 | −24 % |
| dense | w13 | 0.813 | 9.56 | −25 % | 22.3 | −24 % | 28.8 | −23 % |
| graft-pruned | halp_10p4 | 0.813 | 12.58 | −1 % | 26.8 | −9 % | 34.1 | −9 % |
| prune | r45 | 0.809 | 7.94 | −38 % | 17.2 | −42 % | 21.9 | −41 % |
| graft-pruned | halp_9p0 | 0.802 | 11.37 | −11 % | 24.5 | −17 % | 33.8 | −9 % |
| prune | r55 | 0.798 | 7.66 | −40 % | 16.0 | −45 % | 20.3 | −45 % |
| graft-pruned | r50_gtay | 0.795 | 10.23 | −20 % | 20.9 | −29 % | 27.0 | −28 % |
| prune | r30 | 0.790 | 8.28 | −35 % | 17.3 | −41 % | 22.6 | −40 % |
| graft-pruned | r60_gtay | 0.777 | 8.84 | −31 % | 18.8 | −36 % | 25.1 | −32 % |
| graft-pruned | r60 | 0.759 | 9.00 | −37 % | 18.4 | −37 % | 25.1 | −33 % |

## What the table says

**1. The graft penalty is roughly halved on x86 — it is partly an Orin property.**
`winner-v1 noneck` is **+39 % on the Orin but +21 % at t6 and +13 % on all22**; `v2topdown`
+42 % → +22 %; `v3pan` +44 % → +26 %. The depthwise OFA backbone is punished about twice as
hard by the Orin as by this CPU — consistent with large x86 caches hiding part of a DRAM-bound
deficit. The *direction* survives (grafts still lose), the *magnitude* does not.

**2. Dense's advantage evaporates on CPU.** On the Orin, every dense point beats the baseline
(`w25` −11 %, `w22` −9 %, `w20` −12 %). On CPU, `w25` is **+0 %**, `w22` **+2 %**, `w20` only
−6 %. The dense-scaling win is Orin-specific to a degree the README's story doesn't anticipate.

**3. Pruning wins on both devices, and by more on CPU.** `prune r20` is −25 % on Jetson and
**−28 % at t6 / −24 % on all22**; `r55` −40 % → −45 %. The README's headline — *"the pruned
family owns the Pareto frontier below 0.85"* — is the one claim that transfers cleanly.

**4. `graft-pruned r40` improves markedly on CPU**: −7 % on Jetson → **−22 % at t6**. The
pruned-graft rungs are more competitive on x86 than the CP 6.2-G verdict implies.

**5. The penalty is bandwidth-sensitive *within* the CPU.** Fitting `latency ~ params` over
`dense + prune + baseline` (15 models, 0.9–3.9 M; `yolo11s` excluded as a 9.7 M leverage point)
and taking each graft's residual:

| config | ρ vs Jetson | τ-b | graft residual (% of predicted) |
|---|---|---|---|
| `t1` | 0.888 | 0.714 | **5.9 %** |
| `t2` | 0.912 | 0.749 | 10.0 % |
| `t4` | 0.934 | 0.788 | 13.2 % |
| `t6` | 0.939 | 0.803 | **14.4 %** |
| `all22` | 0.940 | 0.793 | 15.3 % *(confounded)* |

As cores are added and DRAM becomes the constraint, grafts drift further above the line, and the
CPU ordering agrees *more* with the Orin's (ρ 0.888 → 0.939). Both point the same way: the graft
deficit is a memory-system effect. It is simply *smaller* on this memory system.

Per family, the residual's **slope** separates two different failure modes:

| family | `t1` residual | `t6` residual | Δ |
|---|---|---|---|
| `dense_nas` (search) | 48.9 % | 37.5 % | **−11.5** |
| `graft_pruned` | 7.0 % | 16.6 % | **+9.6** |
| `graft` | 3.2 % | 9.5 % | **+6.2** |
| `dense` | 5.4 % | 7.0 % | +1.6 |
| `prune` | −4.8 % | −6.0 % | −1.2 |

Grafts start near the line and **climb** under bandwidth pressure (memory-bound). The `search`
family is the mirror image — a large penalty at 1 thread that **shrinks** with parallelism
(compute-bound). Both are "slow"; they are slow for opposite reasons.

**6. `all22` is slower than `t6` for most models** (baseline 37.3 vs 29.4 ms). Adding E-cores
costs more in scheduling and pressure than their throughput returns. It is reported as the
"whole machine" number and excluded from the scaling analysis, since it varies core *type* and
count together.

## Reproducibility

`t1` and `t6` were re-measured end-to-end in a second run: ρ 0.884/0.936 (vs 0.888/0.939),
graft residual 6.1 %/13.6 % (vs 5.9 %/14.4 %), yolo11n 77.4/30.4 ms (vs 77.1/29.4). Run-to-run
spread ~0.8 pt against an ~8 pt effect.

## Methodology finding: ORT spin-wait corrupts interleaved benching

The first complete run was **wrong in the shape of the hypothesis** — it reported the graft
penalty rising 1.15 → 6.12 ms with thread count, which looked like clean confirmation.

ORT's intra-op workers **spin-wait** after `run()` returns instead of sleeping, to shave wake-up
latency for a server calling *one* session in a loop. This bench rotates 29 live sessions, so
every model's run collided with the previous pools still burning the same pinned cores. At `t6`,
29 sessions held, interleaved:

| | t6 baseline p50 |
|---|---|
| spinning ON | **564.5 ms** |
| spinning OFF | **30.4 ms** |
| isolated (no interleaving) | 20.3 ms |

An **18.6× artifact that grows with thread count** — indistinguishable in shape from the effect
under test. The tell was the raw scaling column: yolo11n appeared to get *slower* with more
cores (77 → 117 ms), which is physically impossible.

Two traps worth recording: **spinning and interleaving interact, so neither alone reveals the
fault** — a first test came back negative because it timed one session back-to-back, where the
spin-wait always helps the session that owns it. And disabling spinning removed most of the
thermal drift, because the spin storm was what was cooking the CPU.

Fix: `session.intra_op.allow_spinning = 0`, pinned by `ALLOW_SPINNING` in
`lut/orchestrate/cpu_ort.py`, guarded by `tests/test_cpu_ort.py::test_spinning_stays_disabled`.

## Threats to validity

1. **ORT ≠ TensorRT.** Graph optimisation and kernel selection differ, so part of the
   Jetson-vs-CPU gap is *runtime*, not *device*. This is the hardest limit here: the "graft
   penalty is halved on x86" finding could in principle be TensorRT handling depthwise convs
   worse than ORT does, rather than the Orin's memory system. Separating them needs
   ORT-on-Jetson or TRT-on-x86. **Do not present the halving as a pure device effect.**
2. **Thermal drift.** `t2/t4/t6/all22` are stamped `thermal_drift_detected` (canary p50 moved
   >5 % from round 0); only `t1` is clean. Clocks cannot be locked as on the Jetson — this
   laptop reaches ~99 °C under sustained load regardless of start temperature. Comparisons
   *within* a config survive because interleaving makes drift common-mode (every model appears
   once per round, so each pools over the same thermal states). The **absolute ms** are not
   clean numbers.
3. **`all22` is confounded by construction** (E-cores are slower *and* add pressure).
4. **The residual metric assumes latency is ~linear in params**, and params ignores activation
   volume — precisely what a memory-bound argument is about. Screening statistic; the per-config
   scatter is in `data/cpu/rank_report.json`.
5. **Single machine, fp32 only.** One x86 part, one memory system.

## Artifacts

- `data/cpu/*.json` — 145 rows (29 models × 5 configs), `source: "x86_ort"`, gitignored
- `data/cpu/rank_report.json` — ρ/τ, reference fits, per-model residuals, per-config scatter
- Regenerate: `python -m lut.orchestrate.cpu_bench --configs <cfg>` then
  `python -m lut.orchestrate.cpu_rank_report` (`.venv`)
- Spec: `docs/superpowers/specs/2026-07-15-cpu-cross-device-rank-check-design.md`
