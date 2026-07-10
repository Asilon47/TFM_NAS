# Prune-the-graft latency screen — can pruning drag a searched OFA subnet under the baseline?

**Question (2026-07-10, user).** The OFA-MBv3 graft loses end-to-end on the Nano (17.69 ms fp32 /
12.38 ms fp16 vs yolo11n-pose 12.75 / 7.75) because it is memory-bound. Structured pruning removes
channels → cuts activation memory traffic → could recover latency. Does it drag a *searched* OFA
subnet under the baseline, at a survivable sparsity?

**Method.** `python -m prune.screen_prune_graft` runs the CP 6.1 DepGraph harness
(`prune/prune_graft.py`; group-L2 importance, `round_to=16`, Pose-head outputs + DFL protected) on
the winner graft (`d=[2,2,4,3,3]`) at channel ratios 0.2 / 0.4 / 0.6, exports each end-to-end graft
to ONNX @640, and `lut.orchestrate.bench_model` measures them on the Nano — **mode 0 / 612 MHz,
clocks locked, one at a time**, TRT 10.3, fp32 + fp16. **No recovery training**: TRT latency is
weight-value-independent, so the channel COUNT sets latency; accuracy is a separate follow-up owed
only if a rung clears the bar. ONNX (`*.onnx`) gitignored; measured JSONs in
`data/e2e/graft_prune_r*` (gitignored). Full record: `screen_prune_graft_result.json`.

## Result (measured, one session)

| channel prune | param sparsity | fp32 e2e | vs base 12.75 | fp16 e2e | vs base 7.75 | fp16 FPS |
|---|---|---|---|---|---|---|
| unpruned graft | 0% | 17.69 | +38.7% | 12.38 | +59.7% | 81 |
| **r=0.20** | 39% | 14.27 | +11.9% | 10.15 | +31.0% | 98 |
| **r=0.40** | 64% | 11.81 | −7.4% | 8.41 | +8.5% | 119 |
| **r=0.60** | 84% | 9.01 | −29.4% | **6.58** | **−15.2%** | 152 |

baseline yolo11n-pose: 12.75 fp32 / 7.75 fp16 (129 FPS). Locked-clock std < 0.07 ms.

## Verdict — pruning DOES reduce latency, but only an 84%-gutted graft beats the fp16 baseline

- **fp32:** r ≥ 0.40 (64 % params gone) beats the baseline.
- **fp16 (the deployment precision):** only **r = 0.60 (84 % params gone)** beats it; the crossover
  is ~r = 0.48 (~73 % params). The pruned e2e beat the channel-linear prior because the adapter +
  head internals prune too and the compute-bound 1×1s drop super-linearly (FLOPs ~ width²).
- **Every rung already meets 60 FPS** — even the unpruned graft (81 FPS). The deployment bar was
  never the constraint; *beating the baseline* is, and that needs catastrophic pruning.

**So pruning reduces latency as asked, and can technically place a NAS-searched net under the
baseline — but only at 84 % param sparsity, from an unpruned proxy of just 0.61 mAP.** The dense
family already beats the baseline latency (`prune_base_*` fp16 < 7.75 ms) while holding 0.83+ mAP,
so the graft — even pruned to win on latency — is **accuracy-dominated by the dense arm**. Whether
an 84 %-pruned graft retains any usable accuracy is the one open question (CP 6.2 recovery train);
the prior is that it loses. Latency is **measured-only** (off the LUT grid).
