# Pruned-graft recovery (CP 6.2-G) — does an 84 %-pruned graft keep usable accuracy?

**Question (the open half of `models/screen_prune_graft/`).** The latency screen showed only
r=0.60 (84 % param sparsity) drags the winner graft under the fp16 baseline (6.58 vs 7.75 ms).
Prior: accuracy craters. This trains the screened architectures to capacity and measures it.

**Method** (`prune/recover_graft.py`, Kaggle acct1 kernel v23, 2026-07-11): per ratio, build the
winner graft fresh (OFA-ImageNet backbone + warm gate head, all trainable) → prune with the SAME
CP 6.1 DepGraph harness the screen used (identical channel outcome — r60 params match the screen
exactly: 493,971) → BN re-estimate → 100-epoch bare-AdamW train (prune-then-TRAIN, the
`dense_scaling`-comparable protocol) → val + deploy-ONNX export. Single seed (0). Latencies are
the screen's measured numbers (TRT latency is weight-value-independent). Unpruned anchor =
winner-v1 full-FT 0.841 (not re-trained).

## Result

| rung | params | sparsity | mAP50-95 | mAP50 | fp32 ms | fp16 ms |
|---|---|---|---|---|---|---|
| unpruned (anchor) | 3.00M | — | 0.841 | — | 17.69 | 12.38 |
| **r40** | 1.07M | 64.3 % | **0.8163** | 0.9218 | 11.81 | 8.41 |
| **r60** | 0.49M | 83.6 % | **0.7589** | 0.8958 | 9.01 | 6.58 |

## Verdict — retention is real; dominance is not

- **The crater-prior was wrong**: −2.5 pts at 64 % params gone, −8.2 pts at 84 %, mAP50 ≥ 0.896
  everywhere. No accuracy cliff — the narrow-MBConv regime is *viable*.
- **Accuracy-per-param is mid-pack**: r40 (0.816 @ 1.07M) edges dense w13 (0.813 @ 1.0M) despite
  training bare-AdamW vs the dense wave's stock Ultralytics recipe. The graft's deficit was never
  capacity — it is **ms-per-param** (memory-bound depthwise).
- **Against the measured frontier (`models/README.md`) both rungs are strictly dominated**:
  r60 (0.759 @ 9.01/6.58) loses to prune r55 (0.798 @ 7.66/5.07), r35 (0.826 @ 8.36/5.38) and
  r20 (0.838 @ 9.52/5.91) on both axes; r40 loses to r35/w18. Gap to the front at matched
  latency ≈ +7–8 pts. **The screen's prior is confirmed: pruned to latency-parity, the graft is
  accuracy-dominated by the dense arm.**
- **Caveat — these rungs are a lower bound**: the pruner ran its floor configuration (uniform
  per-layer ratios, magnitude importance, one-shot at extreme sparsity, bare-AdamW recovery,
  no KD). The pruning-as-search program (procedure.md "CP 6.2-G CLOSED") upgrades the technique
  and applies it symmetrically to both families.

Binaries here are copies of `data/cp33_kaggle_out/recover_graft_r{40,60}/` (gitignored;
regenerable). Full run reports: `recover_graft_result.json` + the per-rung `*.meta.json` there.
