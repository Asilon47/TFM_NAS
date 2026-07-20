# CP 10.3 wave-1 finals — the honest MCU Pareto read (2026-07-20)

Two 192-topdown screen leaders trained 100-ep (bare-AdamW, no KD) and re-priced on GAP8.

| point | params | final mAP50-95 | GAP8 FPS | vs baseline |
|---|---|---|---|---|
| **yolo11n-pose @160 (baseline)** | 2.65M | **0.6227** | **2.92** | — |
| 863c75818953 (192-td) | 1.23M | 0.6026 | 2.91 | matched FPS, **−2.0 pts**, 2.2× fewer params |
| d0d520f8a66e (192-td) | 0.77M | 0.5921 | 4.09 | **+40 % FPS**, −3.1 pts, 3.4× fewer params |

## Verdict: a genuine trade, not a domination — but the gap collapsed

- **At matched FPS the baseline still wins by 2.0 pts** (0.6227 vs 0.6026 at ~2.9 FPS).
  Same *shape* as the Orin result (faster-but-less-accurate), NOT the hoped-for match.
- **But the gap collapsed from −9 to −2**: the res-screen @160 measured the graft at
  0.5347 (−0.088); resolution 160→192 + the topdown neck + a fresh arch closed 6.8 of
  those points. The remaining 2 are within reach of the two untried levers below.
- **d0d520 is the only true Pareto point**: at 4.09 FPS it is faster than anything the
  baseline offers, for −3 pts — the "want more FPS than 2.9" operating point (3.4× fewer
  params). 863c is dominated by the baseline (same speed, less accurate).

## Two reasons the −2 is likely closable (both untried here)

1. **Recipe confound.** These finals are bare-AdamW; the baseline is its stock recipe
   (multi_scale, cos-LR, long schedule). Beat-n measured **recipe-lite = +5.0 pts** on a
   dense recovery three days ago — a recipe-lite 192-td twin is the single highest-EV next
   experiment and could flip the matched-FPS verdict outright.
2. **This is the screen, not the search.** MOTPE has not run. The proxy ranked the
   192-topdown region correctly (863c proxy 0.352 > d0d520 0.293 → finals held) even though
   it fails inside the anchors' 1.9-pt band (ρ=0.2) — so region-guided search + finals-for-
   ties is viable.

## Proxy note (corrects the wave-1 entry)

The two 192-td finals confirm the proxy's **region and coarse ranking** (0.352 > 0.293 →
0.6026 > 0.5921 held). The fidelity FAILURE is specifically **within tight bands** (the 4
pruned anchors at 160, ρ=0.2). So: MOTPE trusts the proxy to pick res × neck × capacity
regions, and spends finals only to break near-ties. Cross-res comparability now has one
data point — 192 proxies did predict the best finals, consistent with the table.

## UPDATE 2026-07-20 — recipe-lite does NOT transfer to the graft (2nd confirmation)

The recipe-lite twin (same arch/res/seed, 100 ep, cos-LR+warmup+EMA+close-mosaic) came
back **below** its bare-AdamW sibling on both points:

| point | bare-AdamW | recipe-lite | Δ |
|---|---|---|---|
| 863c (192-td 1.23M) | 0.6026 | 0.5950 | **−0.0076** |
| d0d520 (192-td 769K) | 0.5921 | 0.5861 | **−0.0060** |

**Recipe-lite hurt the graft** — the opposite of beat-n (where it bought +5 pts on a dense
recovery), and the **second** negative after Gate 1 (the A2 recipe-parity twin @160:
0.4846 vs bare 0.5347, −5 pts). The lever that won the Orin does not carry to the MCU
graft family; the regime differs (from-init-ish OFA backbone + pose head vs a converged
dense donor). **The −2 pt matched-FPS gap is NOT a recipe artifact — it is structural.**

Consequence: the two class-levers that could close an accuracy gap are both spent on the
graft — recipe (negative, ×2) and KD (null-to-negative, on record). What remains is a
**better base architecture** — exactly what the wave-2 proxy batch (8 matched-FPS-and-
faster 192-topdown-region candidates) is testing. If none beats 863c's 0.6026, the honest
MCU verdict is the trade: −2 pts at matched FPS, structural.
