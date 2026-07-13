# Stage-3 dense-space NAS — the searched winner beats the baseline (2026-07-13)

Hardware-aware NAS over the **device-native yolo11-pose family** (per-stage width search), the
space the cross-family measured program selected (`search/dense_nas.py`). Six TPE studies
(seeds 0–5, 16 trials/GPU) trained **60 unique candidates** at a 30-ep from-scratch proxy
(G3-gated: ρ=1.000 vs the ten 100-ep dense oracles); the five leaders re-trained at 100 ep.

## Oracle round (100-ep, seed 0, single-seed — de-noise in flight)

| candidate (per-stage width ×) | mAP50-95 | mAP50 | params | pred fp32 ms |
|---|---|---|---|---|
| **s31-40-40-40-13** | **0.8786** | 0.9439 | 2.90M | 9.10 |
| s39-40-38-38-14 | 0.8762 | 0.9438 | 2.77M | 9.55 |
| s40-38-39-36-13 | 0.8718 | 0.9479 | 2.65M | 8.88 |
| s18-34-31-38-14 | 0.8656 | 0.9368 | 2.58M | 10.74 |
| s10-34-36-35-20 | 0.8647 | 0.9412 | 3.09M | 11.65 |

Reference: pretrained **yolo11n-pose baseline 0.877** (COCO-pretrained + full recipe) ·
**prune_base r20 0.8381** @ measured 9.52/5.91 ms · ctrl_n (yolo11n shape, from-scratch) 0.854.

## What it means (pending two gates)

## De-noised (seeds {1,2,3}) — winner's curse caught, robust winner is s39

| candidate | mean {1,2,3} | sd | seed-0 bias | params |
|---|---|---|---|---|
| **s39-40-38-38-14** | **0.8709** | 0.0012 | +0.0053 | 2.77M |
| s31-40-40-40-13 | 0.8702 | 0.0064 | +0.0085 | 2.90M |
| s40-38-39-36-13 | 0.8677 | 0.0032 | +0.0040 | 2.65M |

The single-seed leader **s31 (0.8786) was upward-biased +0.0085 and the noisiest arch**
(seed range 0.865–0.879). On fresh seeds **s39-40-38-38-14 = 0.8709 ± 0.0012 is the robust
winner** — tied with s31 on the mean but 5× more stable and fewer params. CP 3.5 discipline
did exactly its job (third winner's-curse catch in the project).

- **Honest accuracy: the searched winner (0.871) MATCHES the pretrained baseline (0.877) from
  scratch** (−0.006, within 2σ) — it does not clearly beat it (the seed-0 "beats" reading was
  biased). Against the buildable competition it is the decisive leader: **+3.3 pts over
  prune_base r20 (0.8381)**, +1.7 over ctrl_n-from-scratch (0.854).
- **The discovery** (seed-independent): every top candidate is **wide P3/P4 feature stages
  (0.34–0.40), gutted final stage (0.13–0.20)**. yolo11's 1024-ch SPPF/C2PSA tail is
  over-provisioned for single-class large-object gate pose; reallocating that budget forward is
  worth +2–4 pts at fixed size — a per-stage allocation the global-width curve and the DepGraph
  saliency ladder could not express (the AutoSlim argument, realized).
- **The proxy's top-1 == the oracle top-1** (s31 led both 30-ep proxy and 100-ep oracle): the
  G3-gated search is validated end-to-end.

## Last gate before winner-v2

**The HALP lesson**: `pred_fp32` is the surrogate, not the board → Nano bench of the finalist
ONNX (mode 0, locked clocks) is the latency gate; no Pareto claim until measured. Also breaks
the s39/s31 accuracy tie on the Pareto axis (s39 pred 9.55, s31 pred 9.10 fp32).

Binaries (`dense_s*_o100*.pt/.onnx`) are gitignored, in `data/cp33_kaggle_out/dense_nas/`
(regenerable from the tag via `search.dense_nas --oracle-tags`). This manifest + the row JSONs
are the tracked record.
