# `eval/`

Accuracy-side evaluation harnesses for the NAS pipeline. Runs under **`.venv-nas`**
(needs `ofa` + `torchvision`, plus `tqdm` + `gdown` — importing OFA's `ofa.tutorial.*`
eval helpers transitively requires both — and `scipy` for the rank-fidelity stats; all
pinned in `requirements-nas.txt`). The pure decision logic is importable under `.venv` too.

## `imagenet_sanity.py` — CP 1.4 (ImageNet sanity check)

Validates that the inherited OFA-MBv3-w1.0 supernet, sampled and **BatchNorm-recalibrated**,
reproduces the accuracy OFA's own released predictor assigns each arch. This is the first
exercise of the accuracy axis — a wrong weight load or a skipped BN recalibration would
silently poison every accuracy number from CP 2.4 onward.

**DoD:** measured and predicted top-1 **rank-correlate** across a spread of archs — Spearman
ρ ≥ 0.85 (p < 0.05). OFA's predictor is a *ranking* model with a near-constant absolute offset
(~6.3 pp high; see below), so a per-arch absolute bar is the wrong test — rank fidelity is its
intended use. The OLS affine fit (`measured ≈ slope·predicted + intercept`, expected
slope ≈ 1, r² ≈ 0.98+) is reported as the absolute-scale evidence, and the `max` arch ≈ 77 %
(OFA's biggest-net ballpark) is an external anchor.

### Required data — ImageNet in `ImageFolder` layout

```
<imagenet-path>/
  train/<wnid>/*.JPEG     # for BN recalibration (set_running_statistics)
  val/<wnid>/*.JPEG       # for accuracy measurement
```

- **`val/`** must use standard ImageNet **wnid** folder names: torchvision assigns class
  indices by sorted folder name, and that order must match the supernet's 1000-way output.
  A 2 k-image subset satisfies the DoD wording, but see the noise note below.
- **`train/`** is only forward-passed for BN recalibration (labels ignored). A **few-thousand
  image subset is plenty** and keeps the `ImageFolder` index scan fast — pointing at the full
  1.28 M-image train set works but scans slowly on every run.

### Run

```bash
source .venv-nas/bin/activate

# Rank-fidelity run on a GPU box (e.g. Kaggle's free GPU). Full 50k val keeps the per-arch
# CI tight (±0.4pp) so measurement noise doesn't flip near-tied ranks:
python -m eval.imagenet_sanity --imagenet-path /data/imagenet --device cuda --n-val 0 --n-random 8
```

Evaluates the corners (`max`, `min`) plus `--n-random` sampled interior archs, then prints a
per-arch table (measured / predicted / raw gap / **calib gap** / 95% CI), the ranking +
affine-fit summary, and the DoD verdict. Defaults: `--archs max,min`, `--n-random 8`
(10 archs total), `--rank-threshold 0.85`, `--resolution 224`, `--bn-images 2000`,
`--n-val 0`, `--bar 1.5` (the per-arch *calibrated*-gap info band, **not** the gate). Bump
`--n-random` to ~14–18 for a tighter ρ interval. Exit code is `0` on PASS, `1` on FAIL.
(CPU works but is slow — ~10 archs × BN-recalib × 50k val; a GPU is strongly preferred.)

### Why the accuracy *predictor* — and why *ranking*, not absolute

The comparison uses OFA's released accuracy predictor
(`ofa/tutorial/accuracy_predictor.py`), the artifact OFA itself uses to *rank* candidate
subnets during evolutionary search. We do **not** compare against OFA's specialized nets
(`note10_lat@…_top1@…_finetune@75`): those carry 25–75 extra fine-tune epochs, so a direct
subnet would legitimately score several points lower (a fine-tuning artifact, not a bug).

**The first real run (2026-06-17) showed the predictor reads ~6.3 pp high in absolute top-1,
with a near-constant offset:** a single offset anchored on the `max` arch reconciled every
arch to <1 pp, and the rank order was identical. That is the signature of a ranking model
trained on a higher absolute scale (likely a train-holdout subset) — harmless for OFA's use
and ours, since search only needs the *order*. Hence the rank-fidelity DoD; the affine fit
quantifies and removes the offset for an absolute read.

### Three things to watch on the first run

- **Predictor scale.** The predictor emits a *fraction* (~0.84 for the max arch); the harness
  normalizes anything ≤ 1.0 by ×100, so the `predicted` column lands in the ~80s (a percent),
  not ~0.8.
- **The raw offset is expected — read the calib gap + ρ.** Every `raw gap` will be ≈ −6 pp
  (the predictor reads high). What matters is the **calib gap**
  (`measured − affine(predicted)`), which should sit within ~1 pp, and the **Spearman ρ**
  gate. A large raw gap with a small calib gap and high ρ is a PASS, not a weight bug.
- **Noise vs ranking.** Rank fidelity is noise-sensitive when archs cluster in accuracy — a
  CI overlap can flip two near-tied ranks. Use full 50 k val (`--n-val 0`, ±0.4 pp) and
  enough archs (`--n-random ≥ 8`; the corners widen the range) so true ordering dominates.
