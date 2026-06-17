# `eval/`

Accuracy-side evaluation harnesses for the NAS pipeline. Runs under **`.venv-nas`**
(needs `ofa` + `torchvision`, plus `tqdm` + `gdown` — importing OFA's `ofa.tutorial.*`
eval helpers transitively requires both; they're pinned in `requirements-nas.txt`). The
pure decision logic is importable under `.venv` too.

## `imagenet_sanity.py` — CP 1.4 (ImageNet sanity check)

Validates that the inherited OFA-MBv3-w1.0 supernet, sampled and **BatchNorm-recalibrated**,
reproduces the top-1 accuracy OFA's own released model predicts for an arch. This is the
first exercise of the accuracy axis — a wrong weight load or a skipped BN recalibration
would silently poison every accuracy number from CP 2.4 onward.

**DoD:** for a sampled subnet, `|measured − predicted top-1| ≤ 1.5 pp`.

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

# CPU-local (this machine, no CUDA) — fine for a one-shot sanity check:
python -m eval.imagenet_sanity --imagenet-path /data/imagenet --n-val 10000

# On the remote GPU box (faster; enables full 50k val):
python -m eval.imagenet_sanity --imagenet-path /data/imagenet --device cuda --n-val 0
```

Prints a per-arch table (measured / predicted / gap / 95% CI / verdict) and the overall DoD
verdict. Defaults: `--archs max,min,random`, `--resolution 224`, `--bn-images 2000`,
`--bar 1.5`. Exit code is `0` on PASS, `1` on FAIL.

### Why the accuracy *predictor* is the reference

The comparison uses OFA's released accuracy predictor
(`ofa/tutorial/accuracy_predictor.py`), trained on direct-extraction + BN-recalibrated
subnets of *this exact* w1.0 space — so predicted-vs-measured is apples-to-apples. We do
**not** compare against OFA's specialized nets (`note10_lat@…_top1@…_finetune@75`): those
carry 25–75 extra fine-tune epochs, so a direct subnet would legitimately score several
points lower. That gap would be a fine-tuning artifact, not a weight-loading bug.

### Three things to watch on the first run

- **Predictor scale.** The predictor's released weights emit a *fraction* (verified: ~0.84
  for the max arch), and the harness normalizes anything ≤ 1.0 by ×100. The printed `max`
  *predicted* value should land in the ~80s (a percent), not ~0.8 — confirms the scaling
  fired.
- **Corners are diagnostic, not gated.** `max`/`min` sit at the edge of the predictor's
  training distribution, where it extrapolates *optimistically* (it predicts ~83.6 % for
  the max arch — above OFA's realistic ceiling even with fine-tuning). The DoD therefore
  gates on the interior `random` (sampled) arch; corners print a `(diag)` tag and inform
  ranking/range only. A corner reading `FAIL (diag)` is expected, not a weight bug.
- **Measurement noise vs the bar.** A 2 k-image val set carries ≈ ±1.8 pp (95% CI) of
  binomial noise — *wider* than the 1.5 pp bar — so the report prints the CI per arch and a
  `noise` verdict for gaps that bust the bar but sit inside the CI. Prefer `--n-val 10000`
  (±0.8 pp) or the full 50 k (±0.4 pp) for a clean read.
