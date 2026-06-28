# `kaggle/` — push the CP 3.3 search to Kaggle GPU via the API

Run the warm-head Bayesian-Optimization search (`search.bo`, the accuracy half of
CP 3.3) on Kaggle GPU without uploading anything by hand. A **script kernel**
(`run.py`) clones the public repo, installs the stack, attaches a **data Dataset**
(gate dataset + LUT + NSGA-II seeds + the frozen gate-head donor), and runs the search.

## One-time setup

1. **Token + username** — Kaggle → *Settings* → *API* → *Create New Token* yields a
   new-style token string (`KGAT_…`). Save just that string at **`secrets/access_token`**,
   and put your Kaggle username at **`secrets/kaggle_username`** (the new token doesn't
   embed it). `secrets/` is gitignored; `push.sh` refuses to run if any credential is
   git-tracked. *(A legacy `secrets/kaggle.json` with `{"username","key"}` still works.)*
2. **CLI** — `pip install kaggle` (also in `requirements.txt`).
3. **Repo** — the GitHub repo is public, so the kernel clones it with no token. For a
   private repo, add a `GITHUB_TOKEN` Kaggle *Secret* to the kernel.

## Usage

```bash
bash kaggle/push.sh --data     # (occasional) create/version the 1.6 GB data Dataset
bash kaggle/push.sh            # push + run the kernel (does the search)
bash kaggle/push.sh --status   # kernel run status
bash kaggle/push.sh --pull     # download /kaggle/working/* into data/cp33_kaggle_out/
```

`--data` stages `dataset/`, `data/lut.jsonl`, `data/phase3_nsga2_frontier.json`, and the
donor `runs/.../best.pt` via **hardlinks** (no 1.6 GB copy) and uploads them as
`<user>/tfm-nas-gate-pose`. Re-run it only when the data or LUT changes (e.g. after the
@640 sweep). The OFA checkpoint is **not** uploaded — the kernel re-downloads it
(SHA-pinned) over the internet.

## What the kernel runs

`run.py`'s CONFIG block defaults to a cheap **proving run** (a timed calibration + a
1-seed, budget-8 search) that verifies the whole pipeline inside one GPU session. For
the real DoD, edit the CONFIG block and re-push:

| knob | proving | DoD |
|---|---|---|
| `SEEDS` | 1 | 5 |
| `BUDGET` | 8 | 50 |
| `N_INIT` | 4 | 20 |
| `RES` / `T_MAX_MS` | 224 / 16.7 | 640 / `min(baseline, 16.7)` after the Jetson @640 sweep |

`CALIBRATE` first reports per-eval wall-clock so you can size the 5-seed budget against
Kaggle's weekly GPU quota before committing. The run is **resumable** (`--cache`): a
re-pushed kernel skips evals already in the output cache.

**Dual-GPU:** on a **GPU T4 x2** session, `run.py` automatically fans the seeds across
both GPUs (one `search.bo --seed-start … --seeds …` worker per device, `CUDA_VISIBLE_DEVICES`-pinned),
runs them in parallel, then `--merge`s the per-worker outputs into one verdict over all
seeds — ~halving wall-clock **and** quota (Kaggle bills GPU *session* time). 1-GPU or
1-seed sessions run sequentially. So for the DoD, pick **GPU T4 x2** in the UI.

## Outputs

`/kaggle/working/cp33_bo.json` — per-seed Pareto hypervolume + the BO-vs-random verdict
(the CP 3.3 DoD). Pull it with `bash kaggle/push.sh --pull`.

> Precision/resolution note: until the Jetson @640 LUT sweep lands (see
> `lut/docs/jetson_640_runbook.md`), keep `RES=224` — the fine-tune still trains at
> `imgsz=640`, only the latency/feasibility lookup uses the measured @224 LUT.
