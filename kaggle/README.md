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

`run.py`'s CONFIG block is the **full 5-seed DoD** (`SEEDS=5, BUDGET=50, N_INIT=20`,
`RES=224` until the Jetson @640 sweep lands). At ~700 s/eval it is **too big for one
session** (~59 h wall-clock even on dual-T4), so it is **resumable across sessions**:

- Each commit stops starting new evals after `DEADLINE_H` hours (default 10.5 — a clean
  boundary under Kaggle's 12 h kill), having appended per-seed caches to `/kaggle/working`.
- The merged `cp33_bo.json` carries a `complete` flag — `true` only once **every** seed has
  spent its full budget. That, with `passes`, is the real DoD verdict.

### Resuming — the ~5-session protocol

1. **Session 1** — set Accelerator to **GPU T4 x2**, Save & Run All. It runs ~10.5 h then
   finishes `PARTIAL`, saving its caches as the kernel output.
2. **Session 2+** — **+ Add Input → Notebook → this notebook** (its own latest output),
   then Save & Run All again. `run.py` restores the cache shards from that input and the
   workers continue from where they stopped.
3. Repeat until the log prints `complete=True` (JSON `"complete": true`). Pull it with
   `bash kaggle/push.sh --pull`.

> The cache is namespaced by resolution (`cp33_bo_cache_r<RES>`), so switching to `RES=640`
> after the sweep starts a **fresh** cache rather than reusing @224 latencies. For a
> shorter calendar, lower `BUDGET` so a single dual-T4 session self-completes (re-opens D2).
> `CALIBRATE` still reports per-eval wall-clock at the top of each run.

**Dual-GPU:** on a **GPU T4 x2** session, `run.py` automatically fans the seeds across
both GPUs (one `search.bo --seed-list …` worker per device, `CUDA_VISIBLE_DEVICES`-pinned),
runs them in parallel, then `--merge`s the per-worker outputs into one verdict over all
seeds — ~halving wall-clock **and** quota (Kaggle bills GPU *session* time). Each session it
**re-balances**: it reads how many evals each seed still owes from the restored caches and
LPT-distributes the *unfinished* work, so no GPU sits idle on already-done seeds late in the
campaign. 1-GPU or 1-seed sessions run sequentially. So for the DoD, pick **GPU T4 x2**.

## Outputs

`/kaggle/working/cp33_bo.json` — per-seed Pareto hypervolume + the BO-vs-random verdict
(the CP 3.3 DoD). Pull it with `bash kaggle/push.sh --pull`.

> Precision/resolution note: until the Jetson @640 LUT sweep lands (see
> `lut/docs/jetson_640_runbook.md`), keep `RES=224` — the fine-tune still trains at
> `imgsz=640`, only the latency/feasibility lookup uses the measured @224 LUT.
