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
bash kaggle/push.sh --resume   # between sessions: --pull + --cache (version the resume store)
bash kaggle/push.sh --cache    # (called by --resume) version the cp33-bo-cache Dataset
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

Kaggle **cannot attach a notebook's own output as its own input** (a dependency cycle), so
the caches round-trip through a tiny dedicated `tfm-nas-cp33-bo-cache` **Dataset** instead.
`push.sh` seeds it automatically on the first kernel push; the kernel attaches it via
`kernel-metadata.json` and `run.py` restores the shards from it (no notebook self-reference).

1. **Session 1** — `bash kaggle/push.sh` (seeds the empty cache Dataset, pushes the kernel),
   then in the UI set Accelerator to **GPU T4 x2** and Save & Run All. It runs ~10.5 h, then
   finishes `PARTIAL`, leaving its caches in the kernel output.
2. **Between sessions (laptop)** — `bash kaggle/push.sh --resume`. This pulls the finished
   session's output and versions its cache shards into the `tfm-nas-cp33-bo-cache` Dataset.
3. **Session 2+** — in the Kaggle editor, refresh the **`tfm-nas-cp33-bo-cache`** input to its
   newest version (the input panel shows an update), keep **GPU T4 x2**, and Save & Run All.
   `run.py` prints `[resume] restored N shard(s)` (**N > 0** confirms the round-trip) and the
   workers continue from where they stopped.
4. Repeat 2–3 until the log prints `complete=True` (JSON `"complete": true`). Pull the final
   verdict with `bash kaggle/push.sh --pull`.

> Refreshing the dataset version in the editor preserves your GPU T4 x2 setting; an API
> re-push (`bash kaggle/push.sh`) always attaches the latest cache version but **resets the
> accelerator**, so you'd have to re-select GPU T4 x2 in the UI before running.

> The cache is namespaced by resolution (`cp33_bo_cache_r<RES>`), so switching to `RES=640`
> after the sweep starts a **fresh** cache rather than reusing @224 latencies. For a
> shorter calendar, lower `BUDGET` so a single dual-T4 session self-completes (re-opens D2).
> `CALIBRATE` still reports per-eval wall-clock at the top of each run.

### Reusing prior fine-tunes across resolutions (accuracy memo)

Accuracy is measured at a fixed `imgsz=640`, so it is **independent of the LUT latency
resolution** — the fine-tunes from a previous (`RES=224`) run are valid at `RES=640`.
Distil them into a shared memo so the @640 search doesn't re-pay for archs it already
measured:

```bash
bash kaggle/push.sh --resume                       # pull the prior run's cache shards
python -m search.build_acc_memo \
    --cache "data/cp33_kaggle_out/cp33_bo_cache_r*.jsonl" \
    --out data/cp33_acc_memo.json                  # {arch, acc} memo
bash kaggle/push.sh --data                          # ships the memo (+ the @640 LUT) to Kaggle
```

`run.py` finds `cp33_acc_memo.json` and passes `--acc-memo`; the memo is consulted
*before* the GPU oracle by **both** BO and random search, so a hit only saves compute and
**does not** bias the BO-vs-random DoD (each method still earns hypervolume only for the
archs it independently proposes). On a hit the log shows `[acc-memo] N prior measurement(s)`.

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
