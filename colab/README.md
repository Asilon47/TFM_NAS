# `colab/` — resume the CP 3.4 TPE search (and anchor-B) on a free Colab T4

A **third search backend** alongside `kaggle/` (Kaggle GPU) and `jetson/` (AGX compute
node). The compute is identical — `search.tpe` warm-head architecture search + an optional
`ultralytics` anchor-B fine-tune — only the two I/O planes move so it runs on Colab:

| plane | Kaggle | **Colab** |
|---|---|---|
| big data (1.6 GB gate set + LUT + donor + memo) | attached `/kaggle/input` Dataset | **pulled** from the same `tfm-nas-gate-pose` Kaggle Dataset via the CLI |
| durable state (resume caches, outputs, anchor run dir) | round-tripped through a Kaggle Dataset | a mounted **Google Drive** folder |
| code | `git clone` in the kernel | `git clone` in the notebook |

Use it when the **Kaggle GPU quota is exhausted** — Colab is a separate quota pool. (The
TPU is *not* an option: ultralytics + OFA are CUDA/CPU PyTorch with no XLA path.)

## Why this is cheap right now

The @640 TPE DoD is **~80 % done** — it stalled when Kaggle quota ran out mid-campaign:

| seed | TPE proposals | random control |
|---|---|---|
| 0, 1 | 50/50 ✅ | 50/50 (cached, reused free) |
| 2 | 35/50 | 50/50 |
| 3 | 39/50 | 50/50 |
| 4 | 17/50 | 50/50 |

So only **≤ 59 TPE evals** remain — and fewer in practice, because the 305-entry accuracy
memo serves repeat proposals for free. That is one Colab session, maybe two.

## One-time setup (into your Google Drive, `MyDrive/tfm_nas/`)

1. **Credentials** — upload your repo `secrets/access_token` (the `KGAT_` token) and
   `secrets/kaggle_username` to `MyDrive/tfm_nas/secrets/`. (Or set `KAGGLE_USERNAME` /
   `KAGGLE_KEY` env vars, or Colab userdata secrets of the same names.)
2. **Resume cache** — upload your local `data/cp33_kaggle_out/cp33_bo_cache_r640.*.jsonl`
   to `MyDrive/tfm_nas/cache/`. These are the authoritative newest shards, so the run
   *resumes* seeds 2–4 instead of restarting. (Skip this and pass `--seed-cache` to instead
   pull the older `tfm-nas-cp33-bo-cache` Kaggle Dataset; a brand-new Drive with neither
   just starts fresh — still correct, only slower.)

Nothing else is uploaded by hand — the 1.6 GB dataset transfers Kaggle→Colab.

## Usage

Open **`colab/TFM_NAS_colab.ipynb`** in Colab (`File → Open notebook → GitHub`, or upload
it), set **Runtime → T4 GPU**, and run the cells top to bottom:

1. Mount Drive.
2. Clone/refresh the repo.
3. *(optional)* `anchor_b.py` — the yolo11s-pose gate fine-tune (~1 h on T4). **Off the
   CP 3.5 critical path** — the ceiling-first winner needs neither anchor — so run it only
   to reclaim your laptop CPU. It competes with the TPE run for the single T4.
4. `run_colab.py` — the CP 3.4 TPE @640 DoD. Every eval persists to Drive as it lands.

**Resume protocol:** if Colab disconnects (idle/12 h cap), just **re-run the `run_colab.py`
cell** — the Drive cache continues the unfinished seeds and reloads the finished ones
instantly. Repeat until the log prints `complete=true`. There is no between-session laptop
step (unlike Kaggle) because Drive *is* the persistent plane.

## Outputs (on Drive, `MyDrive/tfm_nas/out/`)

- `cp34_tpe.json` — the TPE-vs-random hypervolume verdict (the CP 3.4 DoD).
- `anchor_yolo11s_pose_640_map.json` — anchor-B accuracy (if you ran cell 3).

Download both into the laptop repo's `data/`, then select the winner over the BO∪TPE union:

```bash
python -m search.select_winner --frontier data/cp33_bo.json data/cp34_tpe.json
```

The winner is **ceiling-first** (max accuracy under `T_max = 12.75 ms`, λ-free); anchor-B
only feeds the λ *robustness check* and Phase-8 teacher scouting. See `procedure.md`
"CP 3.5 refinement".

## Files

- `colab_common.py` — shared staging: torch-pinned install, Kaggle-CLI auth + dataset pull,
  repo symlink wiring, OFA checkpoint fetch.
- `run_colab.py` — the TPE (or BO) resume entry; Drive cache + `--deadline-s`.
- `anchor_b.py` — the yolo11s-pose anchor-B fine-tune; Drive-resumable run dir.
- `TFM_NAS_colab.ipynb` — the thin notebook driver.
