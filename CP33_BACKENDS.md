# CP 3.3 search — running across two backends (Kaggle + AGX Jetson Orin)

The CP 3.3 warm-head BO campaign (`search.bo`, 5 seeds × 50-eval budget, `RES=640`,
`T_max=12.75 ms`) runs on **either** Kaggle GPU or the AGX Jetson Orin — whichever has
quota/availability at the time. Both write to the **same cache format**
(`cp33_bo_cache_r640.seed{N}.{bo,rs}.jsonl`), so the campaign is portable: stop on one
backend, hand its progress to the other, resume. This doc is the single reference for
that workflow. Backend-specific detail lives in `kaggle/README.md` and `jetson/README.md`;
this doc is about the parts that span both — checking progress and switching backends.

## Why this works (read this once)

- `search.bo`'s eval cache is **append-only JSONL, keyed by canonical arch**, resumable
  from any prior state regardless of which machine produced it.
- Both `kaggle/run.py` and `jetson/run_search.py` run the identical `search.bo` CLI with
  the identical config (`RES=640`, `T_MAX_MS=12.75`, `epochs=5/imgsz=640/batch=16/
  freeze-head`) — see "Invariants" below. Never change these; a mismatch makes seeds
  from the two backends incomparable in the merged verdict.
- The **shared local staging point** on your laptop is `data/cp33_kaggle_out/` — both
  `kaggle/push.sh --pull` and `jetson/deploy.sh --pull` write there. That's what makes
  switching backends possible: whichever backend you pulled from *last* is what the
  *other* backend's next resume will pick up.

## Continue on the Jetson

```bash
export XAVIER_HOST=CVAR@192.168.55.1

bash jetson/deploy.sh --resume    # after the very first launch, always --resume (not --run)
bash jetson/deploy.sh --logs      # follow live logs (Ctrl-C to stop watching, not the run)
bash jetson/deploy.sh --stop      # stop; progress is safe in the on-disk cache
bash jetson/deploy.sh --status    # container state + current verdict JSON
bash jetson/deploy.sh --pull      # copy results to data/cp33_kaggle_out/ (safe anytime)
```

After any `search/`, `eval/`, `catalog/`, etc. code change: `git commit`, then
`--sync` (ships code) → `--build` (rebuilds the image) → `--resume`.

**Checking progress directly on the Jetson** (no SSH needed if you're already logged in
at the board):

```bash
docker logs -f cp33                      # live per-eval progress: "[eval] seed=N bo/rs k/50 ..."
cd ~/tfm_nas_data/out
for s in 0 1 2 3 4; do echo "seed$s: bo=$(wc -l < cp33_bo_cache_r640.seed$s.bo.jsonl 2>/dev/null||echo 0)/50 rs=$(wc -l < cp33_bo_cache_r640.seed$s.rs.jsonl 2>/dev/null||echo 0)/50"; done
timeout 5 sudo tegrastats --interval 1000 | grep -oE "GR3D_FREQ [0-9]+%"   # GPU load
sudo nvpmodel -q | grep "Power Mode"     # confirm still MAXN (a reboot resets this)
```

Full detail (build, board-specific sm_87 notes, troubleshooting): `jetson/README.md`.

## Continue on Kaggle

```bash
bash kaggle/push.sh --resume   # pulls the last session's output, versions the resume-cache Dataset
```

Then **in the Kaggle UI** (never via a fresh API kernel push — see the hard rule below):
refresh the **`tfm-nas-cp33-bo-cache`** input to its newest version, keep **GPU T4 x2**,
**Save & Run All**. Watch for `[resume] restored N shard(s)` in the log.

```bash
bash kaggle/push.sh --status   # kernel run status
bash kaggle/push.sh --pull     # copy results to data/cp33_kaggle_out/ (safe anytime)
```

Full detail (the 5-session protocol, dual-GPU fan-out, the accuracy memo): `kaggle/README.md`.

### Hard rule — never re-push the kernel via the API

`bash kaggle/push.sh` **with no flag** pushes+runs the kernel fresh via the API, which
**resets the accelerator to P100 and crashes** this GPU-bound run. The kernel already
exists — only ever resume it from the **Kaggle UI** (refresh dataset input → Save & Run
All). `--resume`/`--pull`/`--cache`/`--status` are all safe API calls; the no-flag form is
the only dangerous one.

## Switching backends

The two directions are **not symmetric** — read this before running anything.

### Jetson → Kaggle (Jetson made progress, hand it to Kaggle)

```bash
bash jetson/deploy.sh --stop     # optional but tidy: avoids pulling a shard mid-write
bash jetson/deploy.sh --pull     # Jetson's advanced shards -> data/cp33_kaggle_out/
bash kaggle/push.sh --cache      # version the Kaggle resume-cache Dataset from THAT folder
```

**Do NOT use `bash kaggle/push.sh --resume` for this direction.** `--resume` is
`--pull` (fetches Kaggle's *last* — now stale — kernel output) **then** `--cache`. That
`--pull` step would overwrite the Jetson's freshly-placed, more-advanced shards in
`data/cp33_kaggle_out/` with Kaggle's older ones, silently discarding the Jetson's
progress. `--cache` alone stages whatever is *currently* in `data/cp33_kaggle_out/`
without re-fetching from Kaggle first — that's what makes it safe here.

Then, in the Kaggle UI: refresh `tfm-nas-cp33-bo-cache` to its newest version, GPU T4 x2,
Save & Run All.

### Kaggle → Jetson (Kaggle made progress, hand it to the Jetson)

```bash
bash kaggle/push.sh --resume     # here --resume IS correct: Kaggle is the freshest source
bash jetson/deploy.sh --sync     # ships data/cp33_kaggle_out/*.jsonl -> the Jetson's /data/out
bash jetson/deploy.sh --resume
```

This direction is safe because `kaggle/push.sh --resume`'s pull genuinely fetches the
newest state (nothing else has written to `data/cp33_kaggle_out/` since), and
`jetson/deploy.sh --sync` only ever copies *from* the laptop *to* the Jetson — it can't
be clobbered by a stale pull the way the Kaggle-bound direction can.

### Jetson → a different Jetson

```bash
bash jetson/deploy.sh --pull                       # from the old board
XAVIER_HOST=user@new-host bash jetson/deploy.sh --sync --build --resume   # to the new board
```

### Quick reference

| From \ To | Kaggle | Jetson |
|---|---|---|
| **Kaggle** | `push.sh --resume` + UI Save & Run All | `push.sh --resume` → `deploy.sh --sync --resume` |
| **Jetson** | `deploy.sh --pull` → `push.sh --cache` (NOT `--resume`) + UI Save & Run All | `deploy.sh --pull` → (new host) `--sync --build --resume` |

## Invariants — must match on both backends

- `RES=640`, `T_MAX_MS=12.75` — the @640 regime, matched to the measured
  `data/baseline_anchor.json` ceiling and the cache namespace (`cp33_bo_cache_r640.*`).
- Proxy hyperparams: `epochs=5, imgsz=640, batch=16, --freeze-head`.
- `data/lut.jsonl` must carry the @640 rows (it does; never hand-edit it).

Changing any of these on one backend but not the other makes that backend's seeds
incomparable in the merged verdict — don't.

## Done when

```bash
cat data/cp33_kaggle_out/cp33_bo.json | python3 -m json.tool | grep -E '"complete"|"passes"'
```
`"complete": true` — every seed spent its full budget across however many backend
switches it took. `"passes"` is the CP 3.3 DoD verdict. Then: set the λ/μ iso-J numbers
from the @640 baseline `(0.877 mAP, 12.75 ms)`, advance `state/plan_state.yaml`, write the
`procedure.md` "CP 3.3 CLOSED" entry.
