# `jetson/` — finish the CP 3.3 BO search on an AGX Jetson (Kaggle-quota fallback)

Run the warm-head Bayesian-Optimization search (`search.bo`, the accuracy half of CP 3.3)
on your AGX Jetson instead of Kaggle, packaged as a Docker image built **on the board**
over SSH. This is a **direct continuation** of the @640 Kaggle run — the resume cache and
the accuracy memo carry over, so it picks up at seed 2 (seeds 0–1 already done) rather than
restarting.

## Why this is correct (and not a CLAUDE.md violation)

The Jetson here is a **compute node, not a measurement node**. `search.bo` reads
`data/lut.jsonl` (the *Orin-Nano*-measured latencies) as a **static file** — it never
benchmarks anything. So the search still optimizes for Orin Nano latency; the AGX is just
the cheapest available GPU for the 5-epoch accuracy proxy. Running PyTorch on *this* board
does not violate the "no PyTorch on the Jetson" rule, which protects the 8 GB Orin Nano's
*memory*-measurement fidelity — a different board with a different job.

On one GPU, `kaggle/run.py`'s dual-T4 fan-out collapses to a single
`search.bo --seed-list 0,1,2,3,4` call; the per-seed cache resumes unfinished seeds and
reloads finished ones. No 12 h kill → it runs to completion in one detached process, and
the cache makes it crash / reboot-resumable.

## Prerequisites (on the Jetson)

- **JetPack 5.1.x, 6.x, or 7.x** (L4T r35 / r36 / r38-r39) → torch ≥ 2.0. `deploy.sh`
  auto-detects the release and picks the base image; the build aborts if torch < 2
  (botorch's `MixedSingleTaskGP` / `qLogExpectedImprovement` require it).
- Docker with the **nvidia runtime** (`docker info | grep -i runtime` shows `nvidia`; set
  `"default-runtime": "nvidia"` in `/etc/docker/daemon.json` if not).
- **Internet during the build** (pulls the base image + pip + the SHA-pinned OFA ckpt).
- SSH key access from the laptop; `rsync` on both ends.

## Usage (from the laptop, in the repo root)

```bash
export XAVIER_HOST=user@jetson         # the SSH target

bash jetson/deploy.sh --sync           # rsync code (build context) + data (mount) + resume shards
bash jetson/deploy.sh --build          # SSH in, auto-detect L4T, docker build natively
bash jetson/deploy.sh --run            # max clocks, then run the container detached
bash jetson/deploy.sh --stop           # stop the run (progress is saved in the cache)
bash jetson/deploy.sh --resume         # continue from the cache (skips the calibrate pass)
bash jetson/deploy.sh --logs           # follow the run (docker logs -f)
bash jetson/deploy.sh --status         # container state + the current cp33_bo.json verdict
bash jetson/deploy.sh --pull           # rsync results back into data/cp33_kaggle_out/
# (no arg) = --sync then --build then --run
```

The base image is auto-selected from the detected L4T: `R38/R39 (JetPack 7) →
ultralytics/ultralytics:latest-nvidia-arm64`, `R36 → dustynv/ultralytics:r36.2.0`,
`R35 → :r35.4.1`. Override with `L4T_BASE=<image>` if a tag isn't on the registry. Tune the
run with `BUDGET=50 CALIBRATE=1` (env, passed into the container).

## How the continuation works

`deploy.sh --sync` ships the data plane to `$HOME/tfm_nas_data` on the Jetson (bind-mounted
at `/data`), including the pulled `cp33_bo_cache_r640.seed{0,1}.{bo,rs}.jsonl` resume shards
into `/data/out/`. `run_search.py` points `--cache /data/out/cp33_bo_cache_r640`, whose
shard names **match** those files, so on first run it prints:

```
[acc-memo] N prior measurement(s) loaded ...
[resume] 2/5 seeds complete; remaining evals/seed: {0: 0, 1: 0, 2: 100, 3: 100, 4: 100}
```

That line is the proof the Kaggle state carried over. New seed 2–4 shards are appended in
`/data/out/`; `--pull` brings the whole set (incl. the updated `cp33_bo.json` DoD verdict)
back to `data/cp33_kaggle_out/` on the laptop. Re-run `--run` any time to resume.

> A `--calibrate 1` pass runs first and prints `… s/eval -> 5-seed budget … GPU-h` — the
> real time budget on *this* board (your AGX Orin 64 GB is ≈ T4-class or better, single-GPU
> and 24/7). With ~31 % free memo hits, only seeds 2–4 actually fine-tune.

### First-build GPU check (JetPack 7)

The JP7 arm64 torch wheel targets Thor (sm_110); run one real CUDA op to confirm it also
carries Orin's sm_87 kernels *before* committing to the long run:

```bash
ssh $XAVIER_HOST 'docker run --rm --runtime nvidia --ipc=host tfm-nas-cp33:latest \
  python3 -c "import torch,botorch,gpytorch,ofa,ultralytics; \
  x=torch.randn(64,64,device=\"cuda\"); print(torch.__version__, (x@x).sum().item())"'
```

A number (not `no kernel image is available for execution on the device`) means Orin is
supported. If it errors, see the sm_87 row under Troubleshooting.

## Stop and resume

The run is **detached** on the Jetson (`docker run -d`), so closing your laptop or dropping
the SSH connection does **not** stop it — only `--stop`, a reboot, or a power-off does.

`search.bo` appends each finished eval to the per-seed cache shards in `/data/out/` (a
host-filesystem bind mount) *as it goes*, so stopping is safe at any moment — no "safe
point" needed:

```bash
bash jetson/deploy.sh --stop      # docker stop; at most the one in-flight eval is recomputed later
# ... hours or days later ...
bash jetson/deploy.sh --resume    # re-runs, reloads the cache, continues where it left off
```

`--resume` reloads the shards, skips every arch already scored, and prints
`[resume] K/5 seeds complete; remaining …` so you can see the progress it picked up. A
partial last line from a hard kill is tolerated (`_load_eval_cache` skips it). It also
re-applies MAXN clocks (which reset on reboot) and skips the one-off `--calibrate` pass.

**After a reboot / power-cut:** identical — `bash jetson/deploy.sh --resume`. The cache is
on disk; nothing is lost beyond the single eval that was mid-flight.

**Resume somewhere else** (another Jetson, or back on Kaggle): `--pull` the shards, then
either `--sync` to a different `XAVIER_HOST` or `bash kaggle/push.sh --resume`. The cache is
machine-agnostic, so the campaign is portable across Kaggle ↔ Jetson ↔ Jetson. Run `--pull`
any time (even while it's running) to back up progress to the laptop.

## Invariants — do NOT change (or the campaign breaks)

- **`RES=640`, `T_MAX_MS=12.75`** (set in `run_search.py`) — the @640 regime matched to the
  measured `data/baseline_anchor.json` ceiling and the cache namespace. Changing either
  invalidates the seed 0–1 cache.
- **Proxy hyperparams** `epochs=5, imgsz=640, batch=16, --freeze-head` — identical to
  Kaggle. The warm-head proxy mAP is a *ranking* signal; changing these shifts its
  distribution and makes the Jetson seeds incomparable to the cached T4 seeds. (The AGX's
  large RAM could fit a bigger batch — keep 16 for comparability.)
- **`data/lut.jsonl`** must carry the @640 rows (resolution-aware catalog, 2710→2801). It
  does today — ship it as-is; never hand-edit it.

Unlike Kaggle (which git-clones `origin/main`), the Jetson builds from the **rsync'd working
tree**, so no GitHub push is needed for code to take effect — just re-run `--sync --build`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build aborts `torch <2.0` | Board is JetPack < 5.1; flash 5.1.x+/6/7, or set a torch≥2 `L4T_BASE`. |
| `manifest unknown` on the base | That tag isn't published; pass an available one via `L4T_BASE=<image>`. |
| `permission denied ... /var/run/docker.sock` | SSH user isn't in the `docker` group: `ssh -t $XAVIER_HOST 'sudo usermod -aG docker $USER'`, then reconnect (or reboot) so it applies. |
| `no kernel image ... on the device` (JP7 Orin) | JP7 arm64 torch lacks Orin's sm_87, and jetson-ai-lab has no JP7 channel yet (jp6 + sbsa only). Cheapest fix: run the **JP6 container on the JP7 host** — `L4T_BASE=dustynv/ultralytics:r36.2.0 bash jetson/deploy.sh --build` (its torch carries sm_87; the CUDA-13 driver is forward-compatible with the container's CUDA-12.6 runtime). If the CUDA op still fails, downflash the Orin to JetPack 6.2. |
| pip can't resolve botorch | Pin it to the torch: torch 2.1 → `botorch==0.10.0`, torch 2.3+ → `botorch>=0.11` (edit the Dockerfile pip line). |
| `could not set MAXN clocks` | Run `sudo nvpmodel -m 0 && sudo jetson_clocks` on the board manually. |
| `[resume] 0/5 seeds complete` unexpectedly | The shards didn't ship — check `data/cp33_kaggle_out/cp33_bo_cache_r640.*.jsonl` exists before `--sync`. |

## On close

When `--status` shows `"complete": true`, that run's `"passes"` is the CP 3.3 DoD verdict.
Then set the λ/μ iso-J numbers from the @640 baseline `(0.877 mAP, 12.75 ms)`, advance
`state/plan_state.yaml`, and write the `procedure.md` "CP 3.3 CLOSED" entry.
