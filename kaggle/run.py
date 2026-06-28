#!/usr/bin/env python3
"""Kaggle kernel entry — CP 3.3 warm-head Bayesian-Optimization search.

Pushed by ``kaggle/push.sh`` and run on Kaggle GPU. It clones the (public) repo,
installs the NAS + BO stack *without* disturbing Kaggle's torch, wires the attached
data Dataset (gate dataset + LUT + NSGA-II seeds + the frozen gate head donor), fetches
the SHA-pinned OFA checkpoint, then runs the search (``search.bo``).

The CONFIG block is the full 5-seed DoD, *resumable across Kaggle sessions*: each commit
stops starting new evals after ``DEADLINE_H`` hours — a clean boundary under Kaggle's 12 h
kill — having appended per-seed caches to ``/kaggle/working``. Re-run the kernel with its
own previous output attached as an input to restore those caches and continue, until the
merged JSON reports ``complete: true``. Outputs land in ``/kaggle/working`` and are pulled
by ``push.sh --pull``.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---- CONFIG (the full 5-seed DoD; resumable across sessions) ----------------
REPO_URL   = "https://github.com/Asilon47/TFM_NAS.git"
DATASET    = "tfm-nas-gate-pose"  # attached Kaggle Dataset slug (no <user>/ prefix)
RES        = 224                  # LUT key resolution: 224 until the @640 sweep lands
T_MAX_MS   = 16.7                 # hard latency ceiling = min(baseline, 60 FPS)
CALIBRATE  = 1                    # time N real evals first (also warms ultralytics)
SEEDS      = 5                    # the 5-seed DoD
BUDGET     = 50                   # BO budget per seed (decision D2)
N_INIT     = 20                   # initial-design size
DEADLINE_H = 10.5                 # stop new evals after this many h (clean resumable boundary)
# -----------------------------------------------------------------------------


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    START = time.time()
    work = Path("/kaggle/working")
    repo = work / "TFM_NAS"

    # 1. code: shallow clone (public; GITHUB_TOKEN Kaggle secret supports a private repo)
    token = os.environ.get("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
    if not repo.exists():
        sh(f"git clone --depth 1 {url} {repo}")
    os.chdir(repo)

    # 2. deps: pin Kaggle's torch so botorch/ultralytics never upgrade it (the same
    #    discipline the .venv uses for the 2.3.1 pin) — then add the NAS + BO stack.
    torch_ver = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"]).decode().strip()
    constraint = work / "constraints.txt"
    constraint.write_text(f"torch=={torch_ver}\n")
    sh(f"{sys.executable} -m pip install -q --constraint {constraint} "
       "'ofa==0.1.0.post202307202001' 'ultralytics>=8.3' gdown botorch gpytorch")

    # 3. data: locate the attached Dataset ROBUSTLY. Kaggle's mount path is not
    #    guaranteed to be /kaggle/input/<slug>, so search /kaggle/input and fail
    #    loudly (printing the tree) rather than silently skip a missing file.
    input_root = Path("/kaggle/input")
    print("=== /kaggle/input (2 levels) ===", flush=True)
    if input_root.exists():
        for d in sorted(input_root.iterdir()):
            print("  ", d, flush=True)
            if d.is_dir():
                for sub in sorted(d.iterdir())[:25]:
                    print("      ", sub, flush=True)
    else:
        print("  (/kaggle/input missing — no dataset attached)", flush=True)
    print("=== end tree ===", flush=True)

    def find(name: str):
        hits = sorted(input_root.rglob(name)) if input_root.exists() else []
        return hits[0] if hits else None

    lut_src = find("lut.jsonl")
    frontier_src = find("phase3_nsga2_frontier.json")
    head = find("gate_best.pt")                              # warm-head donor (frozen)
    yaml_src = find("dataset.yaml")                          # the gate-pose data root
    missing = [n for n, v in (("lut.jsonl", lut_src), ("gate_best.pt", head),
                              ("dataset.yaml", yaml_src)) if v is None]
    if missing:
        raise SystemExit(f"FATAL: {missing} not found under /kaggle/input — is the "
                         f"'{DATASET}' dataset attached? See the tree above.")

    sh(f"rm -rf dataset && ln -s {yaml_src.parent} dataset")  # detect.evaluate.DEFAULT_DATA_YAML
    Path("data").mkdir(exist_ok=True)
    sh(f"ln -sf {lut_src} data/lut.jsonl")
    if frontier_src:
        sh(f"ln -sf {frontier_src} data/phase3_nsga2_frontier.json")

    # 4. OFA supernet checkpoint (internet on; SHA-verified in supernet/download_ofa.py)
    sh(f"{sys.executable} -m supernet.download_ofa")

    # 4.5 resume: Kaggle wipes /kaggle/working between commits, so the eval caches must
    #     round-trip through the persistent /kaggle/input plane. If a prior run's output
    #     is attached as an input (Add Input -> Notebook -> this notebook), restore its
    #     cache shards so the workers continue instead of restarting. First session: none.
    restored = 0
    for src in (sorted(input_root.rglob("cp33_bo_cache*.jsonl"))
                if input_root.exists() else []):
        dst = work / src.name
        if not dst.exists():            # never clobber a shard this session already wrote
            shutil.copy(src, dst)
            restored += 1
    print(f"[resume] restored {restored} eval-cache shard(s) from prior output", flush=True)

    # 5. search: timed calibration first (de-risks the budget), then the BO run.
    common = (f"--device cuda --imgsz 640 --res {RES} --lut data/lut.jsonl "
              f"--head-weights {head} --freeze-head --t-max-ms {T_MAX_MS}")
    if CALIBRATE:
        sh(f"{sys.executable} -m search.bo --calibrate {CALIBRATE} {common}")
    out = work / "cp33_bo.json"
    cache = work / f"cp33_bo_cache_r{RES}"   # RES-namespaced so @224 and @640 caches stay distinct
    deadline_s = max(600, int(DEADLINE_H * 3600 - (time.time() - START)))
    budget = f"--budget {BUDGET} --n-init {N_INIT} --deadline-s {deadline_s}"
    print(f"[deadline] workers stop new evals after ~{deadline_s / 3600:.1f} h", flush=True)
    ngpu = int(subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
    ).decode().strip() or "0")
    print(f"[gpu] {ngpu} CUDA device(s) visible", flush=True)

    if ngpu >= 2 and SEEDS > 1:
        # Fan the seeds across GPUs (one worker per device) and run them in PARALLEL,
        # then merge -> ~halves wall-clock AND Kaggle GPU quota. Seed indices stay
        # disjoint, so the per-seed caches never collide and the run is reproducible.
        # The calibrate step above already warmed ultralytics (settings.json + Arial.ttf);
        # a short stagger makes any first-touch race a non-issue if calibrate was skipped.
        per = -(-SEEDS // ngpu)  # ceil -> balance seeds across the GPUs
        procs, parts, start = [], [], 0
        for g in range(ngpu):
            count = min(per, SEEDS - start)
            if count <= 0:
                break
            part = work / f"cp33_bo.part{g}.json"
            parts.append(part)
            wcmd = (f"{sys.executable} -m search.bo --seed-start {start} --seeds {count} "
                    f"{budget} {common} --out {part} --cache {cache}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            print(f"+ [gpu{g}] seeds {start}..{start + count - 1}", flush=True)
            procs.append(subprocess.Popen(wcmd, shell=True, env=env))
            start += count
            time.sleep(10)  # let the first worker win any ultralytics first-touch race
        for pr in procs:
            pr.wait()  # ignore rc: a worker exits 1 on a partial DoD-FAIL; check outputs
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise SystemExit(f"GPU worker(s) produced no output: {missing}")
        sh(f"{sys.executable} -m search.bo --merge {' '.join(map(str, parts))} --out {out}")
    else:
        # 1 GPU (or 1 seed): run sequentially. search.bo exits 1 on a DoD-FAIL *verdict*
        # (a valid result), so only fail if it wrote no output — the verdict is in the JSON.
        cmd = (f"{sys.executable} -m search.bo --seeds {SEEDS} {budget} "
               f"{common} --out {out} --cache {cache}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode
        if not out.exists():
            raise SystemExit(f"search.bo produced no results (rc={rc})")
    # report completion so you know whether to re-run (resume) or stop
    try:
        payload = json.loads(out.read_text())
        done = payload.get("complete")
        nxt = ("COMPLETE — DoD final" if done else
               "PARTIAL — re-run the kernel (with this output attached as input) to resume")
        print(f"[done] {out.name}: complete={done} passes={payload.get('passes')} -> {nxt}",
              flush=True)
    except (OSError, ValueError) as e:
        print(f"[done] wrote {out} (could not parse completion: {e})", flush=True)


if __name__ == "__main__":
    main()
