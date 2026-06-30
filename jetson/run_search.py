#!/usr/bin/env python3
"""AGX Jetson entry — CP 3.3 warm-head Bayesian-Optimization search (single-GPU).

A continuation of the Kaggle run (``kaggle/run.py``) minus the Kaggle plumbing: no git
clone (the code is baked into the image), no ``/kaggle/input`` dataset round-trip (data is
bind-mounted at ``/data``), and the dual-T4 fan-out collapses to the 1-GPU branch — one
``search.bo --seed-list 0,1,2,3,4`` call whose per-seed cache resumes unfinished seeds and
reloads finished ones. There is no 12 h kill on the Jetson, so it runs to completion in one
process; the eval cache makes it crash / reboot-resumable.

The board is a COMPUTE node, not a measurement node: ``search.bo`` reads ``data/lut.jsonl``
(the Orin-Nano-measured latencies) as a static file, so the search still optimizes for Orin
Nano latency. Mount the data plane at ``/data`` (see ``jetson/deploy.sh``)::

    /data/lut.jsonl                   the @640 LUT (resolution-aware catalog)
    /data/gate_best.pt                frozen warm-head donor
    /data/cp33_acc_memo.json          prior fine-tunes (free reuse; ~31% hit rate)
    /data/phase3_nsga2_frontier.json  BO warm-start seeds
    /data/dataset/dataset.yaml        the gate-pose target
    /data/out/cp33_bo_cache_r640.*    the resume shards pulled from Kaggle
    /data/out/cp33_bo.json            the merged DoD verdict (written here)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/workspace/TFM_NAS")
sys.path.insert(0, str(REPO))  # in-process `from search.bo import ...` (script dir != repo root)

# ---- CONFIG (the @640 real-DoD regime — keep identical to the Kaggle run) ----------
DATA = Path(os.environ.get("DATA_DIR", "/data"))
OUT = DATA / "out"
RES = 640                 # @640 LUT + 12.75 ms baseline -> the real DoD regime
T_MAX_MS = 12.75          # hard latency ceiling = measured yolo11n-pose @640 (baseline_anchor.json)
SEEDS = 5
N_INIT = 20
BUDGET = int(os.environ.get("BUDGET", "50"))       # BO budget per seed (decision D2)
CALIBRATE = int(os.environ.get("CALIBRATE", "1"))  # time N real evals first (also warms ultralytics)
# ------------------------------------------------------------------------------------


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO)


def find(name: str):
    hits = sorted(DATA.rglob(name)) if DATA.exists() else []
    return hits[0] if hits else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Locate the mounted data plane robustly; fail loudly (printing the tree) like run.py.
    print("=== /data (2 levels) ===", flush=True)
    if DATA.exists():
        for d in sorted(DATA.iterdir()):
            print("  ", d, flush=True)
            if d.is_dir():
                for sub in sorted(d.iterdir())[:12]:
                    print("      ", sub, flush=True)
    print("=== end tree ===", flush=True)

    lut = find("lut.jsonl")
    head = find("gate_best.pt")
    memo = find("cp33_acc_memo.json")
    yaml_src = find("dataset.yaml")
    frontier = find("phase3_nsga2_frontier.json")
    missing = [n for n, v in (("lut.jsonl", lut), ("gate_best.pt", head),
                              ("dataset.yaml", yaml_src)) if v is None]
    if missing:
        raise SystemExit(f"FATAL: {missing} not found under {DATA} — is the data volume "
                         "mounted (-v <host>:/data)? See the tree above.")

    # Wire the repo's expected paths to the mounted files (mirror run.py:102-106).
    sh(f"rm -rf dataset && ln -s {yaml_src.parent} dataset")   # detect.evaluate.DEFAULT_DATA_YAML
    (REPO / "data").mkdir(exist_ok=True)
    sh(f"ln -sf {lut} data/lut.jsonl")
    if frontier:  # search.bo._load_nsga_frontier reads ROOT/data/phase3_nsga2_frontier.json
        sh(f"ln -sf {frontier} data/phase3_nsga2_frontier.json")

    common = (f"--device cuda --imgsz 640 --res {RES} --lut {lut} "
              f"--head-weights {head} --freeze-head --t-max-ms {T_MAX_MS}")
    if memo:  # prior fine-tunes are accuracy-only (imgsz-fixed) -> valid across resolutions
        common += f" --acc-memo {memo}"
        print(f"[acc-memo] attached {memo}", flush=True)

    if CALIBRATE:  # report real per-eval wall-clock on THIS board before committing days
        sh(f"python3 -m search.bo --calibrate {CALIBRATE} {common}")

    cache = OUT / f"cp33_bo_cache_r{RES}"   # shard names match the pulled Kaggle files
    out = OUT / "cp33_bo.json"

    # How much each seed still owes (from the restored caches) -> a continuity check.
    from search.bo import seed_remaining_evals
    remaining = {s: seed_remaining_evals(cache, s, BUDGET) for s in range(SEEDS)}
    ndone = sum(1 for s in remaining if remaining[s] == 0)
    print(f"[resume] {ndone}/{SEEDS} seeds complete; remaining evals/seed: {remaining}",
          flush=True)

    # Single GPU: one call over all seeds; the cache resumes the unfinished ones and reloads
    # the finished ones instantly. rc=1 is a valid DoD-FAIL verdict (data in the JSON), so
    # only a missing output file is fatal.
    seed_list = ",".join(str(s) for s in range(SEEDS))
    cmd = (f"python3 -m search.bo --seed-list {seed_list} "
           f"--budget {BUDGET} --n-init {N_INIT} {common} --out {out} --cache {cache}")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=REPO).returncode
    if not out.exists():
        raise SystemExit(f"search.bo produced no results (rc={rc})")

    payload = json.loads(out.read_text())
    done = payload.get("complete")
    nxt = ("COMPLETE — DoD final" if done else
           "PARTIAL — re-run `deploy.sh --run` to resume (the cache continues)")
    print(f"[done] {out.name}: complete={done} passes={payload.get('passes')} -> {nxt}",
          flush=True)


if __name__ == "__main__":
    main()
