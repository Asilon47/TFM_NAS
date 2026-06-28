#!/usr/bin/env python3
"""Kaggle kernel entry — CP 3.3 warm-head Bayesian-Optimization search.

Pushed by ``kaggle/push.sh`` and run on Kaggle GPU. It clones the (public) repo,
installs the NAS + BO stack *without* disturbing Kaggle's torch, wires the attached
data Dataset (gate dataset + LUT + NSGA-II seeds + the frozen gate head donor), fetches
the SHA-pinned OFA checkpoint, then runs the search (``search.bo``).

Edit the CONFIG block for the full 5-seed DoD run; the defaults are a cheap *proving*
run (calibrate + 1 seed) that verifies the whole pipeline within a single GPU session.
Outputs land in ``/kaggle/working`` and are downloadable as the kernel output / pulled
by ``push.sh --pull``.
"""
import os
import subprocess
import sys
from pathlib import Path

# ---- CONFIG (edit for the full DoD run) -------------------------------------
REPO_URL  = "https://github.com/Asilon47/TFM_NAS.git"
DATASET   = "tfm-nas-gate-pose"   # attached Kaggle Dataset slug (no <user>/ prefix)
RES       = 224                   # LUT key resolution: 224 until the @640 sweep lands, then 640
T_MAX_MS  = 16.7                  # hard latency ceiling = min(baseline, 60 FPS)
CALIBRATE = 2                     # time N real warm-head evals (0 to skip)
SEEDS     = 1                     # -> 5 for the DoD
BUDGET    = 8                     # -> 50 for the DoD
N_INIT    = 4                     # -> 20 for the DoD
# -----------------------------------------------------------------------------


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
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

    # 5. search: timed calibration first (de-risks the budget), then the BO run.
    common = (f"--device cuda --imgsz 640 --res {RES} --lut data/lut.jsonl "
              f"--head-weights {head} --freeze-head --t-max-ms {T_MAX_MS}")
    if CALIBRATE:
        sh(f"{sys.executable} -m search.bo --calibrate {CALIBRATE} {common}")
    out = work / "cp33_bo.json"
    cmd = (f"{sys.executable} -m search.bo --seeds {SEEDS} --budget {BUDGET} "
           f"--n-init {N_INIT} {common} --out {out} --cache {work}/cp33_bo_cache")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    # search.bo exits 1 on a DoD-FAIL *verdict* (a valid result — e.g. the cheap proving
    # run). The kernel's job is to PRODUCE results, so only fail if none were written;
    # the pass/fail verdict lives in the JSON.
    if not out.exists():
        raise SystemExit(f"search.bo produced no results (rc={rc})")
    print(f"done (rc={rc}; DoD pass/fail is in the JSON) -> {out}", flush=True)


if __name__ == "__main__":
    main()
