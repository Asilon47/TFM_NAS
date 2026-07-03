#!/usr/bin/env python3
"""Colab entry — resume the CP 3.4 TPE (or CP 3.3 BO) warm-head search on a free T4.

A single-GPU sibling of ``kaggle/run.py``. It pins Colab's CUDA torch, installs the NAS
stack, pulls the big data from the ``tfm-nas-gate-pose`` Kaggle Dataset, fetches the
SHA-pinned OFA checkpoint, then runs ``search.<METHOD>`` over all seeds with:

* ``--cache`` pointed **straight at a Google Drive folder**, so every appended eval is
  durably persisted the instant it is written (a Colab disconnect loses at most one eval),
  and a later session resumes the unfinished seeds while reloading finished ones instantly;
* ``--deadline-s`` so the workers stop starting new evals a safe margin before Colab's
  ~12 h kill, leaving a clean, resumable boundary.

The @640 TPE DoD is a RESUME: seeds 0/1 already spent their full budget and the entire
random-search control is cached, so only seeds 2–4's remaining TPE proposals (minus
accuracy-memo hits) still cost GPU. Seed the Drive cache once from your local
``data/cp33_kaggle_out/*.jsonl`` (or let ``--seed-cache`` pull the Kaggle cache Dataset).

    python colab/run_colab.py --drive /content/drive/MyDrive/tfm_nas
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "colab"))
import colab_common as C  # noqa: E402

# ---- CONFIG (the full 5-seed DoD; resumable across Colab sessions) -----------
METHOD = "tpe"       # "tpe" (CP 3.4 fallback) or "bo" (CP 3.3, closed) — same DoD/oracle/ceiling
RES = 640            # @640 LUT + 12.75 ms baseline landed -> the real DoD regime
T_MAX_MS = 12.75     # min(yolo11n-pose @640 = 12.755 ms, 60-FPS 16.7 ms) at the deploy res
SEEDS = 5
BUDGET = 50          # per-seed budget (decision D2)
N_INIT = 20
CALIBRATE = 1        # time one real eval first (also warms ultralytics)
DEADLINE_H = 11.0    # stop starting new evals after this many h (Colab ~12 h ceiling)
OUT_NAME = {"bo": "cp33_bo.json", "tpe": "cp34_tpe.json"}[METHOD]
# -----------------------------------------------------------------------------


def seed_cache_from_kaggle(user: str, cache_prefix: Path) -> None:
    """Best-effort: if the Drive cache is empty, seed it from the Kaggle resume Dataset so
    even a brand-new Drive resumes. (The AUTHORITATIVE seed is uploading your local
    ``data/cp33_kaggle_out/*.jsonl`` to the Drive cache dir — that may be newer.)"""
    existing = list(cache_prefix.parent.glob(cache_prefix.name + "*"))
    if existing:
        print(f"[cache] {len(existing)} shard(s) already on Drive — not seeding", flush=True)
        return
    try:
        tmp = Path("/content/_cacheseed")
        C.sh(f"kaggle datasets download -d {user}/{C.CACHE_SLUG} -p {tmp}")
        C._unzip_all(tmp)
        n = 0
        for shard in tmp.rglob("cp33_bo_cache_r*.jsonl"):
            dst = cache_prefix.parent / shard.name
            if not dst.exists():
                dst.write_bytes(shard.read_bytes())
                n += 1
        print(f"[cache] seeded {n} shard(s) from {user}/{C.CACHE_SLUG}", flush=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[cache] no Kaggle seed ({e}); starting from whatever is on Drive", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume the CP 3.4 TPE search on Colab T4.")
    ap.add_argument("--drive", type=Path, default=Path(C.DRIVE_DEFAULT),
                    help="mounted Google Drive root for durable caches + outputs")
    ap.add_argument("--seed-cache", action="store_true",
                    help="if the Drive cache is empty, seed it from the Kaggle cache Dataset")
    args = ap.parse_args()

    start = time.time()
    os.chdir(REPO)

    # 1. credentials + stack + data (all machine-to-machine; no hand upload) ----
    user = C.ensure_kaggle_credentials(args.drive)
    C.pin_torch_and_install("'ofa==0.1.0.post202307202001' 'ultralytics>=8.3' gdown optuna")
    staged = C.stage_kaggle_dataset(user, Path("/content/kagdata"))
    paths = C.wire_repo_data(REPO, staged)
    C.download_ofa(REPO)

    # 2. durable plane on Drive ------------------------------------------------
    cache = args.drive / "cache" / f"cp33_bo_cache_r{RES}"  # search appends .seed{s}.{method}.jsonl
    out = args.drive / "out" / OUT_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.seed_cache:
        seed_cache_from_kaggle(user, cache)

    # 3. search: calibrate (de-risks the budget), then all seeds sequentially ---
    common = (f"--device cuda --imgsz 640 --res {RES} --lut data/lut.jsonl "
              f"--head-weights {paths['donor']} --freeze-head --t-max-ms {T_MAX_MS}")
    if paths["memo"]:
        common += f" --acc-memo {paths['memo']}"
        print(f"[acc-memo] {paths['memo']} — prior fine-tunes reused free", flush=True)

    if CALIBRATE:
        C.sh(f"{sys.executable} -m search.{METHOD} --calibrate {CALIBRATE} {common}")

    deadline_s = max(600, int(DEADLINE_H * 3600 - (time.time() - start)))
    print(f"[deadline] stop starting new evals after ~{deadline_s / 3600:.1f} h", flush=True)
    allseeds = ",".join(str(s) for s in range(SEEDS))
    cmd = (f"{sys.executable} -m search.{METHOD} --seed-list {allseeds} "
           f"--budget {BUDGET} --n-init {N_INIT} --deadline-s {deadline_s} "
           f"{common} --out {out} --cache {cache}")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode  # rc=1 is a valid DoD-FAIL verdict
    if not out.exists():
        raise SystemExit(f"search.{METHOD} produced no output (rc={rc})")

    # 4. report completion so you know whether to re-run (resume) or stop -------
    try:
        payload = json.loads(out.read_text())
        done = payload.get("complete")
        nxt = ("COMPLETE — DoD final; pull it from Drive" if done else
               "PARTIAL — re-run this cell (Drive cache resumes) until complete=true")
        print(f"[done] {out.name}: complete={done} passes={payload.get('passes')} -> {nxt}",
              flush=True)
    except (OSError, ValueError) as e:
        print(f"[done] wrote {out} (could not parse completion: {e})", flush=True)


if __name__ == "__main__":
    main()
