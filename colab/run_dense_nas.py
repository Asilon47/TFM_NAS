#!/usr/bin/env python3
"""Colab entry — Stage-3 dense-space NAS wave-2 on a free T4 (compute-scarce fallback).

Kaggle quota ran out mid-wave-2 (2026-07-13), so the recalibrated re-search moves to Colab.
A single-GPU sibling of the ``kaggle/run.py`` ``dense_nas`` block: it pins Colab's CUDA torch,
installs ONLY the light stack (ultralytics + optuna — no OFA, dense_nas is supernet-free),
stages the gate dataset off Kaggle, then runs ``search.dense_nas`` with the **wave-2 corrected
search** (constrained box ``STAGE_HI_FEASIBLE`` + the physical act_mbytes fence). Per-candidate
row files are written straight to a **Google Drive** folder, so a Colab disconnect loses at most
the in-flight train and a re-run resumes (the TPE objective reloads existing rows).

    python colab/run_dense_nas.py --drive /content/drive/MyDrive/tfm_nas --budget 20 --seed 20

Re-run the same cell after a disconnect (same --seed, same --drive) to continue; bump --seed for
an independent parallel study. Pull the rows back with ``colab/`` → local ``data/dense_nas_w2/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "colab"))
import colab_common as C  # noqa: E402

PROXY_EPOCHS = 30          # G3-validated proxy (ρ=1.000 vs 100-ep oracles)
CEILING_FP32 = 12.0        # physical act_mbytes fence → measured ≤ ~baseline 12.75 ms
STAGE_HI = "feasible"      # the wave-2 constrained box (STAGE_HI_FEASIBLE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-3 dense-NAS wave-2 on Colab T4.")
    ap.add_argument("--drive", type=Path, default=Path(C.DRIVE_DEFAULT),
                    help="mounted Google Drive root for durable row output")
    ap.add_argument("--budget", type=int, default=20, help="TPE trials (infeasible rejected free)")
    ap.add_argument("--seed", type=int, default=20, help="TPE study seed (bump for parallel)")
    ap.add_argument("--oracle-tags", type=str, default=None,
                    help="comma sNN-.. tags → 100-ep oracle re-train instead of the search")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (oracle=100)")
    args = ap.parse_args()

    start = time.time()
    os.chdir(REPO)

    # 1. stack + data (machine-to-machine; dense_nas needs neither OFA nor the LUT) ----
    user = C.ensure_kaggle_credentials(args.drive)
    C.pin_torch_and_install("'ultralytics>=8.3' optuna")
    staged = C.stage_kaggle_dataset(user, Path("/content/kagdata"))
    C.wire_repo_data(REPO, staged)   # symlinks repo/dataset → the gate images + yaml

    # 2. durable output on Drive (rows persist across disconnects → resumable) ---------
    out_dir = args.drive / "dense_nas_w2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.oracle_tags:
        cmd = (f"{sys.executable} -m search.dense_nas --oracle-tags {args.oracle_tags} "
               f"--proxy-epochs {args.epochs or 100} --data dataset/dataset.yaml "
               f"--out-dir {out_dir} --device 0")
    else:
        cmd = (f"{sys.executable} -m search.dense_nas --budget {args.budget} --seed {args.seed} "
               f"--proxy-epochs {args.epochs or PROXY_EPOCHS} --ceiling-fp32-ms {CEILING_FP32} "
               f"--stage-hi {STAGE_HI} --data dataset/dataset.yaml --out-dir {out_dir} --device 0")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode

    # 3. report the feasible frontier so far (survives disconnects on Drive) -----------
    rows = []
    for f in glob.glob(str(out_dir / "dense_s*.row.json")):
        try:
            rows.append(json.loads(Path(f).read_text()))
        except (OSError, ValueError):
            pass
    rows = [r for r in rows if "map" in r]
    print(f"\n[done] rc={rc}  {len(rows)} rows on Drive ({out_dir})", flush=True)
    for r in sorted(rows, key=lambda r: -r["map"])[:8]:
        print(f"  {r['tag']:26s} map={r['map']:.4f} pred_fp32={r.get('pred_fp32_ms')} "
              f"params={r.get('params'):,}  ({time.time() - start:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
