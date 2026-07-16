#!/usr/bin/env python3
"""Colab entry — fine-tune yolo11s-pose on the gate set (anchor-B accuracy) on a free T4.

The GPU sibling of the laptop ``train_anchor_s.py``: same recipe, but ``device=0`` and the
run directory lives on **Google Drive** so the run is resumable across Colab's ~12 h VM
recycle exactly as the laptop version is across reboots. On finish it evaluates ``best.pt``
at imgsz 640 and writes the accuracy half of anchor B beside the TPE outputs on Drive.

Anchor B is OFF the CP 3.5 critical path — the ceiling-first winner needs neither anchor.
It feeds only the λ **robustness check** and Phase-8 teacher scouting, so a "converged-enough"
gate mAP is all this needs (λ is insensitive to it). ~1 h on a T4 vs ~18 h on the laptop CPU.

    python colab/anchor_b.py --drive /content/drive/MyDrive/tfm_nas
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "colab"))
import colab_common as C  # noqa: E402

NAME = "gate_anchor_yolo11s"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune yolo11s-pose for anchor B on Colab T4.")
    ap.add_argument("--drive", type=Path, default=Path(C.DRIVE_DEFAULT))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16, help="T4-sized (the laptop CPU used 4)")
    args = ap.parse_args()

    # data: reuse the same Kaggle Dataset the TPE run pulls (no separate upload) ----
    user = C.ensure_kaggle_credentials(args.drive)
    C.pin_torch_and_install("'ultralytics>=8.3'")
    staged = C.stage_kaggle_dataset(user, Path("/content/kagdata"))
    yaml_src = C.find(staged, "dataset.yaml")
    if yaml_src is None:
        raise SystemExit(f"dataset.yaml not found under {staged}")

    # local-path rewrite so the dataloader resolves against the staged copy
    spec = yaml.safe_load(yaml_src.read_text())
    spec["path"] = str(yaml_src.parent)
    data_yaml = Path(tempfile.mkstemp(suffix="_dataset.yaml", text=True)[1])
    data_yaml.write_text(yaml.safe_dump(spec))
    print(f"[data] {data_yaml} (path -> {spec['path']})", flush=True)

    from ultralytics import YOLO

    # run dir on Drive -> resumable across Colab sessions (same device each time = cuda)
    run_dir = args.drive / "runs" / "pose" / "experiments" / NAME
    last_ckpt = run_dir / "weights" / "last.pt"
    if last_ckpt.exists():
        # re-materialise the data yaml at the path baked into args.yaml if a recycle wiped it
        ckpt_data = Path(yaml.safe_load((run_dir / "args.yaml").read_text())["data"])
        if not ckpt_data.exists():
            ckpt_data.write_text(yaml.safe_dump(spec))
            print(f"[resume] re-created missing data yaml at {ckpt_data}", flush=True)
        print(f"[resume] continuing from {last_ckpt}", flush=True)
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        print("[fresh] no checkpoint — training from COCO weights", flush=True)
        model = YOLO("yolo11s-pose.pt")  # ultralytics auto-downloads the COCO weights
        model.train(
            data=str(data_yaml), epochs=args.epochs, patience=args.patience,
            batch=args.batch, imgsz=640, device=0, optimizer="SGD", lr0=0.01,
            cos_lr=False, seed=33, pose=50.0, kobj=10.0, degrees=10.0, scale=0.05,
            flipud=0.5, fliplr=0.0, mosaic=0.0, close_mosaic=10, workers=8,
            project=str(run_dir.parent), name=NAME, exist_ok=True, plots=False, verbose=True,
        )
    print("TRAIN_DONE", flush=True)

    # accuracy half of anchor B: eval best.pt @640 (same metric/size as anchor A) ---
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=640, device=0)
    mp, mp50 = float(metrics.pose.map), float(metrics.pose.map50)
    out = args.drive / "out" / "anchor_yolo11s_pose_640_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "name": "yolo11s_pose_gate_anchor", "weights": str(best), "imgsz": 640,
        "device": "cuda", "val_images": 140, "kpt_shape": [8, 3],
        "map": mp, "map50": mp50, "metric": "pose OKS mAP50-95 (ultralytics .val)",
        "trained_from": "yolo11s-pose.pt (COCO)", "epochs_cap": args.epochs,
        "patience": args.patience, "batch": args.batch, "backend": "colab-t4",
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2) + "\n")
    print(f"ANCHOR_MAP_WRITTEN {out} map={mp:.4f} map50={mp50:.4f}", flush=True)


if __name__ == "__main__":
    main()
