#!/usr/bin/env python3
"""Remote entry — gate-train a yolo11-family pose model as a KD teacher (winner-v2-OFA Track 2t).

The KD contract (``distill/kd_loss``) needs the teacher to share the yolo11 head structure, so
a bigger teacher is just a bigger yolo11*-pose gate-trained from COCO weights — this is the
size-generalized sibling of ``anchor_b.py`` (which hardwired yolo11s). ``--model yolo11x-pose.pt``
produces the T-C teacher; ``yolo11m-pose.pt`` a middle rung. Output ``best.pt`` on the durable
out-dir is then passed as ``--teacher-pt`` to ``run_prune_graft`` for the wave-2 teacher A/B.

Why gate-train, not stock: stock COCO pose = 17 human keypoints; the gate task is nc=1 / 8
keypoints, so the head must be re-headed + trained on the gate set (ultralytics does the
re-head automatically from the data yaml's kpt_shape). Resumable across free-tier recycles via
the run dir on the durable out-dir.

    python colab/train_teacher.py --model yolo11x-pose.pt --out-dir /content/tfm_out
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate-train a yolo11-family pose KD teacher.")
    ap.add_argument("--model", default="yolo11x-pose.pt",
                    help="COCO seed (yolo11{n,s,m,l,x}-pose.pt) — ultralytics auto-downloads")
    ap.add_argument("--out-dir", type=Path, required=True, help="durable run dir (Drive/studio)")
    ap.add_argument("--secrets-root", type=Path, default=Path(C.DRIVE_DEFAULT))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8,
                    help="x is ~57M params — a T4 wants a small batch")
    a = ap.parse_args()

    data_root = a.data_root or (Path("/content/kagdata") if Path("/content").exists()
                                else Path.home() / "kagdata")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    user = C.ensure_kaggle_credentials(a.secrets_root)
    C.pin_torch_and_install("'ultralytics>=8.3'")
    staged = C.stage_kaggle_dataset(user, data_root)
    yaml_src = C.find(staged, "dataset.yaml")
    if yaml_src is None:
        raise SystemExit(f"dataset.yaml not found under {staged}")
    spec = yaml.safe_load(yaml_src.read_text())
    spec["path"] = str(yaml_src.parent)
    data_yaml = Path(tempfile.mkstemp(suffix="_dataset.yaml", text=True)[1])
    data_yaml.write_text(yaml.safe_dump(spec))

    from ultralytics import YOLO

    name = f"gate_teacher_{Path(a.model).stem}"
    run_dir = a.out_dir / "runs" / "pose" / name
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        ckpt_data = Path(yaml.safe_load((run_dir / "args.yaml").read_text())["data"])
        if not ckpt_data.exists():
            ckpt_data.write_text(yaml.safe_dump(spec))
        print(f"[resume] {last}", flush=True)
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        print(f"[fresh] {a.model} from COCO weights", flush=True)
        model = YOLO(a.model)
        model.train(
            data=str(data_yaml), epochs=a.epochs, patience=a.patience, batch=a.batch,
            imgsz=640, device=0, optimizer="SGD", lr0=0.01, cos_lr=False, seed=33,
            pose=50.0, kobj=10.0, degrees=10.0, scale=0.05, flipud=0.5, fliplr=0.0,
            mosaic=0.0, close_mosaic=10, workers=8, project=str(run_dir.parent), name=name,
            exist_ok=True, plots=False, verbose=True)
    print("TRAIN_DONE", flush=True)

    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=640, device=0)
    mp, mp50 = float(metrics.pose.map), float(metrics.pose.map50)
    out = a.out_dir / f"teacher_{Path(a.model).stem}_map.json"
    out.write_text(json.dumps({
        "teacher": name, "weights": str(best), "seed_model": a.model, "imgsz": 640,
        "map": mp, "map50": mp50, "epochs_cap": a.epochs, "batch": a.batch,
        "metric": "pose OKS mAP50-95 (ultralytics .val)",
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2) + "\n")
    print(f"TEACHER_MAP_WRITTEN {out} map={mp:.4f} map50={mp50:.4f}  best={best}", flush=True)


if __name__ == "__main__":
    main()
