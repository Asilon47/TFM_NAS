"""Gate-train a COCO-seeded yolo11*-pose at a chosen resolution — the baseline arm, at imgsz R.

The MCU resolution screen (`mcu/res_screen.py`, 2026-07-16) compared a 640-trained graft and a
640-trained baseline *evaluated* at 160, and found the graft Pareto-dominated. Both arms were
lower bounds, so the screen could not settle whether the graft's extra fragility is architectural
or an artifact. This produces the baseline's half of the answer: yolo11n-pose actually **trained**
at the target resolution.

**The recipe is copied from the deployed baseline's own run**, not chosen here:
``runs/pose/experiments/gate_baseline/args.yaml`` is what produced the 0.8774 anchor, so anything
this script does differently would silently become the finding. Only ``imgsz`` moves. The two
settings that matter most for a resolution question, and which the graft's training path does NOT
have (`grep multi_scale prune/ eval/ detect/` is empty — the bare-AdamW loop never touches
Ultralytics' trainer, where multi_scale lives):

  * ``multi_scale=True``  — Ultralytics jitters the input 0.5-1.5x per batch, so a 640 run saw
    ~320-960 px. This is why the baseline degrades gracefully off-resolution and the graft does
    not, and it is a **recipe** confound on the screen's "2x more fragile" reading.
  * ``epochs=2000, patience=300`` — vs the graft's flat 100. Early stopping does the real work.

That confound is NOT removed here, deliberately. Each family keeps its native recipe, so the
comparison at 160 carries the *same* confound as the comparison at 640 (0.7625 vs 0.8774, gap
-0.115) — which is what lets the **change in gap** be read as a resolution effect even though the
absolutes stay recipe-tainted. Recipe parity itself is Phase 7, not this.

Run on the AGX (all training, 2026-07-15 compute decision), as a module::

    python -m detect.train_baseline --imgsz 160 --data-yaml /data/dataset/dataset.yaml \\
        --out-dir /data/out/baseline_r160
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Copied verbatim from runs/pose/experiments/gate_baseline/args.yaml (the run that produced the
# 0.8774 anchor). Do not "improve" these — a divergence here is indistinguishable from a finding.
GATE_RECIPE: dict = {
    "epochs": 2000, "patience": 300, "batch": 4, "seed": 33, "deterministic": True,
    "pretrained": True, "multi_scale": True, "cos_lr": False, "close_mosaic": 10,
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 50.0, "kobj": 10.0, "nbs": 64,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 10.0, "translate": 0.1,
    "scale": 0.05, "shear": 0.0, "perspective": 0.0, "flipud": 0.5, "fliplr": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.4, "workers": 8,
}
ANCHOR_640_MAP = 0.8774  # the deployed baseline's recorded pose mAP50-95 @640 (models/README.md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="COCO seed; ultralytics auto-downloads and re-heads to the yaml's "
                         "nc=1 / kpt_shape=(8,3)")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="durable dir (AGX: /data/out/...)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's 2000")
    ap.add_argument("--batch", type=int, default=None, help="override the recipe's 4")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    from detect.evaluate import resolve_data_yaml

    data = str(resolve_data_yaml(args.data_yaml))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.model).stem}_r{args.imgsz}"
    run_dir = args.out_dir / "runs" / tag

    recipe = {**GATE_RECIPE, "imgsz": args.imgsz, "device": args.device}
    if args.epochs is not None:
        recipe["epochs"] = args.epochs
    if args.batch is not None:
        recipe["batch"] = args.batch

    last = run_dir / "weights" / "last.pt"
    if last.exists():   # AGX container restarts / --resume land here
        print(f"[resume] {last}", flush=True)
        YOLO(str(last)).train(resume=True)
    else:
        print(f"[fresh] {args.model} from COCO weights @ imgsz={args.imgsz}", flush=True)
        print(f"[recipe] gate_baseline/args.yaml verbatim, imgsz -> {args.imgsz}; "
              f"multi_scale={recipe['multi_scale']} epochs={recipe['epochs']}", flush=True)
        YOLO(args.model).train(data=data, project=str(run_dir.parent), name=tag,
                               exist_ok=True, plots=False, verbose=True, **recipe)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no {best}")
    weights = args.out_dir / f"baseline_{tag}.pt"
    shutil.copy(best, weights)

    metrics = YOLO(str(best)).val(data=data, imgsz=args.imgsz, device=args.device)
    mp, mp50 = float(metrics.pose.map), float(metrics.pose.map50)

    out = args.out_dir / f"baseline_{tag}.json"
    out.write_text(json.dumps({
        "note": "Baseline arm trained AT imgsz — the matched half of the MCU resolution question. "
                "Recipe = gate_baseline/args.yaml verbatim (incl. multi_scale + 2000ep/patience "
                "300); the graft arm keeps its own bare-AdamW recipe, so absolutes stay recipe-"
                "confounded and only the CHANGE in gap vs @640 reads as a resolution effect.",
        "model": args.model, "tag": tag, "imgsz": args.imgsz, "weights": str(weights),
        "map": mp, "map50": mp50, "metric": "pose OKS mAP (ultralytics .val)",
        "anchor_640_map": ANCHOR_640_MAP, "delta_vs_640_anchor": mp - ANCHOR_640_MAP,
        "recipe": recipe,
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2))
    print(f"[done] {tag}: map={mp:.4f} map50={mp50:.4f} "
          f"(vs the @640 anchor {ANCHOR_640_MAP}: {mp - ANCHOR_640_MAP:+.4f}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
