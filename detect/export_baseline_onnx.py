"""Export the yolo11n-pose baseline to ONNX @640 for the Jetson latency anchor.

The deployed **baseline-to-beat** (D1). Its @640 latency on the Orin Nano sets the
CP 3.3 / D4 hard ceiling ``T_max``; this produces the ONNX that
``lut.orchestrate.bench_model`` turns into a TRT engine and times on-device.

ultralytics is imported lazily, so this module imports cleanly under ``.venv``; the
actual export needs ultralytics (``.venv-nas`` / Kaggle / Colab). Ship the resulting
``.onnx`` to the laptop and run::

    python -m detect.export_baseline_onnx --weights yolo11n-pose.pt --imgsz 640
    # then, on the laptop:
    python -m lut.orchestrate.bench_model --onnx yolo11n_pose_640.onnx --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export_baseline_onnx(weights: str, *, imgsz: int = 640, opset: int = 12,
                         out: Path | None = None) -> Path:
    """Export ``weights`` to a static-shape ONNX at ``imgsz`` and return its path.

    Static shapes (``dynamic=False``) + ``simplify`` match how the engine is built +
    benchmarked downstream (one fixed 640×640 input, batch 1) so the timed graph is
    exactly the deployed one. ultralytics writes the ONNX beside the weights; it is
    moved to ``out`` when given.
    """
    from ultralytics import YOLO  # lazy: keeps the module importable under .venv

    produced = Path(YOLO(weights).export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False))
    if out is None:
        return produced
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="yolo11n-pose.pt",
                    help="the deployed baseline weights (or a fine-tuned gate .pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("yolo11n_pose_640.onnx"))
    args = ap.parse_args(argv)

    path = export_baseline_onnx(args.weights, imgsz=args.imgsz, opset=args.opset, out=args.out)
    print(f"exported {args.weights} -> {path} (imgsz={args.imgsz}, static)")
    print("next (laptop): python -m lut.orchestrate.bench_model "
          f"--onnx {path} --imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
