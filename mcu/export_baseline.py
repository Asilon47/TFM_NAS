"""Export the yolo11n-pose baseline for the GAP8 probes — the symmetric twin of
``detect.export_grafted_onnx --raw-head`` (CP 10.1).

The baseline must be exported *the same way* as the graft or the comparison is not about
architecture. That means: the gate donor's head config (nc=1, 8 kpts — the stock COCO
checkpoint's nc=80/17-kpt head would inflate the baseline's head cost), static 1x3xHxW,
opset 12, and **raw per-stride head maps** rather than the decoded tensor.

Why raw maps: the decode postprocess (DFL softmax, anchor concat, strided reshapes) breaks
nntool's ``adjust_order`` for BOTH families identically (CP 10.1 probe, 2026-07-15) and does
not belong in a tiled MCU graph — on GAP8 the decode runs in C on the fabric controller while
AutoTiler owns the conv work. ``head.training = True`` selects the raw-map branch per-module
while BN stays in eval (the trick ``distill/kd_loss.load_frozen_teacher`` already relies on).

Run under ``.venv-nas`` (needs ultralytics), as a module so the repo root is importable::

    python -m mcu.export_baseline --imgsz 224 --out models/res224/yolo11n_pose_224_raw.onnx
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DONOR = REPO / "runs/pose/experiments/gate_baseline/weights/best.pt"


def export_baseline(donor: Path, out: Path, *, imgsz: int = 224, opset: int = 12,
                    raw_head: bool = True) -> tuple[Path, dict]:
    """Export the donor's model to ONNX; returns ``(path, meta)`` and writes a meta sidecar."""
    import torch
    from ultralytics import YOLO

    # Same TorchScript-path exporter the graft/LUT graphs use: torch >= 2.9 defaults to the
    # dynamo exporter, which needs onnxscript (absent by design) and emits a different graph.
    from detect.export_grafted_onnx import RAW_HEAD_OUTPUTS, _legacy_export

    model = YOLO(str(donor)).model.float().eval()
    head = model.model[-1]
    if raw_head:
        head.training = True
        output_names = RAW_HEAD_OUTPUTS
    else:
        head.export = True
        head.format = "onnx"
        output_names = ["output0"]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _legacy_export(model, torch.randn(1, 3, imgsz, imgsz), out, opset=opset,
                       output_names=output_names, preserve_training=raw_head)

    meta = {
        "donor": str(donor),
        "nc": int(getattr(head, "nc", -1)),
        "kpt_shape": list(getattr(head, "kpt_shape", [])),
        "raw_head": raw_head,
        "imgsz": imgsz,
        "opset": opset,
        "params": sum(p.numel() for p in model.parameters()),
        "onnx": out.name,
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donor", type=Path, default=DEFAULT_DONOR,
                    help=f"gate-trained yolo11n-pose checkpoint (default: {DEFAULT_DONOR})")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--decoded", action="store_true",
                    help="export the decoded deploy tensor instead of raw head maps")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    path, meta = export_baseline(args.donor, args.out, imgsz=args.imgsz, opset=args.opset,
                                 raw_head=not args.decoded)
    print(f"exported yolo11n-pose (nc={meta['nc']}, kpt={meta['kpt_shape']}, "
          f"raw_head={meta['raw_head']}, params {meta['params']:,}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
