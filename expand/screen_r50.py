"""OFA-ResNet50 latency screen — the D3 "is a different supernet worth it?" decision gate.

Stage 0 proved the OFA-MBv3 graft is device-dominated on the Nano (depthwise ≈ 0.30 vs the
dense baseline's ≈ 0.60 TFLOP/s effective), so every grafted subnet is 17–18 ms e2e against the
12.75 ms yolo11n baseline. The only *pretrained + samplable + dense* alternative supernet is
OFA-ResNet50; a full re-search is worth building only if even its SMALLEST subnet backbone fits
the honest latency budget at 640. This exports the min / median / max subnet BACKBONES
(classifier stripped, P3/P4/P5 taps) to ONNX so ``lut.orchestrate.bench_model`` can measure
them one-at-a-time on the Nano (mode 0, clocks locked).

Budget algebra: baseline = 12.75 ms fp32 e2e; the graft's pose adapter+head offset is 3.84 ms
(``data/pose_stem_head_offset.json``) — a LOWER bound on R50's offset, whose adapter is ~10× the
channels — so a backbone must land under ≈ 12.75 − 3.84 = 8.9 ms fp32 to leave any room (the
Phase-3b honest backbone ceiling was 7.16 ms). TRT fp32 latency is weight-value-independent, so
random init is used — no pretrained checkpoint needed.

Run (``.venv-nas``; CPU export is fine)::

    python -m expand.screen_r50 --out-dir models/screen_r50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from catalog.flops import count_flops

# Standard ofa_resnet50 design space (matches ofa.model_zoo 'ofa_resnet50').
R50_SPACE: dict = dict(depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
                       width_mult_list=[0.65, 0.8, 1.0])

# At 640 input the ResNet stem downsamples by 4 (conv s2 + maxpool s2 → 160²); the four stages
# then sit at strides 4/8/16/32 → spatial 160/80/40/20. A detection/pose head reads P3/P4/P5
# (strides 8/16/32), so these are the tap sizes.
TAP_SIZES: tuple[int, int, int] = (80, 40, 20)


class R50Backbone(nn.Module):
    """An OFA-ResNet50 subnet as a P3/P4/P5 detection backbone (classifier dropped).

    Runs ``input_stem → max_pooling → blocks`` and returns the LAST feature map at each of
    strides 8/16/32 (the standard FPN choice — deepest block per stride). Unlike OFA-MBv3-w1.0
    (fixed widths), R50 tap channels vary with the elastic width multiplier, so they are
    reported per subnet rather than assumed invariant.
    """

    def __init__(self, subnet: Any) -> None:  # OFA static ResNets — duck-typed
        super().__init__()
        self.input_stem = subnet.input_stem
        self.max_pooling = subnet.max_pooling
        self.blocks = subnet.blocks

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)
        taps: dict[int, Tensor] = {}
        for b in self.blocks:
            x = b(x)
            s = int(x.shape[-1])
            if s in TAP_SIZES:
                taps[s] = x  # overwrite → keep the deepest block at each stride
        missing = [s for s in TAP_SIZES if s not in taps]
        if missing:
            raise RuntimeError(f"subnet never produced tap sizes {missing} at 640")
        return taps[80], taps[40], taps[20]


def build_backbone(net: Any, which: str) -> R50Backbone:
    """Set the {min,mid,max} corner of the R50 space and return its backbone."""
    if which == "min":
        net.set_active_subnet(d=min(net.depth_list), e=min(net.expand_ratio_list), w=0)
    elif which == "max":
        net.set_max_net()
    elif which == "mid":
        net.set_active_subnet(
            d=sorted(net.depth_list)[len(net.depth_list) // 2],
            e=sorted(net.expand_ratio_list)[len(net.expand_ratio_list) // 2],
            w=len(net.width_mult_list) // 2)
    else:
        raise ValueError(f"unknown corner {which!r} (expected min|mid|max)")
    return R50Backbone(net.get_active_subnet(preserve_weight=False).eval()).eval()


def export_one(net: Any, which: str, out_dir: Path, imgsz: int = 640) -> dict:
    """Build → forward-check → ONNX-export one corner; return its meta row."""
    bb = build_backbone(net, which)
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        feats = bb(x)
    tap_shapes = [list(t.shape) for t in feats]
    flops = count_flops(bb, (1, 3, imgsz, imgsz))
    params = sum(p.numel() for p in bb.parameters())

    onnx = out_dir / f"ofa_r50_{which}_bb_{imgsz}.onnx"
    # dynamo=False forces the legacy TorchScript exporter (torch 2.11 defaults to the dynamo
    # path, which needs onnxscript — not in .venv-nas; see the plan's "force legacy exporter").
    torch.onnx.export(bb, x, str(onnx), opset_version=17, do_constant_folding=True,
                      input_names=["images"], output_names=["p3", "p4", "p5"], dynamo=False)
    try:  # smoke-load — the TRT build on the Jetson is the real validation
        import onnxruntime as ort  # type: ignore[import-untyped]
        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        sess.run(None, {"images": x.numpy()})
        ort_ok = True
    except ImportError:
        ort_ok = None  # onnxruntime not installed — export still valid

    row = {
        "which": which, "imgsz": imgsz, "params": int(params), "flops": int(flops),
        "gflops": round(flops / 1e9, 2), "n_blocks": len(bb.blocks),
        "tap_channels": [s[1] for s in tap_shapes], "tap_shapes": tap_shapes,
        "onnxruntime_ok": ort_ok, "onnx": str(onnx),
    }
    (out_dir / f"ofa_r50_{which}.meta.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main(argv: list[str] | None = None) -> int:
    from ofa.imagenet_classification.elastic_nn.networks import OFAResNets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("models/screen_r50"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--which", default="min,mid,max", help="comma list of corners to export")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    net = OFAResNets(n_classes=1000, dropout_rate=0, **R50_SPACE).eval()
    rows = [export_one(net, w.strip(), a.out_dir, a.imgsz) for w in a.which.split(",")]
    (a.out_dir / "screen_r50.json").write_text(
        json.dumps({"space": R50_SPACE, "imgsz": a.imgsz, "rows": rows}, indent=2) + "\n")
    print(f"{'corner':6s} {'params':>12s} {'GFLOPs':>8s}  blocks  tap_channels")
    for r in rows:
        print(f"{r['which']:6s} {r['params']:>12,} {r['gflops']:>8.1f}  "
              f"{r['n_blocks']:>6d}  {r['tap_channels']} -> {Path(r['onnx']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
