"""Export a single block to FP32 ONNX. TRT handles FP16 cast at engine build time."""
from pathlib import Path

import torch

from catalog.blocks import build_block, count_params


def export_block(block: str, cfg: dict, input_shape, out_path: Path,
                 opset: int = 17) -> dict:
    model = build_block(block, cfg).eval()
    dummy = torch.randn(*input_shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path.as_posix(),
        input_names=["input"], output_names=["output"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,  # Static shapes — matches how NAS queries the LUT.
    )
    return {"params": count_params(model), "onnx_path": str(out_path)}
