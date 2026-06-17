"""Assemble a sampled subnet (and the fixed stem/head) for whole-net measurement.

This is the *measured* side of CP 2.2's additivity DoD. ``search.cost.cost`` SUMS
the per-block LUT latencies over ``arch_to_blocks``; the DoD checks that sum against
a measured whole-subnet latency. To keep the comparison fair, the measured net is
built from the **same block implementations the per-block LUT timed**
(``catalog.blocks.build_block``) — not the real OFA modules. Building the OFA subnet
instead would conflate implementation differences with the cross-block TensorRT
fusion the DoD is actually probing (peer-review R4.2).

``arch_to_blocks`` emits only the searchable MBConv backbone, and its block list has
verified channel + resolution continuity (16ch@112 -> ... -> 160ch@7), so the blocks
chain into a plain ``nn.Sequential``. The fixed stem (3->16) and head (final-expand /
feature-mix / classifier) are NOT in that list — they are the constant cost offset
(``search.cost``'s ``stem_head``), exported here separately so it can be calibrated
on-device.

Pure torch + ``catalog`` — runs under ``.venv`` (CPU), no ``ofa``/CUDA. Latency is
weight-value-independent, so random init (as the per-block sweep used) is correct.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from catalog.blocks import build_block, count_params
from catalog.contracts import ArchDict
from catalog.ofa_mbv3 import FIRST_BLOCK, STAGES, STEM_RES
from search.arch_to_blocks import arch_to_blocks

# Stem input == network input; the stride-2 stem halves it to STEM_RES (224 -> 112).
STEM_INPUT_SHAPE = (1, 3, STEM_RES * 2, STEM_RES * 2)            # (1, 3, 224, 224)
# Head input == last stage's output: out_c channels at res_in // stride resolution.
_LAST = STAGES[-1]
_HEAD_RES = _LAST["res_in"] // _LAST["stride"]
HEAD_INPUT_SHAPE = (1, _LAST["out_c"], _HEAD_RES, _HEAD_RES)     # (1, 160, 7, 7)

# OFA-MBv3-w1.0 head widths (catalog/ofa_mbv3.py table: [..., 160, 960, 1280]).
_HEAD_EXPAND_C = 960
_HEAD_FEATURE_C = 1280
_NUM_CLASSES = 1000


def assemble_subnet(arch_dict: ArchDict) -> nn.Module:
    """Chain the arch's MBConv backbone into one module (same blocks the LUT timed)."""
    return nn.Sequential(*(build_block(name, cfg)
                           for name, cfg, _ in arch_to_blocks(arch_dict)))


class _Stem(nn.Module):
    """OFA-MBv3 stem: 3->16, k3, stride 2, BN, h-swish (224 -> 112)."""

    def __init__(self) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(3, FIRST_BLOCK["in_c"], 3, 2, 1, bias=False),
            nn.BatchNorm2d(FIRST_BLOCK["in_c"]),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class _Head(nn.Module):
    """OFA-MBv3 head: final-expand 1x1 (+BN, h-swish) -> global avg pool ->
    feature-mix 1x1 (h-swish) -> linear classifier."""

    def __init__(self, in_c: int, num_classes: int) -> None:
        super().__init__()
        self.final_expand = nn.Sequential(
            nn.Conv2d(in_c, _HEAD_EXPAND_C, 1, 1, 0, bias=False),
            nn.BatchNorm2d(_HEAD_EXPAND_C),
            nn.Hardswish(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_mix = nn.Sequential(
            nn.Conv2d(_HEAD_EXPAND_C, _HEAD_FEATURE_C, 1, 1, 0, bias=False),
            nn.Hardswish(inplace=True),
        )
        self.classifier = nn.Linear(_HEAD_FEATURE_C, num_classes)

    def forward(self, x):
        x = self.pool(self.final_expand(x))
        x = self.feature_mix(x)
        return self.classifier(torch.flatten(x, 1))


def build_stem() -> nn.Module:
    return _Stem()


def build_head(num_classes: int = _NUM_CLASSES) -> nn.Module:
    return _Head(HEAD_INPUT_SHAPE[1], num_classes)


def _export(model: nn.Module, input_shape: tuple[int, ...], out_path: Path,
            opset: int) -> dict:
    """Export ``model`` to static-shape FP32 ONNX (mirrors lut.export.to_onnx)."""
    model = model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(*input_shape), out_path.as_posix(),
        input_names=["input"], output_names=["output"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,   # static shapes — the TRT engine is shape-specialized.
    )
    return {"params": count_params(model), "onnx_path": str(out_path)}


def export_subnet(arch_dict: ArchDict, out_path: Path, opset: int = 17) -> dict:
    """Export a sampled subnet's MBConv backbone as a single ONNX graph.

    Returns ``{onnx_path, params, n_blocks}``. The dummy input shape is the first
    block's ``input_shape`` (FIRST_BLOCK: 16 ch @ STEM_RES) — i.e. the backbone
    starts after the stem, exactly the span ``cost()`` sums.
    """
    blocks = arch_to_blocks(arch_dict)
    meta = _export(assemble_subnet(arch_dict), tuple(blocks[0][2]), Path(out_path), opset)
    return {**meta, "n_blocks": len(blocks)}


def export_stem(out_path: Path, opset: int = 17) -> dict:
    """Export the fixed stem (constant across all archs)."""
    return _export(build_stem(), STEM_INPUT_SHAPE, Path(out_path), opset)


def export_head(out_path: Path, opset: int = 17) -> dict:
    """Export the fixed head (constant across all archs)."""
    return _export(build_head(), HEAD_INPUT_SHAPE, Path(out_path), opset)
