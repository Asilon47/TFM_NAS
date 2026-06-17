"""Tests for search/export_subnet.py — whole-subnet & stem/head ONNX export.

Runs in .venv (CPU torch). The forward-shape tests are the correctness crux:
they prove the catalog-block chain assembled from ``arch_to_blocks`` is
dimensionally valid end-to-end (channel + resolution continuity), which is the
precondition for measuring a sampled subnet as a *single* TRT engine. The ONNX
smokes confirm the export is loadable; ``onnx`` is optional so they skip if absent.
"""
import random

import pytest

torch = pytest.importorskip("torch")

from search.arch_to_blocks import arch_to_blocks, random_arch_dict  # noqa: E402
from search.export_subnet import (  # noqa: E402
    HEAD_INPUT_SHAPE,
    STEM_INPUT_SHAPE,
    assemble_subnet,
    build_head,
    build_stem,
    export_head,
    export_stem,
    export_subnet,
)


def test_assemble_subnet_forwards_backbone_shape():
    arch = random_arch_dict(random.Random(0))
    in_shape = tuple(arch_to_blocks(arch)[0][2])      # FIRST_BLOCK in: (1,16,112,112)
    model = assemble_subnet(arch).eval()
    with torch.no_grad():
        out = model(torch.randn(*in_shape))
    assert in_shape == (1, 16, 112, 112)
    assert tuple(out.shape) == (1, 160, 7, 7)         # last stage: 160 ch @ res 7


def test_assemble_subnet_spans_all_depths():
    # Both depth extremes must assemble + forward without a shape mismatch.
    for d in (2, 4):
        arch = {"ks": [3] * 20, "e": [3] * 20, "d": [d] * 5}
        model = assemble_subnet(arch).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 16, 112, 112))
        assert tuple(out.shape) == (1, 160, 7, 7)


def test_build_stem_forward_shape():
    with torch.no_grad():
        out = build_stem().eval()(torch.randn(*STEM_INPUT_SHAPE))
    assert STEM_INPUT_SHAPE == (1, 3, 224, 224)
    assert tuple(out.shape) == (1, 16, 112, 112)


def test_build_head_forward_shape():
    with torch.no_grad():
        out = build_head().eval()(torch.randn(*HEAD_INPUT_SHAPE))
    assert HEAD_INPUT_SHAPE == (1, 160, 7, 7)
    assert tuple(out.shape) == (1, 1000)


def test_export_subnet_writes_loadable_onnx(tmp_path):
    arch = random_arch_dict(random.Random(1))
    out = tmp_path / "subnet.onnx"
    meta = export_subnet(arch, out)
    assert out.exists() and out.stat().st_size > 0
    assert meta["n_blocks"] == len(arch_to_blocks(arch))
    assert meta["params"] > 0
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(onnx.load(str(out)))


def test_export_stem_and_head_write_loadable_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    for fn, name in ((export_stem, "stem.onnx"), (export_head, "head.onnx")):
        p = tmp_path / name
        meta = fn(p)
        assert p.exists() and p.stat().st_size > 0
        assert meta["params"] > 0
        onnx.checker.check_model(onnx.load(str(p)))
