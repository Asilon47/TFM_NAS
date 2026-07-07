"""prune/yolo_tp_prep.py — the C2f-split rewrite (function- + param-preserving) and the
attention-block collection. Guards the 2026-07-07/08 Kaggle rc=137 OOM fix (see
prune/yolo_tp_prep.py docstring + procedure.md). Needs the ultralytics blocks → .venv-nas."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from prune.yolo_tp_prep import (  # noqa: E402
    ATTENTION_BLOCK_NAMES,
    C2fSplit,
    attention_modules,
    split_c2f_,
)


def _c3k2(in_c: int = 64, out_c: int = 64):
    from ultralytics.nn.modules.block import C3k2

    blk = C3k2(in_c, out_c, n=1, shortcut=True)
    blk.i, blk.f, blk.type = 4, -1, "C3k2"     # the routing attrs BaseModel forward reads
    return blk.eval()


def test_c2fsplit_function_preserving_eval() -> None:
    blk = _c3k2()
    split = C2fSplit(blk).eval()
    x = torch.randn(2, 64, 32, 32)
    with torch.no_grad():
        ref, got = blk(x), split(x)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max().item()


def test_c2fsplit_preserves_param_count() -> None:
    blk = _c3k2()
    n_before = sum(p.numel() for p in blk.parameters())
    # cv2 + m are adopted by reference; cv0+cv1 hold cv1's rows split in two → same total.
    split = C2fSplit(blk)
    n_after = sum(p.numel() for p in split.parameters())
    assert n_after == n_before


def test_c2fsplit_inherits_train_mode() -> None:
    """A fresh nn.Module defaults to train mode; the split must inherit the source's."""
    blk = _c3k2().train()
    assert C2fSplit(blk).training is True
    assert C2fSplit(blk.eval()).training is False


def test_split_c2f_replaces_and_keeps_routing() -> None:
    from torch import nn
    from ultralytics.nn.modules.block import C2f

    model = nn.Sequential(_c3k2(32, 32), nn.Conv2d(32, 8, 1))
    n = split_c2f_(model)
    assert n == 1
    assert isinstance(model[0], C2fSplit)
    assert not isinstance(model[0], C2f)
    assert (model[0].i, model[0].f, model[0].type) == (4, -1, "C3k2")


def test_attention_modules_detects_c2psa() -> None:
    from torch import nn
    from ultralytics.nn.modules.block import C2PSA

    model = nn.Sequential(C2PSA(128, 128), nn.Conv2d(128, 8, 1))  # c=64 → 1 head (c//64)
    found = attention_modules(model)
    assert len(found) == 1 and type(found[0]).__name__ in ATTENTION_BLOCK_NAMES
