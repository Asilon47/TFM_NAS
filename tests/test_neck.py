"""CP 5.1 — detect/neck.py: identity at init (the DoD), ReZero grad dynamics, dispatch."""
import pytest
import torch
from torch import nn

from detect.neck import NECK_KINDS, ZeroGatedTopDownNeck, build_neck

torch.manual_seed(0)

CH = (64, 128, 256)


def _feats(imgsz: int = 64) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(1, c, imgsz // s, imgsz // s)
                 for c, s in zip(CH, (8, 16, 32), strict=True))


@pytest.mark.parametrize("kind", ["topdown", "pan"])
def test_identity_at_init_exactly(kind: str) -> None:
    neck = build_neck(kind, CH)
    assert neck is not None
    feats = _feats()
    out = neck(feats)
    for f, o in zip(feats, out, strict=True):
        assert torch.equal(o, f)                       # gates are 0 → EXACT identity (DoD)


def test_open_gates_change_outputs_and_preserve_shapes() -> None:
    neck = ZeroGatedTopDownNeck(CH, bottom_up=True)
    with torch.no_grad():
        for g in (neck.g54, neck.g43, neck.g34, neck.g45):
            g.fill_(1.0)
    feats = _feats()
    out = neck(feats)
    for f, o in zip(feats, out, strict=True):
        assert o.shape == f.shape
        assert not torch.equal(o, f)                   # fusion is actually flowing


def test_rezero_grad_dynamics_at_init() -> None:
    """At init the GATES get gradient, the projections do not — once a gate moves, they do."""
    neck = ZeroGatedTopDownNeck(CH)
    out = neck(_feats())
    sum(o.sum() for o in out).backward()
    assert neck.g54.grad is not None and float(neck.g54.grad.abs()) > 0
    assert neck.g43.grad is not None and float(neck.g43.grad.abs()) > 0
    assert neck.lat54.weight.grad is not None
    assert float(neck.lat54.weight.grad.abs().sum()) == 0.0    # gated off → no signal yet

    neck.zero_grad()
    with torch.no_grad():
        neck.g54.fill_(0.5)
    out = neck(_feats())
    sum(o.sum() for o in out).backward()
    assert float(neck.lat54.weight.grad.abs().sum()) > 0       # gate open → projection trains


def test_gate_values_reporting() -> None:
    td = ZeroGatedTopDownNeck(CH)
    assert td.gate_values() == {"g54": 0.0, "g43": 0.0}
    pan = ZeroGatedTopDownNeck(CH, bottom_up=True)
    assert set(pan.gate_values()) == {"g54", "g43", "g34", "g45"}


def test_build_neck_dispatch() -> None:
    assert build_neck(None) is None
    td = build_neck("topdown")
    pan = build_neck("pan")
    assert isinstance(td, ZeroGatedTopDownNeck) and not td.bottom_up
    assert isinstance(pan, ZeroGatedTopDownNeck) and pan.bottom_up
    assert isinstance(pan.down34, nn.Conv2d) and pan.down34.stride == (2, 2)
    with pytest.raises(ValueError, match="unknown neck kind"):
        build_neck("bifpn")
    with pytest.raises(ValueError, match="3 scales"):
        ZeroGatedTopDownNeck((64, 128))
    assert set(NECK_KINDS) == {"topdown", "pan"}
