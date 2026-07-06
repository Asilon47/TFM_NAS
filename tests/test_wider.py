"""CP 4.1 — net2net/wider.py: Net2Wider is function-preserving to 1e-5 (the DoD)."""
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from net2net.wider import widen_conv2d, widen_linear, widen_mapping

torch.manual_seed(0)


def test_mapping_identity_prefix_deterministic() -> None:
    m = widen_mapping(8, 13, random.Random(0))
    assert len(m) == 13
    assert m[:8] == list(range(8))                       # identity prefix
    assert all(0 <= src < 8 for src in m[8:])            # extras replicate real channels
    assert m == widen_mapping(8, 13, random.Random(0))   # deterministic under the seed
    assert widen_mapping(8, 8, random.Random(0)) == list(range(8))
    with pytest.raises(ValueError):
        widen_mapping(8, 7, random.Random(0))


def test_widen_conv_preserves_function_through_relu() -> None:
    conv, nxt = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 5, 3, padding=1)
    x = torch.randn(2, 3, 16, 16)
    ref = nxt(F.relu(conv(x)))
    new_conv, new_next, new_bn, mapping = widen_conv2d(conv, nxt, 13)
    assert new_bn is None and len(mapping) == 13
    assert new_conv.out_channels == 13 and new_next.in_channels == 13
    out = new_next(F.relu(new_conv(x)))
    assert torch.allclose(out, ref, atol=1e-5)
    assert conv.out_channels == 8 and nxt.in_channels == 8   # originals untouched


def test_widen_conv_with_bn_preserves_function() -> None:
    conv, bn, nxt = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1)
    bn.train()
    for _ in range(3):                                    # give BN non-trivial running stats
        bn(conv(torch.randn(4, 3, 8, 8)))
    bn.eval()
    x = torch.randn(2, 3, 8, 8)
    ref = nxt(F.hardswish(bn(conv(x))))
    new_conv, new_next, new_bn, _ = widen_conv2d(conv, nxt, 12, bn=bn)
    assert new_bn is not None
    new_bn.eval()
    out = new_next(F.hardswish(new_bn(new_conv(x))))
    assert torch.allclose(out, ref, atol=1e-5)


def test_widen_equal_width_is_exact_copy() -> None:
    conv, nxt = nn.Conv2d(2, 4, 1), nn.Conv2d(4, 3, 1)
    new_conv, new_next, _, mapping = widen_conv2d(conv, nxt, 4)
    assert mapping == [0, 1, 2, 3]
    assert torch.equal(new_conv.weight, conv.weight)
    assert torch.equal(new_next.weight, nxt.weight)


def test_widen_linear_preserves_function() -> None:
    lin, nxt = nn.Linear(4, 6), nn.Linear(6, 3)
    x = torch.randn(5, 4)
    ref = nxt(F.relu(lin(x)))
    new_lin, new_next, mapping = widen_linear(lin, nxt, 10)
    assert len(mapping) == 10
    out = new_next(F.relu(new_lin(x)))
    assert torch.allclose(out, ref, atol=1e-5)


def test_widen_conv_rejects_bad_pairs() -> None:
    with pytest.raises(ValueError, match="groups=1"):
        widen_conv2d(nn.Conv2d(4, 4, 3, groups=2, padding=1), nn.Conv2d(4, 2, 1), 6)
    with pytest.raises(ValueError, match="producer/consumer"):
        widen_conv2d(nn.Conv2d(3, 8, 1), nn.Conv2d(6, 2, 1), 10)
    with pytest.raises(ValueError, match="bn tracks"):
        widen_conv2d(nn.Conv2d(3, 8, 1), nn.Conv2d(8, 2, 1), 10, bn=nn.BatchNorm2d(6))
