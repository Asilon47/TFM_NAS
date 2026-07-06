"""CP 4.2 — net2net/deeper.py: identity inserts leave the forward exactly unchanged (the DoD)."""
import pytest
import torch
from torch import nn

from net2net.deeper import identity_conv2d, identity_linear, inserted

torch.manual_seed(0)


def test_identity_conv_is_exact_identity() -> None:
    x = torch.randn(2, 6, 10, 10)
    for k in (1, 3, 5):
        assert torch.equal(identity_conv2d(6, k)(x), x) or torch.allclose(
            identity_conv2d(6, k)(x), x, atol=1e-7)


def test_identity_conv_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd kernel"):
        identity_conv2d(4, kernel_size=2)


def test_identity_linear_is_exact_identity() -> None:
    x = torch.randn(3, 7)
    assert torch.allclose(identity_linear(7)(x), x, atol=1e-7)


def test_deepen_conv_net_forward_unchanged() -> None:
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))
    x = torch.randn(2, 3, 12, 12)
    ref = net(x)
    deeper = inserted(net, 2, identity_conv2d(8))            # plain identity insert
    assert torch.allclose(deeper(x), ref, atol=1e-6)
    assert len(deeper) == 4 and len(net) == 3                # original untouched
    # conv+ReLU block insert after an existing ReLU — idempotency makes it exact
    deeper_block = inserted(net, 2, identity_conv2d(8), nn.ReLU())
    assert torch.allclose(deeper_block(x), ref, atol=1e-6)


def test_deepen_linear_net_forward_unchanged() -> None:
    net = nn.Sequential(nn.Linear(5, 9), nn.ReLU(), nn.Linear(9, 2))
    x = torch.randn(4, 5)
    ref = net(x)
    deeper = inserted(net, 2, identity_linear(9), nn.ReLU())
    assert torch.allclose(deeper(x), ref, atol=1e-6)
