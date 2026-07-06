"""CP 4.4 — net2net/graft_init.py: the adapter's first in_c output channels equal its input
exactly (the DoD), extras are exact replicas, and the ChannelAdapter wiring holds per scale."""
import pytest
import torch
from torch import nn

from detect.adapter import ChannelAdapter
from net2net.graft_init import apply_adapter_init, identity_embed_conv1x1_

torch.manual_seed(0)


def test_identity_embed_passthrough_and_replicas() -> None:
    conv = nn.Conv2d(5, 9, kernel_size=1)
    mapping = identity_embed_conv1x1_(conv, seed=0)
    x = torch.randn(2, 5, 4, 4)
    y = conv(x)
    assert torch.allclose(y[:, :5], x, atol=1e-7)              # DoD: exact passthrough
    for j in range(5, 9):
        assert torch.allclose(y[:, j], x[:, mapping[j]], atol=1e-7)   # replicas, not noise
    assert mapping[:5] == list(range(5))


def test_identity_embed_rejects_bad_convs() -> None:
    with pytest.raises(ValueError, match="1x1"):
        identity_embed_conv1x1_(nn.Conv2d(4, 8, kernel_size=3, padding=1))
    with pytest.raises(ValueError, match="expanding"):
        identity_embed_conv1x1_(nn.Conv2d(8, 4, kernel_size=1))
    with pytest.raises(ValueError, match="groups=1"):
        identity_embed_conv1x1_(nn.Conv2d(4, 8, kernel_size=1, groups=2))


def test_apply_adapter_init_on_the_real_graft_shape() -> None:
    adapter = ChannelAdapter((40, 112, 160), (64, 128, 256))
    mappings = apply_adapter_init(adapter, "net2wider", seed=0)
    assert [len(m) for m in mappings] == [64, 128, 256]
    feats = tuple(torch.randn(1, c, s, s) for c, s in ((40, 8), (112, 4), (160, 2)))
    out = adapter(feats)
    for f, o in zip(feats, out, strict=True):
        assert torch.allclose(o[:, : f.shape[1]], f, atol=1e-7)  # per-scale passthrough

    # deterministic: a second adapter under the same seed gets identical weights
    twin = ChannelAdapter((40, 112, 160), (64, 128, 256))
    apply_adapter_init(twin, "net2wider", seed=0)
    for a, b in zip(adapter.adapters, twin.adapters, strict=True):
        assert torch.equal(a.weight, b.weight)


def test_apply_adapter_init_rejects_unknown_kind_and_shape() -> None:
    adapter = ChannelAdapter((4,), (8,))
    with pytest.raises(ValueError, match="unknown adapter_init"):
        apply_adapter_init(adapter, "xavier")
    with pytest.raises(TypeError, match="ChannelAdapter-like"):
        apply_adapter_init(nn.Conv2d(4, 8, 1), "net2wider")
