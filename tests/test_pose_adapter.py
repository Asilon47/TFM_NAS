"""Tests for detect/adapter.py — 1x1 channel adapters between the OFA backbone and the head.

The backbone emits P3/P4/P5 at the fixed channels (40, 112, 160); a cloned YOLO11n-pose head
expects its own neck-output channels. ``ChannelAdapter`` is the 1x1-conv glue. It is torch-only
(no ultralytics), so it is unit-tested under ``.venv`` independently of the head graft.
"""
import pytest

torch = pytest.importorskip("torch")

from detect.adapter import ChannelAdapter  # noqa: E402


def test_channel_adapter_remaps_each_scale_to_target_channels():
    adapter = ChannelAdapter((40, 112, 160), (64, 128, 256)).eval()
    feats = (torch.randn(1, 40, 80, 80), torch.randn(1, 112, 40, 40), torch.randn(1, 160, 20, 20))
    with torch.no_grad():
        out = adapter(feats)
    assert tuple(tuple(t.shape) for t in out) == (
        (1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20))


def test_channel_adapter_preserves_spatial_dims():
    # 1x1 convs must not resample — only channels change, H/W pass through untouched.
    adapter = ChannelAdapter((40, 112, 160), (64, 128, 256)).eval()
    feats = (torch.randn(1, 40, 17, 17), torch.randn(1, 112, 9, 9), torch.randn(1, 160, 5, 5))
    with torch.no_grad():
        out = adapter(feats)
    assert [tuple(t.shape[2:]) for t in out] == [(17, 17), (9, 9), (5, 5)]


def test_channel_adapter_reports_out_channels():
    adapter = ChannelAdapter((40, 112, 160), (64, 128, 256))
    assert adapter.out_channels == (64, 128, 256)
