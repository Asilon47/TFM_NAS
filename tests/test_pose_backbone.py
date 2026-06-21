"""Tests for supernet/pose_backbone.py — OFA-MBv3 subnet as a P3/P4/P5 detection backbone.

The *tap math* (``stage_tap_indices``) is pure and runs under ``.venv``. The forward logic
is torch-only — ``PoseBackbone`` wraps any module exposing ``.first_conv`` + ``.blocks``, so a
small stub that mirrors the real channel/stride chain (but with cheap 1x1 convs) exercises the
tap-capture end to end *without* ``ofa`` or the pinned checkpoint. The real OFA integration is
the module's ``__main__`` smoke test, run under ``.venv-nas`` (mirrors ``supernet/sampler.py``).
"""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from catalog.ofa_mbv3 import STAGES, stage_in_c  # noqa: E402
from supernet.pose_backbone import PoseBackbone, stage_tap_indices  # noqa: E402

# --- stage_tap_indices: pure P3/P4/P5 block-index math -----------------------
# blocks[0] is OFA's fixed first block; stages 1/3/4 sit at strides 8/16/32, so the
# taps are the last-block indices of those stages (cumulative active depths).

def test_stage_tap_indices_max_corner():
    assert stage_tap_indices([4, 4, 4, 4, 4]) == (8, 16, 20)


def test_stage_tap_indices_min_corner():
    assert stage_tap_indices([2, 2, 2, 2, 2]) == (4, 8, 10)


def test_stage_tap_indices_mixed_depths():
    assert stage_tap_indices([2, 3, 4, 2, 3]) == (5, 11, 14)


def test_stage_tap_indices_p5_is_the_final_block():
    # P5 is always blocks[-1]: index == sum(d) == len(blocks) - 1, and taps strictly ascend.
    for d in ([4] * 5, [2, 3, 4, 2, 3], [3, 3, 3, 3, 3]):
        p3, p4, p5 = stage_tap_indices(d)
        assert p5 == sum(d)
        assert p3 < p4 < p5


def test_stage_tap_indices_rejects_wrong_length():
    with pytest.raises(ValueError, match="5"):
        stage_tap_indices([4, 4, 4])


# --- PoseBackbone forward: taps the right blocks, at the right shapes ---------

class _StubBackbone(nn.Module):
    """Torch-only stand-in for an OFA subnet: the real first_conv (3->16, /2) + per-stage
    entry/repeat blocks with the true channels/strides from ``catalog.ofa_mbv3.STAGES``, but
    1x1 convs instead of MBConvs. ``blocks[0]`` is the fixed first block (16->16, s1)."""

    def __init__(self, depths: list[int]) -> None:
        super().__init__()
        self.first_conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        blocks: list[nn.Module] = [nn.Conv2d(16, 16, 1)]
        for s, stage in enumerate(STAGES):
            out_c = stage["out_c"]
            for j in range(depths[s]):
                in_c = stage_in_c(s) if j == 0 else out_c
                stride = stage["stride"] if j == 0 else 1
                blocks.append(nn.Conv2d(in_c, out_c, 1, stride=stride))
        self.blocks = nn.ModuleList(blocks)


def test_pose_backbone_taps_three_scales_at_640():
    depths = [4, 4, 4, 4, 4]
    model = PoseBackbone(_StubBackbone(depths), depths).eval()
    with torch.no_grad():
        p3, p4, p5 = model(torch.randn(1, 3, 640, 640))
    assert tuple(p3.shape) == (1, 40, 80, 80)      # stride 8
    assert tuple(p4.shape) == (1, 112, 40, 40)     # stride 16
    assert tuple(p5.shape) == (1, 160, 20, 20)     # stride 32


def test_pose_backbone_shapes_are_depth_invariant():
    # OFA-w1.0 fixes stage widths/strides, so P3/P4/P5 shapes don't depend on the depths.
    want = ((1, 40, 80, 80), (1, 112, 40, 40), (1, 160, 20, 20))
    for depths in ([2, 2, 2, 2, 2], [2, 3, 4, 2, 3]):
        model = PoseBackbone(_StubBackbone(depths), depths).eval()
        with torch.no_grad():
            outs = model(torch.randn(1, 3, 640, 640))
        assert tuple(tuple(t.shape) for t in outs) == want


def test_pose_backbone_reports_constant_out_channels():
    model = PoseBackbone(_StubBackbone([3] * 5), [3] * 5)
    assert model.out_channels == (40, 112, 160)
