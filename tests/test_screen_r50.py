"""Pure-logic guards for the OFA-ResNet50 latency screen (expand/screen_r50.py).

The OFA wiring needs .venv-nas and is exercised by the module's own run; these tests only
pin the constants and the corner-dispatch error path, so they stay green in .venv/CI (the
module imports there — OFAResNets is imported lazily inside main()).
"""
import pytest

from expand import screen_r50


def test_tap_sizes_are_strides_8_16_32_at_640():
    # 640 / {8, 16, 32} = 80 / 40 / 20 — the P3/P4/P5 spatial sizes.
    assert screen_r50.TAP_SIZES == (80, 40, 20)
    assert list(screen_r50.TAP_SIZES) == sorted(screen_r50.TAP_SIZES, reverse=True)


def test_r50_space_has_the_ofa_resnet50_knobs():
    space = screen_r50.R50_SPACE
    assert set(space) == {"depth_list", "expand_ratio_list", "width_mult_list"}
    assert space["depth_list"] == [0, 1, 2]
    assert min(space["width_mult_list"]) == 0.65  # w=0 index → narrowest (the min corner)


def test_build_backbone_rejects_unknown_corner():
    # The dispatch raises before touching the net, so a sentinel is enough.
    with pytest.raises(ValueError, match="min|mid|max"):
        screen_r50.build_backbone(net=object(), which="bogus")
