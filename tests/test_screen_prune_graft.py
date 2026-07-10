"""Pure-logic guards for the prune-the-graft latency screen (prune/screen_prune_graft.py).

The OFA + torch-pruning wiring needs .venv-nas and is exercised by the module's own run; these
tests pin the ladder + the optimistic-latency prior so they stay green in .venv/CI (the module's
heavy imports all live inside main()).
"""
from prune import screen_prune_graft as s


def test_ratios_are_fractions_ascending():
    assert all(0.0 < r < 1.0 for r in s.RATIOS)
    assert list(s.RATIOS) == sorted(s.RATIOS)


def test_optimistic_backbone_ms_is_linear_and_clamped():
    base = s.BACKBONE_MS_FP32
    assert s.optimistic_backbone_ms(base, 0.0) == base        # no prune -> unchanged
    assert s.optimistic_backbone_ms(base, 0.5) == base * 0.5  # linear in surviving channels
    assert s.optimistic_backbone_ms(base, 1.0) == 0.0
    assert s.optimistic_backbone_ms(base, 1.5) == 0.0         # clamped, never negative


def test_onnx_name_encodes_ratio_and_resolution():
    assert s.onnx_name(0.4, 640) == "graft_prune_r40_e2e_640.onnx"
    assert s.onnx_name(0.2, 512) == "graft_prune_r20_e2e_512.onnx"
