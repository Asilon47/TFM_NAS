"""CP 6.4 HALP-lite — prune/allocate.py knapsack math (pure) + spec consumption guards."""
import pytest

from prune.allocate import FP16_OVER_FP32, STAGE_RUNGS, greedy_allocate
from prune.recover_graft import spec_ratio_dict


def test_rungs_are_a_ratio_ladder():
    assert STAGE_RUNGS[0] == 0.0 and STAGE_RUNGS[-1] == 0.7
    assert all(b > a for a, b in zip(STAGE_RUNGS, STAGE_RUNGS[1:], strict=False))
    assert 0.0 < FP16_OVER_FP32 < 1.0


def test_greedy_prefers_high_latency_low_saliency():
    # stage0: expensive (10 ms) + worthless (flat saliency) → should be cut first/hardest;
    # stage1: cheap (1 ms) + precious (steep saliency) → should stay intact.
    n = len(STAGE_RUNGS)
    flat = [i * 0.01 for i in range(n)]
    steep = [i * 100.0 for i in range(n)]
    spec = greedy_allocate([10.0, 1.0], [flat, steep, flat], target_ms=10.0, head_ms=3.0)
    assert spec["stage_ratios"][0] > spec["stage_ratios"][1]
    assert spec["stage_ratios"][1] == 0.0
    assert spec["predicted_fp32_ms"] <= 10.0
    assert spec["fp16_estimate_ms"] == pytest.approx(
        spec["predicted_fp32_ms"] * FP16_OVER_FP32, abs=1e-3)


def test_greedy_infeasible_target_raises():
    n = len(STAGE_RUNGS)
    flat = [float(i) for i in range(n)]
    with pytest.raises(ValueError, match="infeasible"):
        greedy_allocate([10.0], [flat, flat], target_ms=1.0, head_ms=3.0)


def test_greedy_needs_rest_curve():
    with pytest.raises(ValueError, match="rest"):
        greedy_allocate([10.0, 1.0], [[0.0], [0.0]], target_ms=5.0)


def test_spec_ratio_dict_refuses_zero_rest():
    # validated before the model is touched — MetaPruner's default ratio must be in (0,1)
    with pytest.raises(ValueError, match="rest_ratio"):
        spec_ratio_dict(None, [2, 2, 4, 3, 3],
                        {"rest_ratio": 0.0, "stage_ratios": [0.1] * 5})
