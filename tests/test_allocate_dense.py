"""Arm S (beat-n program) — prune/allocate_dense.py pure math + contract pins.

The honest-build path (donor load → prep rewrite → prune → ONNX → act) needs .venv-nas and
is exercised by the module's own runs; everything here is dependency-free.
"""
import pytest

from prune.allocate_dense import (
    DEFAULT_DONOR,
    FP16_BAR_MS,
    dense_spec_payload,
)
from search.latency_model import (
    PHYSICAL_DENSE_PRUNED_FP16,
    PHYSICAL_DENSE_PRUNED_FP32,
    act_limit_for_ms,
)


def test_fp32_binds_for_this_family():
    """At the baseline bars the fp32 cap (≈263 MB) sits BELOW the fp16 cap (≈274 MB) —
    the reason fences are --target-fp32-ms; if a refit ever flips this, the module's
    fence logic must be revisited."""
    cap32 = act_limit_for_ms(12.74, PHYSICAL_DENSE_PRUNED_FP32)
    cap16 = act_limit_for_ms(FP16_BAR_MS, PHYSICAL_DENSE_PRUNED_FP16)
    assert cap32 == pytest.approx(263.5, abs=1.0)
    assert cap16 == pytest.approx(274.3, abs=1.0)
    assert cap32 < cap16


def test_dense_spec_payload_carries_the_ladder_keys():
    """prune_baseline's spec row copies these — regression guard for spec consumption."""
    p = dense_spec_payload([0.3] * 5, 0.2, act_honest=252.0, act_predicted=250.1,
                           params_after=2_100_000, target_fp32_ms=12.2, act_max=252.4,
                           donor="s39.pt")
    for k in ("stage_ratios", "rest_ratio", "predicted_fp32_ms", "fp16_estimate_ms",
              "act_mbytes_honest"):
        assert k in p, k
    # predictions follow the pinned PRUNED-currency law
    assert p["predicted_fp32_ms"] == pytest.approx(-0.0216 + 0.048426 * 252.0, abs=0.01)
    assert p["fp16_estimate_ms"] == pytest.approx(1.3172 + 0.023450 * 252.0, abs=0.01)
    assert p["family"] == "dense_pruned"
    assert p["donor"] == "s39.pt"
    assert p["fence"]["fit"].endswith("subfamilies.pruned")


def test_default_donor_is_the_searched_winner():
    assert "s39-40-38-38-14" in str(DEFAULT_DONOR)


def test_honest_path_stays_in_the_pruned_currency():
    """Source pin: every probe must apply the yolo_tp_prep rewrite BEFORE pruning and use
    the deploy exporter — the same pipeline that produced every measured prune_base_* row.
    A stock-graph probe would price in the wrong currency (549 vs 197 MB on the baseline)."""
    import inspect

    from prune.allocate_dense import honest_dense_spec_features

    src = inspect.getsource(honest_dense_spec_features)
    assert src.index("prepare_yolo_for_pruning_(model)") < src.index("prune_graft(")
    assert "_export_deploy_onnx" in src
