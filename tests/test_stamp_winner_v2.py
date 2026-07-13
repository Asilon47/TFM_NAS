"""Winner-v2-OFA Track 4 — stamp_winner_v2 pure builders (median, verdict, additive link)."""
import pytest

from search.stamp_winner_v2 import (
    link_v1,
    median_fp16_ms,
    verdict,
    winner_v2_record,
)


def _row(ms, precision="fp16", fresh=True, pm="0"):
    r = {"latency_ms": {"mean": ms}, "precision": precision, "power_mode": pm,
         "clocks_locked": True, "trt_version": "10.3.0"}
    if precision == "fp16":
        r["fresh_timing_cache"] = fresh
    return r


def test_median_fp16_needs_three_fresh_cache_builds():
    with pytest.raises(ValueError, match=">=3"):
        median_fp16_ms([_row(7.4), _row(7.5)])
    assert median_fp16_ms([_row(7.4), _row(7.6), _row(7.5)]) == pytest.approx(7.5)
    # a shared-cache row (fresh_timing_cache missing/false) is refused
    with pytest.raises(ValueError, match="fresh_timing_cache"):
        median_fp16_ms([_row(7.4), _row(7.5, fresh=False), _row(7.6)])


def test_verdict_strict_both_axes():
    v = verdict(9.9, 7.1, _row(12.7, "fp32"), _row(7.75))
    assert v["beats_fp32"] and v["beats_fp16"] and v["beats_both_axes"]
    assert v["margin_fp16_pct"] == pytest.approx(100 * (7.75 - 7.1) / 7.75, abs=1e-6)
    # fp16 over the bar → fails both-axes even though fp32 passes
    v2 = verdict(9.9, 7.9, _row(12.7, "fp32"), _row(7.75))
    assert v2["beats_fp32"] and not v2["beats_fp16"] and not v2["beats_both_axes"]


def test_winner_v2_record_certifies_and_computes_median():
    rec = winner_v2_record(
        arch={"ks": [3] * 20, "e": [4] * 20, "d": [2, 2, 4, 3, 3]},
        compression={"technique": "global_taylor", "kd": {"alpha": 1.0}, "params": 780000},
        accuracy={"map_seed0": 0.803},
        fp32_row=_row(9.9, "fp32"), fp16_build_rows=[_row(7.1), _row(7.2), _row(7.0)],
        base_fp32_row=_row(12.7, "fp32"), base_fp16_row=_row(7.75))
    assert rec["certified"] is True
    assert rec["latency"]["fp16_ms_median"] == pytest.approx(7.1)
    assert rec["latency"]["fp16_builds_n"] == 3
    assert rec["deliverable"] == "winner-v2-OFA"


def test_winner_v2_record_regime_mismatch_raises():
    with pytest.raises(ValueError, match="power_mode|regime|no stamp"):
        winner_v2_record(
            arch={}, compression={}, accuracy={"map_seed0": 0.8},
            fp32_row=_row(9.9, "fp32", pm="0"),
            fp16_build_rows=[_row(7.1), _row(7.2), _row(7.0)],
            base_fp32_row=_row(12.7, "fp32", pm="2"),   # wrong power mode
            base_fp16_row=_row(7.75))


def test_link_v1_is_additive_only():
    v1 = {"arch": {"d": [2]}, "acc": 0.61, "e2e": {"x": 1}}
    linked = link_v1(v1, path="state/winner_v2_ofa/winner.json", summary="pruned graft")
    assert linked["arch"] == v1["arch"] and linked["acc"] == 0.61 and linked["e2e"] == {"x": 1}
    assert linked["winner_v2_ofa"]["path"].endswith("winner.json")
    with pytest.raises(ValueError, match="already has"):
        link_v1(linked, path="x", summary="y")
    assert link_v1(linked, path="x", summary="y", force=True)["winner_v2_ofa"]["path"] == "x"
