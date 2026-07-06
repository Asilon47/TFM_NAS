"""search/stamp_winner_e2e.py — honest-speedup math, additive-only merge, regime guards."""
import pytest

from search.stamp_winner_e2e import e2e_block, stamped


def _row(name: str, mean_ms: float, *, precision="fp32", power_mode="0", locked=True) -> dict:
    return {"name": name, "latency_ms": {"mean": mean_ms}, "peak_mem_mib": 100.0,
            "precision": precision, "power_mode": power_mode, "clocks_locked": locked,
            "trt_version": "10.3.0"}


WINNER = {"arch": {"d": [2, 2, 4, 3, 3]}, "latency_ms": 11.208,
          "vs_yolo11n": {"baseline_latency_ms": 12.755, "latency_speedup_pct": 12.12},
          "acc": 0.6101}


def test_block_math_and_verdict() -> None:
    block = e2e_block(WINNER, _row("w", 12.0), _row("base", 12.8),
                      backbone_row=_row("bb", 10.4),
                      fallback_rows={"idx11": _row("f11", 11.5)})
    assert block["speedup_pct_e2e"] == pytest.approx(100 * (12.8 - 12.0) / 12.8)
    assert block["winner_beats_baseline_e2e"] is True
    assert block["lut_sum_ms"] == 11.208
    assert block["offset_ms_derived"] == pytest.approx(1.6)
    assert block["fallbacks"]["idx11"]["speedup_pct_e2e"] == pytest.approx(
        100 * (12.8 - 11.5) / 12.8)


def test_negative_margin_gets_false_verdict_not_an_error() -> None:
    block = e2e_block(WINNER, _row("w", 13.4), _row("base", 12.8))
    assert block["winner_beats_baseline_e2e"] is False        # honest, not raised — re-pick path
    assert block["speedup_pct_e2e"] < 0


def test_regime_guard_across_rows() -> None:
    with pytest.raises(ValueError, match="one session"):
        e2e_block(WINNER, _row("w", 12.0), _row("base", 12.8, power_mode="2"))
    with pytest.raises(ValueError, match="clocks-locked"):
        e2e_block(WINNER, _row("w", 12.0), _row("base", 12.8, locked=False))


def test_stamp_is_additive_only() -> None:
    block = e2e_block(WINNER, _row("w", 12.0), _row("base", 12.8))
    out = stamped(WINNER, block)
    assert out["e2e"] is block
    assert {k: v for k, v in out.items() if k != "e2e"} == WINNER   # nothing else touched
    with pytest.raises(ValueError, match="--force"):
        stamped(out, block)                                    # re-stamp needs force
    assert stamped(out, block, force=True)["e2e"] is block
