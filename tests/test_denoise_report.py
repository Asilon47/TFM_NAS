"""search/denoise_report.py — the CP 3.5 tie-band sensitivity report (pure logic).

Synthetic candidates mirror the CP 3.5 shape: a saturated top cluster where the tie-band
decides the winner (accuracy-first / knee / latency-first) plus a fast noisy straggler.
No data/ dependency.
"""
import pytest

from search.denoise_report import band_regimes, winners_by_band


def _cand(d: list[int], mean: float, std: float, lat: float) -> dict:
    return {"arch": {"d": d}, "denoised_mean": mean, "denoised_std": std, "latency_ms": lat}


ACC_FIRST = _cand([2, 2, 4, 4, 2], 0.624, 0.003, 12.65)
KNEE = _cand([2, 2, 4, 3, 3], 0.616, 0.005, 11.21)
LAT_FIRST = _cand([2, 2, 4, 3, 2], 0.606, 0.012, 10.73)
STRAGGLER = _cand([2, 4, 3, 4, 4], 0.571, 0.025, 10.15)
CANDS = [ACC_FIRST, KNEE, LAT_FIRST, STRAGGLER]
T_MAX = 12.75


def test_band_regimes_values() -> None:
    r = band_regimes(CANDS)
    assert list(r) == [
        "argmax", "strict (top arch's own σ)", "typical (median σ)", "loose (max σ)",
    ]
    assert r["argmax"] == 0.0
    assert r["strict (top arch's own σ)"] == pytest.approx(0.003)
    assert r["typical (median σ)"] == pytest.approx((0.005 + 0.012) / 2)
    assert r["loose (max σ)"] == pytest.approx(0.025)


def test_each_regime_picks_its_winner() -> None:
    w = winners_by_band(CANDS, t_max=T_MAX)
    assert w["argmax"] is ACC_FIRST                     # plain argmax on the de-noised mean
    assert w["strict (top arch's own σ)"] is ACC_FIRST  # 0.624-0.616=0.008 > 0.003: no tie
    assert w["typical (median σ)"] is KNEE              # 0.008 <= 0.0085 -> fastest in tie
    assert w["loose (max σ)"] is LAT_FIRST              # 0.018 <= 0.025; straggler stays out


def test_custom_band_and_ceiling() -> None:
    w = winners_by_band(CANDS, t_max=T_MAX, bands={"wide": 0.06})
    assert w["wide"] is STRAGGLER                       # 0.053 <= 0.06 -> fastest overall
    w = winners_by_band(CANDS, t_max=11.0)              # ceiling drops the two accurate archs
    assert w["argmax"] is LAT_FIRST


def test_empty_candidates_raise() -> None:
    with pytest.raises(ValueError):
        band_regimes([])
