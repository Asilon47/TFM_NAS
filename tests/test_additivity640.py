"""search/additivity640.py — the @640 additivity analysis (pure math + pairing rule)."""
import json

import pytest

from search.additivity640 import analyze, pair_from_files


def _pair(name: str, s: float, m: float, d: list[int]) -> dict:
    return {"name": name, "sum_ms": s, "measured_ms": m, "arch_d": d,
            "power_mode": "0", "clocks_locked": True}


def test_analyze_recovers_a_planted_affine_law() -> None:
    # measured = 1.2*sum + 0.5, exactly — the fit must recover it and rank perfectly.
    pairs = [_pair(f"a{i}", s, 1.2 * s + 0.5, [2, 2, 4, 3, 3])
             for i, s in enumerate((7.0, 10.0, 12.0, 27.0))]
    r = analyze(pairs)
    assert r["fit"]["slope"] == pytest.approx(1.2)
    assert r["fit"]["intercept"] == pytest.approx(0.5)
    assert r["fit"]["r2"] == pytest.approx(1.0)
    assert r["spearman_measured_vs_sum"] == pytest.approx(1.0)
    assert r["per_arch"][0]["sum_ms"] == 7.0          # sorted by sum
    assert r["ratio_max"] > r["ratio_min"] > 1.0      # affine intercept ⇒ ratio varies


def test_analyze_guards() -> None:
    with pytest.raises(ValueError, match=">=3"):
        analyze([_pair("a", 10, 12, [2] * 5), _pair("b", 11, 13, [2] * 5)])
    bad = [_pair("a", 10, 12, [2] * 5), _pair("b", 11, 13, [2] * 5),
           {**_pair("c", 12, 14, [2] * 5), "clocks_locked": False}]
    with pytest.raises(ValueError, match="unlocked clocks"):
        analyze(bad)


def test_rel_err_correlation_signs() -> None:
    # Plant: error grows with early depth (the DRAM hypothesis's signature).
    pairs = [
        _pair("lo", 10.0, 10.0 * 1.10, [2, 2, 4, 4, 4]),   # early=4, +10%
        _pair("mid", 10.0, 10.0 * 1.20, [3, 3, 4, 4, 4]),  # early=6, +20%
        _pair("hi", 10.0, 10.0 * 1.30, [4, 4, 4, 4, 4]),   # early=8, +30%
    ]
    r = analyze(pairs)
    assert r["rel_err_vs_early_depth_pearson"] == pytest.approx(1.0)
    assert r["rel_err_vs_depth_sum_pearson"] == pytest.approx(1.0)  # depth_sum co-varies here


def test_pair_from_files_pairing_rule(tmp_path) -> None:
    row = tmp_path / "addv_x_backbone_640.json"
    row.write_text(json.dumps({"name": "addv_x_backbone_640", "power_mode": "0",
                               "clocks_locked": True, "latency_ms": {"mean": 13.0}}))
    # candidate-style provenance: cached_latency_ms carries the sum
    (tmp_path / "addv_x_backbone_640.meta.json").write_text(json.dumps(
        {"backbone_only": True, "arch": {"d": [2, 2, 4, 3, 3]},
         "provenance": {"cached_latency_ms": 10.5}}))
    p = pair_from_files(row)
    assert p is not None and p["sum_ms"] == 10.5 and p["measured_ms"] == 13.0

    # winner/corner-style provenance: sum read from the source record
    src = tmp_path / "arch_min.json"
    src.write_text(json.dumps({"arch": {"d": [2] * 5}, "latency_ms": 6.8}))
    (tmp_path / "addv_x_backbone_640.meta.json").write_text(json.dumps(
        {"backbone_only": True, "arch": {"d": [2] * 5}, "provenance": {"source": str(src)}}))
    p = pair_from_files(row)
    assert p is not None and p["sum_ms"] == 6.8

    # e2e sidecars are skipped (backbone_only=False) — the probe is backbone-only
    (tmp_path / "addv_x_backbone_640.meta.json").write_text(json.dumps(
        {"backbone_only": False, "arch": {"d": [2] * 5}, "provenance": {}}))
    assert pair_from_files(row) is None
