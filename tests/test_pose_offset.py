"""search/pose_offset.py — offset algebra, regime guards, and CostOffset compatibility."""
import json

import pytest

from search.cost import load_stem_head_offset
from search.pose_offset import pose_offset_record


def _row(name: str, mean_ms: float, *, peak=100.0, precision="fp32", power_mode="0",
         locked=True) -> dict:
    return {"name": name, "latency_ms": {"mean": mean_ms}, "peak_mem_mib": peak,
            "precision": precision, "power_mode": power_mode, "clocks_locked": locked,
            "trt_version": "10.3.0"}


E2E = _row("winner_e2e", 13.0, peak=180.0)
BACKBONE = _row("winner_backbone", 10.5, peak=120.0)


def test_offset_algebra_and_additivity_point() -> None:
    rec = pose_offset_record(E2E, BACKBONE, offset_params=2_700_000, offset_flops=3_000_000,
                             lut_summed_ms=11.208)
    assert rec["latency_ms"] == pytest.approx(2.5)
    assert rec["peak_mem_mib"] == 180.0                       # e2e whole-model working set
    assert rec["params"] == 2_700_000
    b = rec["backbone_measured_vs_lut_sum"]
    assert b["ratio"] == pytest.approx(10.5 / 11.208)
    assert b["delta_pct"] == pytest.approx(100 * (10.5 - 11.208) / 11.208)


def test_costoffset_roundtrip(tmp_path) -> None:
    rec = pose_offset_record(E2E, BACKBONE, offset_params=7, offset_flops=9)
    p = tmp_path / "pose_stem_head_offset.json"
    p.write_text(json.dumps(rec))
    off = load_stem_head_offset(p)                            # search.cost reads it directly
    assert off == {"latency_ms": pytest.approx(2.5), "peak_mem_mib": 180.0,
                   "params": 7, "flops": 9}


def test_regime_guards() -> None:
    with pytest.raises(ValueError, match="precision"):
        pose_offset_record(E2E, _row("b", 10.5, precision="fp16"))
    with pytest.raises(ValueError, match="power_mode"):
        pose_offset_record(E2E, _row("b", 10.5, power_mode="2"))
    with pytest.raises(ValueError, match="clocks_locked"):
        pose_offset_record(E2E, _row("b", 10.5, locked=False))


def test_negative_offset_refused() -> None:
    with pytest.raises(ValueError, match="inconsistency"):
        pose_offset_record(_row("e", 9.0), _row("b", 10.5))
