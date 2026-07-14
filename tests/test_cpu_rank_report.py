"""Contract tests for the rank-correlation report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lut.orchestrate.cpu_rank_report import (
    fit_reference,
    load_cpu,
    load_jetson,
    rank_stats,
    residuals,
)


def test_rank_stats_perfect_agreement() -> None:
    cpu = {"a": 1.0, "b": 2.0, "c": 3.0}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    got = rank_stats(cpu, jetson)
    assert got["spearman"] == pytest.approx(1.0)
    assert got["kendall"] == pytest.approx(1.0)
    assert got["n"] == 3


def test_rank_stats_is_invariant_to_monotone_scaling() -> None:
    """The reason governor state doesn't matter for a rank check."""
    cpu = {"a": 1.0, "b": 2.0, "c": 3.0}
    slow = {k: v * 1.35 + 4.0 for k, v in cpu.items()}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    assert rank_stats(cpu, jetson)["spearman"] == rank_stats(slow, jetson)["spearman"]


def test_rank_stats_perfect_inversion() -> None:
    cpu = {"a": 3.0, "b": 2.0, "c": 1.0}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    assert rank_stats(cpu, jetson)["spearman"] == pytest.approx(-1.0)


def test_rank_stats_uses_only_paired_names() -> None:
    """An unpaired CPU row must be dropped, not crash or corrupt the correlation."""
    cpu = {"a": 1.0, "b": 2.0, "c": 3.0, "orphan": 9.0}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0, "missing": 5.0}
    got = rank_stats(cpu, jetson)
    assert got["n"] == 3
    # The orphan's wild value must not leak into the correlation.
    assert got["spearman"] == pytest.approx(1.0)


def test_rank_stats_needs_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        rank_stats({"a": 1.0}, {"a": 2.0})


def test_fit_reference_recovers_a_known_line() -> None:
    """latency = 2*params + 1 over the reference families only."""
    params = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "g1": 2.0}
    lat = {"r1": 3.0, "r2": 5.0, "r3": 7.0, "g1": 99.0}
    fams = {"r1": "dense", "r2": "prune", "r3": "baseline", "g1": "graft"}
    slope, intercept = fit_reference(params, lat, fams)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_fit_reference_ignores_the_anchor() -> None:
    """yolo11s at 9.7M would dominate the slope as a leverage point."""
    params = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "anchor": 9.7}
    lat = {"r1": 3.0, "r2": 5.0, "r3": 7.0, "anchor": 500.0}
    fams = {"r1": "dense", "r2": "prune", "r3": "baseline", "anchor": "anchor"}
    slope, _ = fit_reference(params, lat, fams)
    assert slope == pytest.approx(2.0)


def test_fit_reference_ignores_dense_nas() -> None:
    """dense_nas is a distinct design process -- reported, not part of the reference."""
    params = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "n1": 2.5}
    lat = {"r1": 3.0, "r2": 5.0, "r3": 7.0, "n1": 400.0}
    fams = {"r1": "dense", "r2": "prune", "r3": "baseline", "n1": "dense_nas"}
    slope, _ = fit_reference(params, lat, fams)
    assert slope == pytest.approx(2.0)


def test_fit_reference_needs_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_reference({"r1": 1.0}, {"r1": 3.0}, {"r1": "dense"})


def test_residuals_are_ms_above_the_line() -> None:
    params = {"g1": 2.0, "g2": 3.0}
    lat = {"g1": 8.0, "g2": 7.0}
    got = residuals(params, lat, slope=2.0, intercept=1.0, names=["g1", "g2"])
    assert got["g1"] == pytest.approx(3.0)  # 8 - (2*2+1)
    assert got["g2"] == pytest.approx(0.0)  # 7 - (2*3+1)


def test_load_jetson_takes_fp32_p50_only(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps(
            {"name": "a", "precision": "fp32", "latency_ms": {"p50": 12.7}, "source": "jetson_trt"}
        )
    )
    (tmp_path / "a_fp16.json").write_text(
        json.dumps(
            {
                "name": "a_fp16",
                "precision": "fp16",
                "latency_ms": {"p50": 7.7},
                "source": "jetson_trt",
            }
        )
    )
    (tmp_path / "report.json").write_text(json.dumps({"not": "a row"}))
    got = load_jetson(tmp_path)
    assert got == {"a": 12.7}


def test_load_cpu_keys_by_name_and_config(tmp_path: Path) -> None:
    (tmp_path / "a__t4.json").write_text(
        json.dumps({"name": "a", "config": "t4", "latency_ms": {"p50": 40.0}, "source": "x86_ort"})
    )
    got = load_cpu(tmp_path)
    assert got == {("a", "t4"): 40.0}


def test_load_cpu_ignores_the_report_file(tmp_path: Path) -> None:
    """rank_report.json lives in the same dir and has no __config suffix."""
    (tmp_path / "rank_report.json").write_text(json.dumps({"per_config": {}}))
    (tmp_path / "a__t1.json").write_text(
        json.dumps({"name": "a", "config": "t1", "latency_ms": {"p50": 1.0}, "source": "x86_ort"})
    )
    assert load_cpu(tmp_path) == {("a", "t1"): 1.0}
