"""Contract tests for the CPU bench driver.

The interleave tests matter most: model-at-a-time benching would let thermal drift on a laptop
become a systematic bias that tracks bench order -- and since the natural order is by family,
that bias would correlate with architecture family and could manufacture the very effect the
experiment tests for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lut.orchestrate.cpu_bench import (
    build_configs,
    build_row,
    drift_detected,
    root_dir,
    rotate,
    row_path,
    schedule,
)
from lut.orchestrate.cpu_ort import BenchConfig, LatencyStats

P_CORES = [0, 1, 3, 6, 8, 10]


def test_build_configs_shape() -> None:
    cfgs = build_configs(P_CORES)
    assert [c.name for c in cfgs] == ["t1", "t2", "t4", "t6", "all22"]


def test_build_configs_pin_one_thread_per_physical_core() -> None:
    cfgs = {c.name: c for c in build_configs(P_CORES)}
    assert cfgs["t1"].affinity == (0,)
    assert cfgs["t2"].affinity == (0, 1)
    assert cfgs["t4"].affinity == (0, 1, 3, 6)
    assert cfgs["t6"].affinity == (0, 1, 3, 6, 8, 10)
    for name in ("t1", "t2", "t4", "t6"):
        assert cfgs[name].threads == len(cfgs[name].affinity)


def test_all22_is_unpinned_and_default_pool() -> None:
    """The practical number: ORT's own pool, no mask. Confounded by design, labelled as such."""
    cfgs = {c.name: c for c in build_configs(P_CORES)}
    assert cfgs["all22"].affinity == ()
    assert cfgs["all22"].threads == 0


def test_build_configs_rejects_too_few_cores() -> None:
    with pytest.raises(ValueError, match="6 physical P-cores"):
        build_configs([0, 1, 3])


def test_rotate() -> None:
    assert rotate([1, 2, 3, 4], 0) == [1, 2, 3, 4]
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate([1, 2, 3, 4], 5) == [2, 3, 4, 1]


def test_rotate_empty() -> None:
    assert rotate([], 3) == []


def test_schedule_times_every_model_once_per_round() -> None:
    names = ["a", "b", "c"]
    sched = schedule(names, rounds=4)
    assert len(sched) == 12
    for r in range(4):
        in_round = [n for (rr, n) in sched if rr == r]
        assert sorted(in_round) == ["a", "b", "c"], f"round {r} is not a full pass"


def test_schedule_rotates_order_between_rounds() -> None:
    """Rotation is what turns thermal drift into common-mode noise instead of order bias."""
    names = ["a", "b", "c"]
    sched = schedule(names, rounds=3)
    firsts = [next(n for (rr, n) in sched if rr == r) for r in range(3)]
    assert firsts == ["a", "b", "c"]


def test_schedule_zero_rounds_is_empty() -> None:
    assert schedule(["a"], rounds=0) == []


def test_drift_detected_flags_a_hot_laptop() -> None:
    assert drift_detected({0: 10.0, 1: 10.1, 2: 11.0}) is True


def test_drift_detected_passes_a_stable_run() -> None:
    assert drift_detected({0: 10.0, 1: 10.1, 2: 10.2}) is False


def test_drift_detected_uses_round_zero_as_reference() -> None:
    """Drift is measured against the cold first round, not the min."""
    assert drift_detected({0: 10.0, 1: 10.4}, tol=0.05) is False
    assert drift_detected({0: 10.0, 1: 10.6}, tol=0.05) is True


def test_drift_detected_needs_two_rounds() -> None:
    assert drift_detected({0: 10.0}) is False
    assert drift_detected({}) is False


def test_build_row_matches_e2e_schema() -> None:
    cfg = BenchConfig(name="t4", threads=4, affinity=(0, 1, 3, 6))
    stats = LatencyStats(mean=42.0, std=0.5, p50=41.9, p95=43.0, n=60)
    env = {
        "governor": "powersave",
        "cpu_mhz_mean": 3800.0,
        "loadavg_1m": 0.4,
        "on_ac": True,
        "ort_version": "1.27.0",
        "cpu_model": "Intel(R) Core(TM) Ultra 9 185H",
    }
    row = build_row("prune_base_r20_640", cfg, stats, env)

    assert row["name"] == "prune_base_r20_640"
    assert row["config"] == "t4"
    assert row["precision"] == "fp32"
    assert row["imgsz"] == 640
    assert row["threads"] == 4
    assert row["affinity"] == [0, 1, 3, 6]
    assert row["source"] == "x86_ort"
    assert row["latency_ms"] == stats.as_dict()
    assert row["thermal_drift_detected"] is False
    assert "timestamp" in row


def test_build_row_carries_the_env_stamp() -> None:
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    row = build_row("x", cfg, stats, {"governor": "powersave", "on_ac": True})
    assert row["governor"] == "powersave"
    assert row["on_ac"] is True


def test_build_row_never_emits_fp16() -> None:
    """ORT CPU fp16 is emulated -- an fp16 row would measure emulation, not the model."""
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    row = build_row("x", cfg, stats, {})
    assert row["precision"] == "fp32"


def test_build_row_never_claims_jetson_provenance() -> None:
    """No CPU number may be mistakable for a device latency in a later claim."""
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    row = build_row("x", cfg, stats, {})
    assert row["source"] == "x86_ort"
    assert "trt_version" not in row
    assert "clocks_locked" not in row


def test_build_row_is_json_serializable() -> None:
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    json.dumps(build_row("x", cfg, stats, {}))


def test_row_path(tmp_path: Path) -> None:
    assert row_path(tmp_path, "dense_w13_640", "t4") == tmp_path / "dense_w13_640__t4.json"


def test_root_dir_is_the_repo() -> None:
    assert (root_dir() / "pyproject.toml").is_file()
