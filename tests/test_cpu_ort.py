"""Contract tests for the CPU timing core.

The topology tests are the important ones: this laptop's hyperthread siblings
are {0,5} {1,2} {3,4} {6,7} {8,9} {10,11}, so the "obvious" taskset -c 0-5 lands
on THREE physical cores with every thread contended. A regression here would
silently measure HT contention and report it as thread scaling.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lut.orchestrate.cpu_ort import (
    ALLOW_SPINNING,
    BenchConfig,
    LatencyStats,
    all_cpus,
    parse_cpu_list,
    physical_p_cores,
    summarize,
)

# The real Core Ultra 9 185H layout, as probed from /sys on 2026-07-15.
SIBLINGS_185H = {
    0: "0,5", 1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "0,5",
    6: "6-7", 7: "6-7", 8: "8-9", 9: "8-9", 10: "10-11", 11: "10-11",
}


@pytest.fixture
def fake_sysfs(tmp_path: Path) -> Path:
    """A minimal /sys mirroring the 185H: 6 P-cores (SMT) + 10 E-cores."""
    (tmp_path / "devices/cpu_core").mkdir(parents=True)
    (tmp_path / "devices/cpu_core/cpus").write_text("0-11\n")
    (tmp_path / "devices/system/cpu").mkdir(parents=True)
    (tmp_path / "devices/system/cpu/online").write_text("0-21\n")
    for cpu, sibs in SIBLINGS_185H.items():
        topo = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(sibs + "\n")
    return tmp_path


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0-11", list(range(12))),
        ("0,5", [0, 5]),
        ("1-2", [1, 2]),
        ("12-19,20-21", list(range(12, 22))),
        ("3", [3]),
        ("0-21\n", list(range(22))),
    ],
)
def test_parse_cpu_list(spec: str, expected: list[int]) -> None:
    assert parse_cpu_list(spec) == expected


def test_physical_p_cores_picks_one_per_core(fake_sysfs: Path) -> None:
    """One logical CPU per physical P-core -- NOT 0-5."""
    assert physical_p_cores(fake_sysfs) == [0, 1, 3, 6, 8, 10]


def test_physical_p_cores_never_returns_a_sibling_pair(fake_sysfs: Path) -> None:
    """No two returned CPUs may share a physical core (the 0-5 trap)."""
    got = physical_p_cores(fake_sysfs)
    for cpu in got:
        sibs = set(parse_cpu_list(SIBLINGS_185H[cpu]))
        assert sibs & set(got) == {cpu}, f"cpu{cpu} shares a core with another selected cpu"


def test_physical_p_cores_falls_back_when_not_hybrid(tmp_path: Path) -> None:
    """Non-hybrid CPUs have no devices/cpu_core -- fall back to online CPUs."""
    (tmp_path / "devices/system/cpu").mkdir(parents=True)
    (tmp_path / "devices/system/cpu/online").write_text("0-3\n")
    for cpu in range(4):
        topo = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(f"{cpu}\n")
    assert physical_p_cores(tmp_path) == [0, 1, 2, 3]


def test_all_cpus(fake_sysfs: Path) -> None:
    assert all_cpus(fake_sysfs) == list(range(22))


def test_summarize_computes_order_statistics() -> None:
    stats = summarize([10.0, 12.0, 11.0, 13.0, 14.0])
    assert stats.n == 5
    assert stats.mean == pytest.approx(12.0)
    assert stats.p50 == pytest.approx(12.0)
    assert stats.p95 == pytest.approx(14.0, abs=0.5)
    assert stats.std > 0


def test_summarize_single_sample_has_zero_std() -> None:
    stats = summarize([7.5])
    assert stats.n == 1
    assert stats.std == 0.0
    assert stats.p50 == pytest.approx(7.5)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no samples"):
        summarize([])


def test_latency_stats_as_dict_matches_e2e_schema() -> None:
    """Rows must carry the same keys as data/e2e/*.json so tooling reads both."""
    d = LatencyStats(mean=1.0, std=0.1, p50=0.9, p95=1.2, n=60).as_dict()
    assert set(d) == {"mean", "std", "p50", "p95", "n"}
    assert d["n"] == 60


def test_spinning_stays_disabled() -> None:
    """Guards the 18.6x artifact.

    ORT's intra-op workers spin-wait after run() returns. With ~29 sessions rotating, each
    model's run collides with the previous pools still burning the pinned cores. Measured at
    t6, interleaved: 564.5 ms spinning vs 30.4 ms not -- and it scales with thread count, so it
    mimics the bandwidth effect this bench exists to measure. If this test fails, someone is
    about to publish contention as a memory-boundness result.
    """
    assert ALLOW_SPINNING == "0"


def test_bench_config_is_hashable_and_frozen() -> None:
    cfg = BenchConfig(name="t4", threads=4, affinity=(0, 1, 3, 6))
    assert hash(cfg)
    with pytest.raises(FrozenInstanceError):
        cfg.threads = 8  # type: ignore[misc]
