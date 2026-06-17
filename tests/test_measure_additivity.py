"""Tests for the pure helpers of lut/orchestrate/measure_additivity.py.

The SSH/Docker measurement path needs the Jetson and is verified by the on-device
``--limit 1`` smoke; here we cover only the pure transforms — normalizing a remote
bench result into a cost component, and the idempotent-resume pending filter.
"""
import pytest

pytest.importorskip("fabric")   # the driver imports ssh_client (fabric); skip if absent

from lut.orchestrate import measure_additivity as md  # noqa: E402


def test_bench_to_component_extracts_mean_latency_and_peak():
    bench = {
        "latency_ms": {"mean": 2.5, "std": 0.1, "p50": 2.4, "p95": 2.7, "n": 300},
        "peak_mem_mib": 14.0, "io_bytes": 123, "trt_version": "10.3.0",
    }
    assert md.bench_to_component(bench, params=1234, flops=5678) == {
        "latency_ms": 2.5, "peak_mem_mib": 14.0, "params": 1234, "flops": 5678}


def test_pending_entries_returns_only_unmeasured():
    entries = [
        {"id": "a", "measured_ms": None},
        {"id": "b", "measured_ms": 3.1},
        {"id": "c"},                       # missing key counts as pending
    ]
    assert [e["id"] for e in md.pending_entries(entries)] == ["a", "c"]
