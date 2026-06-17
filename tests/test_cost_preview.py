"""Offline cost-preview + additivity-wiring tests (CP 2.2, post-LUT).

Deterministic and CI-safe: the rank/skyline helpers run on hand-built arrays, and
the LUT-driven paths use a synthetic unit LUT (every mbconv row = 1.0 ms / fixed
params+flops) so a subnet's summed latency equals its block count. The one on-disk
test uses the ``lut_path`` fixture and skips while the real sweep is partial
(mirrors tests/test_cost.py / test_lut_keydrift.py).
"""
import json
import math
import random

import pytest

from catalog.contracts import LutRow
from catalog.sweep import iter_sweep
from lut.loader import load_lut
from search.additivity_preview import (
    MAX_BLOCKS,
    MIN_BLOCKS,
    _arch_id,
    load_manifest,
    predictor_stats_from_manifest,
    report_from_manifest,
    select_subnets,
    write_calibration,
    write_manifest,
    write_pairs_csv,
)
from search.arch_to_blocks import random_arch_dict
from search.cost import CostError, load_latency_calibration
from search.cost_preview import (
    bytes_per_param,
    cost_row,
    kendall_tau,
    nondominated_indices,
    sample_costs,
    spearman,
)


def _row(mean: float = 1.0, peak: float = 1.0, params: int = 100,
         flops: int = 1000, key: str = "k") -> LutRow:
    return {
        "row_key": key, "block": "mbconv", "cfg": {}, "input_shape": [1, 3, 8, 8],
        "precision": "fp32",
        "latency_ms": {"mean": mean, "std": 0.0, "p50": mean, "p95": mean, "n": 1},
        "peak_mem_mib": peak, "params": params, "flops": flops,
        "achieved_bw_gbps": 0.0, "trt_version": "10.3.0", "power_mode": "0",
        "jetpack": None, "timestamp": "2026-01-01T00:00:00Z",
    }


@pytest.fixture(scope="module")
def unit_lut() -> dict[str, LutRow]:
    """Every mbconv catalog row = 1.0 ms -> a subnet's summed latency == its depth."""
    return {k: _row(mean=1.0, key=k) for _, _, _, k in iter_sweep(only_blocks=["mbconv"])}


# --- rank statistics ---------------------------------------------------------

def test_spearman_perfect_and_reversed():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_constant_is_nan():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_kendall_perfect_and_reversed():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_kendall_tie_corrected_in_range():
    # A tie in x must not crash and must stay a valid correlation.
    t = kendall_tau([1, 1, 2, 3], [1, 2, 3, 4])
    assert -1.0 <= t <= 1.0


# --- non-dominated (skyline, minimise both axes) -----------------------------

def test_nondominated_basic():
    # idx: 0(1,4) 1(2,2) 2(3,3) 3(2,5) 4(4,1)
    #   0 keep (lowest x); 1 keep (y<4); 2 dominated by 1; 3 dominated by 0&1; 4 keep (y=1)
    assert nondominated_indices([1, 2, 3, 2, 4], [4, 2, 3, 5, 1]) == [0, 1, 4]


def test_nondominated_full_tradeoff():
    assert nondominated_indices([1, 2, 3], [3, 2, 1]) == [0, 1, 2]


def test_nondominated_duplicates_keep_one():
    assert nondominated_indices([1, 1], [2, 2]) == [0]


# --- sample_costs against the unit LUT ---------------------------------------

def test_sample_costs_latency_equals_depth(unit_lut):
    df, gaps = sample_costs(unit_lut, n=25, seed=0, precision="fp32")
    assert gaps == 0 and len(df) == 25
    assert (df["latency_ms"] == df["depth"]).all()      # 1.0 ms per block
    assert {"depth", "latency_ms", "params", "flops",
            "peak_mem_mib", "resident_mem_mib"} <= set(df.columns)


def test_cost_row_resident_mem_scales_with_precision(unit_lut):
    arch = random_arch_dict(random.Random(0))
    r32 = cost_row(arch, unit_lut, "fp32")
    r16 = cost_row(arch, unit_lut, "fp16")
    assert r32["resident_mem_mib"] > r16["resident_mem_mib"]   # 4 vs 2 bytes/param
    assert r32["peak_mem_mib"] == r16["peak_mem_mib"]          # working set unchanged


def test_bytes_per_param_defaults_to_fp32():
    assert bytes_per_param("fp32") == 4
    assert bytes_per_param("fp16") == 2
    assert bytes_per_param(None) == 4


# --- additivity manifest wiring ----------------------------------------------

def test_select_subnets_spans_full_depth_range(unit_lut):
    entries = select_subnets(unit_lut, seed=0)
    depths = [e["depth"] for e in entries]
    assert min(depths) == MIN_BLOCKS == 11
    assert max(depths) == MAX_BLOCKS == 21
    assert depths == sorted(set(depths))        # one entry per distinct depth
    assert len(entries) >= 8
    for e in entries:
        assert e["summed_ms"] == pytest.approx(e["depth"])   # unit LUT: 1 ms/block
        assert e["measured_ms"] is None                      # on-device half deferred


def test_manifest_roundtrips(tmp_path, unit_lut):
    entries = select_subnets(unit_lut, seed=1)
    p = tmp_path / "m.json"
    write_manifest(p, entries)
    assert load_manifest(p) == entries


def test_report_requires_measured_values(tmp_path, unit_lut):
    p = tmp_path / "m.json"
    write_manifest(p, select_subnets(unit_lut, seed=0))
    with pytest.raises(ValueError, match="measured_ms"):
        report_from_manifest(p)


def test_report_passes_within_bar(tmp_path, unit_lut):
    entries = select_subnets(unit_lut, seed=0)
    for e in entries:                       # summed over-predicts a uniform 10%
        e["measured_ms"] = e["summed_ms"] / 1.10
    p = tmp_path / "pass.json"
    write_manifest(p, entries)
    report = report_from_manifest(p)
    assert not report.breached
    assert report.aggregate == pytest.approx(0.10, abs=1e-9)


def test_report_breaches_on_deep_bin(tmp_path, unit_lut):
    entries = select_subnets(unit_lut, seed=0)
    for e in entries:
        e["measured_ms"] = e["summed_ms"]               # perfect everywhere...
    deepest = max(entries, key=lambda e: e["depth"])
    deepest["measured_ms"] = deepest["summed_ms"] / 1.50  # ...except +50% at max depth
    p = tmp_path / "breach.json"
    write_manifest(p, entries)
    report = report_from_manifest(p)
    assert report.breached
    assert deepest["depth"] in report.breaching_depths


# --- k subnets per depth (3/depth measurement design) ------------------------

def test_select_subnets_k_per_depth(unit_lut):
    entries = select_subnets(unit_lut, seed=0, k=3)
    depths = [e["depth"] for e in entries]
    counts = {d: depths.count(d) for d in set(depths)}
    assert min(counts) == MIN_BLOCKS and max(counts) == MAX_BLOCKS  # spans 11..21
    assert all(1 <= c <= 3 for c in counts.values())               # never exceeds k
    assert sum(1 for c in counts.values() if c == 3) >= 7          # most depths reach k
    assert all(e["summed_ms"] == pytest.approx(e["depth"]) for e in entries)  # unit LUT
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))                               # globally unique


def test_select_subnets_id_is_deterministic_and_arch_derived(unit_lut):
    a = select_subnets(unit_lut, seed=0, k=2)
    b = select_subnets(unit_lut, seed=0, k=2)
    assert [e["id"] for e in a] == [e["id"] for e in b]            # seed-deterministic
    assert all(e["id"] == _arch_id(e["arch_dict"]) for e in a)     # purely arch-derived


def test_report_bins_within_depth_with_k(tmp_path, unit_lut):
    entries = select_subnets(unit_lut, seed=0, k=3)
    deepest = max(e["depth"] for e in entries)
    for e in entries:  # perfect everywhere except every arch in the deepest bin +50%
        e["measured_ms"] = e["summed_ms"] / (1.50 if e["depth"] == deepest else 1.0)
    p = tmp_path / "k_breach.json"
    write_manifest(p, entries)
    report = report_from_manifest(p)
    assert report.breached and deepest in report.breaching_depths
    assert MIN_BLOCKS not in report.breaching_depths               # shallow bin is exact


# --- predictor stats + calibration surfaced in `report` (scipy-backed) --------

def _fill(entries: list[dict], factor: float = 1.10) -> list[dict]:
    """measured = summed/factor -> a clean, uniform `factor` over-prediction."""
    for e in entries:
        e["measured_ms"] = e["summed_ms"] / factor
    return entries


def test_predictor_stats_from_manifest_high_rho_and_exact_fit(tmp_path, unit_lut):
    pytest.importorskip("scipy")
    entries = _fill(select_subnets(unit_lut, seed=0, k=3))
    p = tmp_path / "filled.json"
    write_manifest(p, entries)
    stats = predictor_stats_from_manifest(p)
    assert stats.n == len(entries)
    assert stats.spearman_rho == pytest.approx(1.0)        # monotone -> perfect ranking
    assert stats.fit.slope == pytest.approx(1.0 / 1.10)    # measured = summed/1.1 exactly
    assert stats.mape_calibrated < 1e-9                    # the affine fit is exact here


def test_write_calibration_roundtrips_through_cost_loader(tmp_path, unit_lut):
    pytest.importorskip("scipy")
    entries = _fill(select_subnets(unit_lut, seed=0, k=3))
    p = tmp_path / "filled.json"
    write_manifest(p, entries)
    stats = predictor_stats_from_manifest(p)
    out = tmp_path / "latency_calibration.json"
    write_calibration(out, stats, precision="fp32")
    # search.cost reads back exactly slope + intercept
    assert load_latency_calibration(out) == {
        "slope": pytest.approx(1.0 / 1.10),
        "intercept": pytest.approx(stats.fit.intercept)}
    raw = json.loads(out.read_text())                      # provenance is preserved
    assert raw["precision"] == "fp32" and raw["n"] == len(entries)
    assert "spearman_rho" in raw and "mult_factor" in raw


def test_write_pairs_csv_has_expected_columns(tmp_path, unit_lut):
    pytest.importorskip("scipy")
    entries = _fill(select_subnets(unit_lut, seed=0, k=3))
    p = tmp_path / "filled.json"
    write_manifest(p, entries)
    fit = predictor_stats_from_manifest(p).fit
    out = tmp_path / "pairs.csv"
    write_pairs_csv(out, load_manifest(p), fit)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "id,depth,summed_ms,measured_ms,rel_error,calibrated_ms"
    assert len(lines) == 1 + len(entries)


# --- on-disk real LUT smoke (skips while the sweep is partial) ----------------

def test_on_disk_lut_preview_and_manifest(lut_path):
    with open(lut_path) as f:
        precision = json.loads(f.readline()).get("precision")
    lut = load_lut(lut_path, precision=precision)
    df, gaps = sample_costs(lut, n=10, seed=0, precision=precision)
    if gaps or df.empty:
        pytest.skip("LUT partial (coverage gaps) — runs once the sweep is complete")
    assert (df["latency_ms"] > 0).all() and (df["params"] > 0).all()
    try:
        entries = select_subnets(lut, seed=0)
    except CostError:
        pytest.skip("LUT partial — runs once the sweep is complete")
    assert min(e["depth"] for e in entries) == MIN_BLOCKS
    assert max(e["depth"] for e in entries) == MAX_BLOCKS
    assert all(e["summed_ms"] > 0 for e in entries)
