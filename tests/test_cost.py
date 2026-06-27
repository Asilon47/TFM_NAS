"""CP 2.2 contract: composite cost is additive — except peak_mem, which maxes.

``_aggregate`` is exercised directly on hand-built rows (no file dependency);
``cost()``'s lookup + missing-key paths use small in-memory LUTs; one round-trip
test runs against the on-disk LUT via the ``lut_path`` fixture and skips while
the real sweep is still partial (mirrors tests/test_lut_keydrift.py).
"""
import json
import random

import pytest

from catalog.contracts import CostDict, CostOffset, LutRow
from catalog.sweep import row_key
from lut.loader import load_lut
from search.arch_to_blocks import _random_arch_dict, arch_to_blocks, arch_to_keys
from search.cost import (
    IDENTITY_CALIBRATION,
    ZERO_OFFSET,
    CostError,
    _aggregate,
    cost,
    cost_from_path,
    load_latency_calibration,
    load_stem_head_offset,
    offset_from_measurements,
    resident_mem_mib,
)


def _row(mean: float = 1.0, peak: float = 1.0, params: int = 100,
         flops: int = 1000, key: str = "k") -> LutRow:
    """A complete LutRow; only the four costed fields vary per test."""
    return {
        "row_key": key, "block": "mbconv", "cfg": {}, "input_shape": [1, 3, 8, 8],
        "precision": "fp16",
        "latency_ms": {"mean": mean, "std": 0.0, "p50": mean, "p95": mean, "n": 1},
        "peak_mem_mib": peak, "params": params, "flops": flops,
        "achieved_bw_gbps": 0.0, "trt_version": "10.3.0", "power_mode": "0",
        "jetpack": None, "timestamp": "2026-01-01T00:00:00Z",
    }


# ---- _aggregate: the heterogeneous reduce -----------------------------------

def test_latency_params_flops_are_summed():
    rows = [_row(mean=2.0, params=10, flops=100),
            _row(mean=3.0, params=20, flops=200)]
    c = _aggregate(rows, ZERO_OFFSET)
    assert c["latency_ms"] == pytest.approx(5.0)
    assert c["params"] == 30
    assert c["flops"] == 300


def test_peak_mem_is_max_not_sum():
    c = _aggregate([_row(peak=10.0), _row(peak=3.0)], ZERO_OFFSET)
    assert c["peak_mem_mib"] == 10.0   # max, NOT 13.0


def test_offset_adds_to_sums_and_maxes_into_peak():
    offset: CostOffset = {"latency_ms": 1.0, "peak_mem_mib": 7.0,
                          "params": 5, "flops": 50}
    c = _aggregate([_row(mean=2.0, peak=4.0, params=10, flops=100)], offset)
    assert c["latency_ms"] == pytest.approx(3.0)   # 2 + 1
    assert c["params"] == 15                        # 10 + 5
    assert c["flops"] == 150                         # 100 + 50
    assert c["peak_mem_mib"] == 7.0                  # max(4, 7) — not 11


def test_empty_rows_with_zero_offset_is_all_zero():
    assert _aggregate([], ZERO_OFFSET) == {
        "latency_ms": 0.0, "peak_mem_mib": 0.0, "params": 0, "flops": 0}


# ---- cost(): lookup + missing-key -------------------------------------------

def test_cost_sums_over_a_real_arch_against_synthetic_lut():
    """Every block of a real arch resolves -> latency == n_blocks (1.0 each)."""
    arch = _random_arch_dict(random.Random(0))
    blocks = arch_to_blocks(arch)
    lut = {row_key(n, cfg, s): _row(mean=1.0, key=row_key(n, cfg, s))
           for n, cfg, s in blocks}
    assert cost(arch, lut)["latency_ms"] == pytest.approx(float(len(blocks)))


def test_missing_key_raises_costerror_naming_the_block():
    arch = _random_arch_dict(random.Random(1))
    with pytest.raises(CostError) as ei:
        cost(arch, {})            # empty LUT -> first block missing
    assert arch_to_blocks(arch)[0][0] in str(ei.value)   # the block name


def test_stem_head_none_defaults_to_no_offset():
    arch = _random_arch_dict(random.Random(2))
    lut = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch)}
    assert cost(arch, lut) == cost(arch, lut, stem_head=None)


def test_cost_at_640_uses_the_deploy_resolution_keys():
    """Pose costs against the @640 grid: res=640 keys must drive the lookup.

    The @224 LUT (default res) cannot satisfy an @640 arch and vice-versa — the
    resolutions re-key disjointly, so each is costed against its own rows.
    """
    arch = _random_arch_dict(random.Random(8))
    lut_640 = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch, res=640)}
    n_blocks = len(arch_to_blocks(arch, res=640))
    assert cost(arch, lut_640, res=640)["latency_ms"] == pytest.approx(float(n_blocks))
    # the @640 LUT does not cover the @224 keys (disjoint resolutions)
    with pytest.raises(CostError):
        cost(arch, lut_640)  # default res=224


# ---- precision filtering via cost_from_path ---------------------------------

def test_precision_filter_selects_matching_rows(tmp_path):
    arch = _random_arch_dict(random.Random(3))
    keys = list(dict.fromkeys(arch_to_keys(arch)))   # unique: file must not dup-key
    p = tmp_path / "lut.jsonl"
    with open(p, "w") as f:
        for k in keys:
            f.write(json.dumps(_row(mean=1.0, key=k)) + "\n")
    n_blocks = len(arch_to_blocks(arch))
    assert cost_from_path(arch, p, "fp16")["latency_ms"] == pytest.approx(float(n_blocks))
    with pytest.raises(CostError):       # rows are fp16; fp32 filter finds nothing
        cost_from_path(arch, p, "fp32")


# ---- end-to-end against the on-disk LUT (skips while partial) ---------------

def test_roundtrip_on_disk_lut(lut_path):
    """Once the LUT is complete (dummy or full sweep), real archs cost cleanly."""
    with open(lut_path) as f:                     # single-precision file
        precision = json.loads(f.readline()).get("precision")
    lut = load_lut(lut_path, precision=precision)
    try:
        c = cost(_random_arch_dict(random.Random(0)), lut)
    except CostError:
        pytest.skip("LUT partial (real sweep still filling) — runs once complete")
    assert c["latency_ms"] > 0 and c["params"] > 0 and c["peak_mem_mib"] > 0


# ---- stem/head offset: measured stem + head -> a CostOffset -----------------
# The fixed stem (3->16) and head (final-expand/feature-mix/classifier) run
# sequentially like blocks, so the offset aggregates them the same way cost does:
# latency/params/flops SUM, peak_mem MAX (only one resident at a time).

def test_offset_from_measurements_sums_latency_maxes_peak():
    stem = {"latency_ms": 0.30, "peak_mem_mib": 12.0, "params": 500, "flops": 1000}
    head = {"latency_ms": 0.20, "peak_mem_mib": 5.0, "params": 2_600_000, "flops": 9000}
    off = offset_from_measurements(stem, head)
    assert off["latency_ms"] == pytest.approx(0.50)   # 0.30 + 0.20
    assert off["params"] == 2_600_500                  # summed
    assert off["flops"] == 10_000                       # summed
    assert off["peak_mem_mib"] == 12.0                  # max(12, 5), NOT 17


def test_load_stem_head_offset_extracts_four_fields(tmp_path):
    p = tmp_path / "stem_head_offset.json"
    p.write_text(json.dumps({
        "latency_ms": 0.5, "peak_mem_mib": 12.0, "params": 2_600_500, "flops": 10_000,
        "precision": "fp32", "components": {"stem": {}, "head": {}},  # provenance ignored
    }))
    assert load_stem_head_offset(p) == {
        "latency_ms": 0.5, "peak_mem_mib": 12.0, "params": 2_600_500, "flops": 10_000}


def test_offset_roundtrips_and_composes_into_cost(tmp_path):
    stem = {"latency_ms": 0.3, "peak_mem_mib": 12.0, "params": 500, "flops": 1000}
    head = {"latency_ms": 0.2, "peak_mem_mib": 5.0, "params": 2_600_000, "flops": 9000}
    off = offset_from_measurements(stem, head)
    p = tmp_path / "o.json"
    p.write_text(json.dumps({**off, "precision": "fp32"}))
    assert load_stem_head_offset(p) == off
    # the loaded offset shifts a composed cost by exactly its latency
    arch = _random_arch_dict(random.Random(7))
    lut = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch)}
    base = cost(arch, lut)["latency_ms"]
    assert cost(arch, lut, stem_head=off)["latency_ms"] == pytest.approx(base + 0.5)


# ---- resident_mem_mib: deployable memory = resident weights + peak working set --
# CostDict.peak_mem_mib is the measured MAX scratch+IO and EXCLUDES weights
# (schema.md:59); the deployable figure must add every block's resident weights
# back, scaled by the deployment precision's bytes-per-param.

def test_resident_mem_adds_weights_to_peak_working_set():
    c: CostDict = {"latency_ms": 1.0, "peak_mem_mib": 3.0,
                   "params": 2**20, "flops": 0}        # 1 Mi params
    # fp16 (2 bytes/param): 2 MiB weights + 3 MiB working set = 5 MiB.
    assert resident_mem_mib(c, bytes_per_param=2) == pytest.approx(5.0)


def test_resident_mem_scales_with_precision_bytes():
    c: CostDict = {"latency_ms": 1.0, "peak_mem_mib": 3.0,
                   "params": 2**20, "flops": 0}
    # fp32 (4 bytes/param) doubles only the weights term; working set unchanged.
    assert resident_mem_mib(c, bytes_per_param=4) == pytest.approx(7.0)


# ---- latency calibration: measured ~= slope*summed + intercept ----------------
# The additivity fit (search/predictor_stats.py) calibrates the summed backbone
# latency to the device. cost() applies it to the BACKBONE SUM ONLY, before adding
# the stem/head offset (both fit inputs are backbone-only). Default is identity, so
# ranking is untouched; a slope>0 affine map is monotonic => order-preserving.

def test_load_latency_calibration_extracts_two_fields(tmp_path):
    p = tmp_path / "latency_calibration.json"
    p.write_text(json.dumps({
        "slope": 0.92, "intercept": 0.05,
        "r2": 0.99, "mult_factor": 0.93, "n": 33, "precision": "fp32",
        "spearman_rho": 0.98,                      # provenance — must be ignored
    }))
    assert load_latency_calibration(p) == {"slope": 0.92, "intercept": 0.05}


def test_identity_calibration_is_a_noop():
    arch = _random_arch_dict(random.Random(4))
    lut = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch)}
    assert cost(arch, lut, calibration=IDENTITY_CALIBRATION) == cost(arch, lut)


def test_calibration_scales_backbone_then_adds_offset():
    arch = _random_arch_dict(random.Random(5))
    lut = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch)}
    n = len(arch_to_blocks(arch))                  # backbone sum == n (1.0 each)
    cal = {"slope": 0.9, "intercept": 0.0}
    assert cost(arch, lut, calibration=cal)["latency_ms"] == pytest.approx(0.9 * n)
    offset: CostOffset = {"latency_ms": 0.5, "peak_mem_mib": 0.0,
                          "params": 0, "flops": 0}
    # offset is added AFTER calibrating the backbone, not scaled by it
    assert cost(arch, lut, stem_head=offset, calibration=cal)["latency_ms"] == (
        pytest.approx(0.9 * n + 0.5))


def test_calibration_is_affine_so_ranking_preserved():
    arch = _random_arch_dict(random.Random(6))
    lut = {k: _row(mean=1.0, key=k) for k in arch_to_keys(arch)}
    base = cost(arch, lut)["latency_ms"]
    cal = {"slope": 0.9, "intercept": 0.2}
    # latency == slope*base + intercept: an affine, slope>0 (monotonic) transform
    assert cost(arch, lut, calibration=cal)["latency_ms"] == pytest.approx(
        0.9 * base + 0.2)
