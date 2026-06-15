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
    ZERO_OFFSET,
    CostError,
    _aggregate,
    cost,
    cost_from_path,
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
