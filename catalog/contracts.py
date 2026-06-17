"""Static type contracts shared across the pipeline.

These are ``TypedDict``s, NOT dataclasses, deliberately: at runtime every
cfg / arch / LUT row stays a plain dict, because the LUT's ``row_key`` is
``sha1(json.dumps(...))`` over those dicts (catalog/sweep.py) and changing
the runtime representation — or value *types* (bool ``se=True`` vs int
``1``) — silently re-keys the append-only LUT. Type-checkers see the
contract; the wire format never changes.

This module must stay free of third-party imports (no torch/numpy): it is
shared by both venvs and imported from leaf modules — keep it cycle-proof.
"""
from typing import NotRequired, TypedDict


class MBConvCfg(TypedDict):
    """One mbconv LUT grid entry / one searchable OFA block position."""
    in_c: int
    out_c: int
    kernel: int
    stride: int
    expand: int
    se: bool
    res: int


class ArchDict(TypedDict):
    """Canonical OFA arch spec.

    ``ks``/``e`` have one slot per block position (``5 * MAX_DEPTH``);
    ``d`` has one active-depth entry per stage. See catalog/ofa_mbv3.py.
    """
    ks: list[int]
    e: list[int]
    d: list[int]


class LatencyStats(TypedDict):
    mean: float
    std: float
    p50: float
    p95: float
    n: int


class LutRow(TypedDict):
    """One measured (or dummy) LUT row — mirrors lut/docs/schema.md (v1)."""
    row_key: str
    block: str
    cfg: dict
    input_shape: list[int]
    precision: str
    latency_ms: LatencyStats
    peak_mem_mib: float
    params: int
    flops: int
    achieved_bw_gbps: float
    trt_version: str | None
    power_mode: str | None
    jetpack: str | None
    timestamp: str
    # "jetson_trt" on measured rows, "roofline_dummy" on dummy rows; absent
    # on rows written before 2026-06-12 (treat absent as real).
    source: NotRequired[str]
    # True when the sweep preflight verified jetson_clocks at measurement
    # time; None/absent when unknown (--skip-preflight, older rows).
    clocks_locked: NotRequired[bool | None]


# A LUT-keyed block as emitted by search.arch_to_blocks:
# (block_name, cfg, input_shape).
Block = tuple[str, dict, tuple]


class CostDict(TypedDict):
    """Predicted cost of a whole subnet, composed from per-block LUT rows (CP 2.2).

    Aggregation is deliberately *heterogeneous* (see search/cost.py): latency,
    params and flops are summed (additive across a sequential network);
    ``peak_mem_mib`` is the **max** over blocks, never the sum — blocks run one
    at a time and free their scratch, so summing would massively overestimate
    (lut/docs/schema.md: peak_mem is per-block scratch+IO, not additive).

    ``peak_mem_mib`` here keeps the LutRow meaning: the peak single-block working
    set, which **excludes resident weights**. The deployable memory figure is
    ``sum(weights) + max working set``; compose it with
    ``search.cost.resident_mem_mib(cost, bytes_per_param)``, not from this field
    alone (which would undercount by the often-dominant weight bytes).
    """
    latency_ms: float
    peak_mem_mib: float
    params: int
    flops: int


class CostOffset(TypedDict):
    """A constant cost-shaped delta added to every subnet's composed cost.

    Holds the fixed, non-searchable stem (3->16) + head (final-expand,
    feature-mix) contribution: identical for every arch in the OFA-MBv3-w1.0
    space, so it never changes arch *ranking* — only absolute cost. Sourced
    later (CP 2.2 offset is parameterized, default no-op); see search/cost.py.
    """
    latency_ms: float
    peak_mem_mib: float
    params: int
    flops: int


class LatencyCalibration(TypedDict):
    """Affine map from summed-LUT backbone latency to measured device latency.

    ``measured ~= slope*summed + intercept``, fitted on the additivity subnets
    (search/predictor_stats.py) to remove the ~8% over-prediction TensorRT fusion
    causes. Applied to the **backbone sum only** (before the stem/head offset).
    A slope>0 affine map is monotonic, so it shifts *absolute* cost without ever
    changing arch *ranking*; the default (slope 1, intercept 0) is a no-op.
    """
    slope: float
    intercept: float
