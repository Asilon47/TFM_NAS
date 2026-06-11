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
    source: NotRequired[str]  # "roofline_dummy" on dummy rows; absent on real ones


# A LUT-keyed block as emitted by search.arch_to_blocks:
# (block_name, cfg, input_shape).
Block = tuple[str, dict, tuple]
