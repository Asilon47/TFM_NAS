"""CP 2.2 — LUT composite-cost function.

Turn a sampled OFA subnet into a *predicted* cost by composing the per-block
numbers already measured in the latency LUT — so Phase-3 search can rank
thousands of candidates without ever touching the Jetson::

    lut = load_lut(Path("data/lut.jsonl"), precision="fp16")
    c = cost(arch_dict, lut)            # {"latency_ms", "peak_mem_mib", "params", "flops"}

The cost model is **additive with one twist**. Across a sequential network:

* ``latency_ms`` / ``params`` / ``flops`` are **summed** — total runtime,
  weight count and compute are the sum of the parts.
* ``peak_mem_mib`` is the **max**, never the sum. Blocks execute one at a time
  and free their scratch + IO between them, so the resident working set is the
  largest single block's, not the running total (lut/docs/schema.md: peak_mem
  is per-block scratch+IO and is explicitly *not* additive). Summing it would
  overestimate by ~20x.

**Stem/head offset.** ``arch_to_blocks`` emits only the searchable MBConv
backbone; the fixed stem (3->16) and head (final-expand / feature-mix) convs are
a constant offset added separately via ``stem_head``. That offset is identical
for every arch in the OFA-MBv3-w1.0 space (the last stage always outputs 160 ch,
input is fixed at 224), so it never changes arch *ranking* — only absolute cost,
which the measured-vs-summed DoD (deferred until the real fp32 sweep) needs.
CP 2.2 ships it parameterized and defaulting to a no-op; the measured numbers
slot in later without touching this interface.

**Precision.** A subnet is costed against one precision (the dummy LUT is fp16,
the real Jetson sweep is fp32). ``cost`` takes a pre-loaded, precision-filtered
``lut`` dict (load once, cost many); ``cost_from_path`` is the load-then-cost
convenience for scripts and the DoD check.

**Limitation — precision is a validity boundary, not just a filter.** Block latency
*rankings* are not precision-invariant: the subnet that is fastest under the
fp32/TF32 LUT need not be fastest at FP16/INT8 (peer-review R4.3). So a search result
is faithful **at the searched precision** only — re-targeting another precision is a
re-sweep *and* a re-search, not merely a re-keying of the table. ``peak_mem_mib`` is
likewise the measured working set excluding weights; for a deployable memory figure
use :func:`resident_mem_mib` (weights are precision-scaled too).

DoD (PROJECT_PLAN.md:130): measured vs. summed latency for 5 random full subnets
within 15 %. That validation needs the full real LUT + a full-subnet Jetson
measurement, so it is deferred; the cost API and its unit tests run now.

    python -m search.cost      # smoke demo against data/lut.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from catalog.contracts import ArchDict, CostDict, CostOffset, LatencyCalibration, LutRow
from catalog.sweep import row_key
from lut.loader import load_lut
from search.arch_to_blocks import arch_to_blocks

ROOT = Path(__file__).resolve().parents[1]

# A no-op offset: max(block_peak, 0.0) == block_peak and +0 leaves sums
# unchanged, so the default-no-op stem/head behaves correctly with no special
# casing in the reduce.
ZERO_OFFSET: CostOffset = {
    "latency_ms": 0.0, "peak_mem_mib": 0.0, "params": 0, "flops": 0,
}

# Identity calibration: measured = 1.0*summed + 0.0 leaves latency unchanged, so the
# default composed cost is the raw summed-LUT prediction (correct for *ranking*). A
# measured fit (data/latency_calibration.json) slots in for absolute end-to-end latency.
IDENTITY_CALIBRATION: LatencyCalibration = {"slope": 1.0, "intercept": 0.0}


class CostError(Exception):
    """A subnet references a block with no row in the (precision-filtered) LUT.

    Expected when costing against a partial real LUT mid-sweep; never expected
    against the complete dummy LUT. Raised loudly rather than skipped — a
    silently-undercounted cost would quietly corrupt the search ranking.
    """


def _aggregate(rows: list[LutRow], offset: CostOffset,
               calibration: LatencyCalibration = IDENTITY_CALIBRATION) -> CostDict:
    """Reduce per-block LUT rows + a constant offset into one subnet cost.

    This is the heart of the cost model — the aggregation rule differs per
    field, and getting ``peak_mem_mib`` wrong (summing it) is the classic
    mistake. Combine ``rows`` (each a ``LutRow``) with ``offset`` (a
    ``CostOffset``) and ``calibration`` (a ``LatencyCalibration``) into a
    ``CostDict`` so that:

    * ``latency_ms``  = ``calibration`` applied to the summed backbone latency
      (``slope * sum(rows) + intercept``) plus the offset's ``latency_ms``.
      Identity calibration (the default) leaves the plain additive sum;
      calibration scales the **backbone only**, never the offset.
    * ``params``      = sum of every row's ``params``,  plus offset ``params``
      (all weights are resident — additive).
    * ``flops``       = sum of every row's ``flops``,   plus offset ``flops``
      (total compute is additive).
    * ``peak_mem_mib`` = the **maximum** over every row's ``peak_mem_mib`` *and*
      the offset's ``peak_mem_mib`` — NOT the sum. Only one block's working set
      is resident at a time, and the stem/head run sequentially too, so the
      offset's memory is maxed in rather than added.

    Edge case to handle: ``rows`` may be empty (no backbone blocks). Folding the
    offset into the peak via ``max`` keeps that case well-defined (the offset is
    always in the running) instead of ``max([])`` blowing up.

    """
    return {
        "latency_ms": (calibration["slope"] * sum(r["latency_ms"]["mean"] for r in rows)
                       + calibration["intercept"] + offset["latency_ms"]),
        # MAX, not sum: only one block's working set is resident at a time, and
        # the stem/head (offset) run sequentially too — folding the offset into
        # the max also keeps empty `rows` well-defined (offset always present).
        "peak_mem_mib": max([r["peak_mem_mib"] for r in rows] + [offset["peak_mem_mib"]]),
        "params": sum(r["params"] for r in rows) + offset["params"],
        "flops": sum(r["flops"] for r in rows) + offset["flops"],
    }


def cost(arch_dict: ArchDict, lut: dict[str, LutRow], *,
         res: int = 224,
         stem_head: CostOffset | None = None,
         calibration: LatencyCalibration | None = None) -> CostDict:
    """Composite cost of ``arch_dict`` from a pre-loaded, precision-filtered LUT.

    Args:
        arch_dict: canonical OFA arch spec (see catalog/contracts.ArchDict).
        lut: ``row_key -> LutRow``, e.g. ``load_lut(path, precision="fp16")``.
        res: network input resolution keying the per-block lookups — 224 (default,
            the CP 3.2 ImageNet grid) or 640 (the D1 pose deploy grid). The LUT
            passed in must hold rows at this resolution (the @640 rows come from
            the owed deploy-resolution sweep); a mismatch raises ``CostError``.
        stem_head: constant stem+head offset to add; ``None`` -> no-op
            (``ZERO_OFFSET``). The default is correct for *ranking*; supply a
            measured offset for absolute-cost reporting.
        calibration: affine fit applied to the summed backbone latency
            (``slope*summed + intercept``); ``None`` -> identity (no-op). Like
            ``stem_head`` it never changes *ranking* (slope>0 is monotonic) —
            supply a measured fit (``load_latency_calibration``) for device-
            accurate absolute latency.

    Raises:
        CostError: a block in the subnet has no row in ``lut`` (names the block).
    """
    offset = ZERO_OFFSET if stem_head is None else stem_head
    calib = IDENTITY_CALIBRATION if calibration is None else calibration
    rows: list[LutRow] = []
    for name, cfg, shape in arch_to_blocks(arch_dict, res):
        key = row_key(name, cfg, shape)
        try:
            rows.append(lut[key])
        except KeyError:
            raise CostError(
                f"no LUT row for block {name} cfg={cfg} shape={tuple(shape)} "
                f"(key={key}). The precision-filtered LUT is missing this entry: "
                "if the real sweep is still partial, finish it "
                "(`python -m lut.orchestrate.run_sweep`); otherwise regenerate "
                "the dummy (`python -m lut.orchestrate.gen_dummy_lut`)."
            ) from None
    return _aggregate(rows, offset, calib)


def cost_from_path(arch_dict: ArchDict, lut_path: Path, precision: str | None = None,
                   *, res: int = 224, stem_head: CostOffset | None = None,
                   calibration: LatencyCalibration | None = None) -> CostDict:
    """Load the LUT from ``lut_path`` (filtered to ``precision``) then ``cost``.

    Convenience for scripts and the additivity DoD. The search loop should call
    :func:`cost` with a once-loaded dict instead of re-reading the file per arch.
    """
    lut = load_lut(lut_path, precision=precision)
    return cost(arch_dict, lut, res=res, stem_head=stem_head, calibration=calibration)


def resident_mem_mib(cost_dict: CostDict, bytes_per_param: int) -> float:
    """Deployable resident memory: all weights + the peak single-block working set.

    ``CostDict.peak_mem_mib`` is the *max* per-block scratch+IO (what the LUT
    measures) and deliberately **excludes weights** (lut/docs/schema.md:53-62). But
    every block's weights stay resident for the whole forward pass, so the memory a
    deployment actually needs is::

        sum(weights) + max_i(scratch_i + io_i)   ==   params*bytes + peak_mem_mib

    ``bytes_per_param`` is the deployment precision's width (fp16 -> 2, fp32 -> 4),
    passed explicitly so the precision assumption never hides inside the reduce.

    This is the figure the Phase-3 objective's ``mu * max(0, m - budget)**2`` term
    constrains; ``peak_mem_mib`` alone undercounts by the (often dominant) weight
    bytes — for a 5-8 M-param subnet at fp16 that is ~10-16 MiB. The composed
    estimate is validated against a measured whole-net peak on the same subnets as
    the latency additivity check (deferred — needs the Jetson); see
    search/validate_additivity.py.
    """
    return cost_dict["params"] * bytes_per_param / 2**20 + cost_dict["peak_mem_mib"]


# --- stem/head offset calibration (CP 2.2) -----------------------------------
# arch_to_blocks omits the fixed stem (3->16) and head (final-expand/feature-mix/
# classifier); they are a constant CostOffset. ZERO_OFFSET is correct for ranking
# (the offset is identical for every arch); these two helpers measure and load the
# absolute value for end-to-end latency reporting. See search/export_subnet.py for
# the stem/head modules and lut/orchestrate/measure_additivity.py for the on-device
# measurement that writes the offset JSON.

DEFAULT_OFFSET_PATH = ROOT / "data" / "stem_head_offset.json"


def offset_from_measurements(stem: dict, head: dict) -> CostOffset:
    """Combine measured stem + head component costs into a ``CostOffset``.

    Same heterogeneous reduce as the block cost (``_aggregate``): latency / params /
    flops SUM, ``peak_mem_mib`` MAX — the stem and head run sequentially, so only one
    working set is resident at a time. Each input is a dict with ``latency_ms``
    (float, the measured mean), ``peak_mem_mib``, ``params`` and ``flops``.
    """
    return {
        "latency_ms": stem["latency_ms"] + head["latency_ms"],
        "peak_mem_mib": max(stem["peak_mem_mib"], head["peak_mem_mib"]),
        "params": stem["params"] + head["params"],
        "flops": stem["flops"] + head["flops"],
    }


def load_stem_head_offset(path: Path = DEFAULT_OFFSET_PATH) -> CostOffset:
    """Load a measured stem/head offset as a ``CostOffset`` (opt-in absolute cost).

    Reads only the four ``CostOffset`` fields; any provenance keys the measurement
    wrote alongside (``precision``, ``components``, ...) are ignored. Pass the result
    as ``cost(arch, lut, stem_head=...)`` for absolute end-to-end cost; omit it (the
    default ``ZERO_OFFSET``) for ranking, which the offset never changes.
    """
    raw = json.loads(Path(path).read_text())
    return {
        "latency_ms": float(raw["latency_ms"]),
        "peak_mem_mib": float(raw["peak_mem_mib"]),
        "params": int(raw["params"]),
        "flops": int(raw["flops"]),
    }


# --- latency calibration (CP 2.2 additivity) ---------------------------------
# arch_to_blocks' summed latency over-predicts the measured whole-net latency by a
# near-constant ~8% (TensorRT cross-seam fusion; see search/validate_additivity.py and
# search/predictor_stats.py). load_latency_calibration loads the affine fit that corrects
# it; cost(..., calibration=...) applies it to the backbone sum. The default
# IDENTITY_CALIBRATION is ranking-neutral, so search ranking is unaffected either way.

DEFAULT_CALIBRATION_PATH = ROOT / "data" / "latency_calibration.json"


def load_latency_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> LatencyCalibration:
    """Load a measured latency calibration as a ``LatencyCalibration``.

    Reads only ``slope``/``intercept``; any provenance the fit wrote alongside (``r2``,
    ``mult_factor``, ``spearman_rho``, ``precision``, ...) is ignored. Pass the result as
    ``cost(arch, lut, calibration=...)`` for device-accurate absolute latency; omit it
    (the default ``IDENTITY_CALIBRATION``) for ranking, which the calibration never
    changes.
    """
    raw = json.loads(Path(path).read_text())
    return {"slope": float(raw["slope"]), "intercept": float(raw["intercept"])}


if __name__ == "__main__":
    import random

    from search.arch_to_blocks import _random_arch_dict

    lut_file = ROOT / "data" / "lut.jsonl"
    # Match the precision present in the file: dummy=fp16, real sweep=fp32.
    # Try fp16 first (the dummy), then fp32 (a real sweep); a partial file
    # simply yields missing-key CostErrors we report as coverage.
    print(f"Costing random archs against {lut_file}")
    rng = random.Random(0)
    for precision in ("fp16", "fp32"):
        try:
            n_ok = 0
            for _ in range(5):
                arch = _random_arch_dict(rng)
                try:
                    c = cost_from_path(arch, lut_file, precision)
                except CostError as e:
                    print(f"  [{precision}] missing: {e}")
                    continue
                n_ok += 1
                print(f"  [{precision}] {c}")
            print(f"  [{precision}] {n_ok}/5 archs fully covered")
        except FileNotFoundError as e:
            print(e)
            break
