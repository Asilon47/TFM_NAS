"""CP 2.2 additivity DoD harness — measured-vs-summed latency, binned by depth.

The composite cost (search/cost.py) *sums* per-block LUT latencies. Reviewer 4 of
the peer review (peer_review_simulation.md, R4.2) flags the load-bearing risk:
TensorRT fuses across block seams that blocks-timed-in-isolation cannot see, so a
summed LUT tends to **over-predict** whole-net latency, and the error **grows with
depth**. A single aggregate "within 15 %" check (PROJECT_PLAN.md:130) would average
that threat away.

This harness reports relative error ``(summed - measured)/measured`` **binned by
depth** so the deep regime is visible, and flags whether any bin breaches the bar.
That bin-level breach (not just an aggregate miss) is the pre-registered trigger for
CP 2.3 (residual correction) — see PROJECT_PLAN.md "CP 2.3".

Scope: the *logic* is tested now on synthetic points
(tests/test_validate_additivity.py). The real inputs — ``measured_ms`` from
full-subnet Jetson runs and ``summed_ms`` from ``search.cost.cost(arch, lut)`` — need
the full sweep + on-device measurement and are wired post-sweep. Build one
``AdditivityPoint`` per subnet (``depth`` = number of LUT blocks) and pass them here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AdditivityPoint:
    """One subnet's measured-vs-summed latency. ``depth`` = number of LUT blocks."""
    depth: int
    measured_ms: float
    summed_ms: float


@dataclass(frozen=True)
class AdditivityReport:
    """Relative-error breakdown of the additive cost model against measurement.

    ``by_depth`` maps depth -> mean signed relative error in that bin (positive =
    the summed LUT over-predicts, the expected fusion signature). ``aggregate`` is
    the mean over all points — the single number a naive DoD would report.
    ``breached`` / ``breaching_depths`` flag bins whose |mean error| exceeds ``bar``.
    """
    by_depth: dict[int, float]
    aggregate: float
    bar: float
    breached: bool
    breaching_depths: list[int]


def relative_error(measured_ms: float, summed_ms: float) -> float:
    """Signed relative error; positive when the summed LUT over-predicts measurement."""
    return (summed_ms - measured_ms) / measured_ms


def validate_additivity(
    points: Sequence[AdditivityPoint], *, bar: float = 0.15
) -> AdditivityReport:
    """Bin measured-vs-summed error by depth and flag any bin breaching ``bar``.

    Binning is the point: it exposes the fusion-grows-with-depth threat (R4.2) that
    a single aggregate would hide. A breach in any depth bin pre-registers the jump
    to CP 2.3 (residual correction), even when the aggregate passes.
    """
    errors = [relative_error(p.measured_ms, p.summed_ms) for p in points]

    by_depth_samples: dict[int, list[float]] = {}
    for p, e in zip(points, errors, strict=True):
        by_depth_samples.setdefault(p.depth, []).append(e)
    by_depth = {d: sum(es) / len(es) for d, es in by_depth_samples.items()}

    aggregate = sum(errors) / len(errors)
    breaching_depths = sorted(d for d, e in by_depth.items() if abs(e) > bar)
    return AdditivityReport(
        by_depth=by_depth,
        aggregate=aggregate,
        bar=bar,
        breached=bool(breaching_depths),
        breaching_depths=breaching_depths,
    )
