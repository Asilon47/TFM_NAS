"""CP 2.2 additivity DoD, designed to expose fusion-vs-depth (peer-review R4.2).

The summed LUT tends to *over*-predict full-net latency because TensorRT fuses
across block seams that isolated per-block rows can't see, and that error grows
with depth. A single aggregate "within 15 %" bar (PROJECT_PLAN.md:130) averages
that threat away; this harness bins relative error by depth so the deep regime is
visible and can trip the CP 2.3 residual-correction trigger.

These tests are synthetic — real measured-vs-summed inputs need the full sweep +
full-subnet Jetson runs (deferred). They pin the binning/trigger logic now.
"""
import pytest

from search.validate_additivity import (
    AdditivityPoint,
    relative_error,
    validate_additivity,
)


def _point(depth: int, measured: float, error: float) -> AdditivityPoint:
    """A subnet whose summed prediction is off the measurement by ``error`` (signed)."""
    return AdditivityPoint(depth=depth, measured_ms=measured,
                           summed_ms=measured * (1.0 + error))


def test_relative_error_is_signed_overprediction_positive():
    assert relative_error(measured_ms=10.0, summed_ms=11.0) == pytest.approx(0.10)
    assert relative_error(measured_ms=10.0, summed_ms=9.0) == pytest.approx(-0.10)


def test_depth_growing_residual_trips_bin_but_not_aggregate():
    # Fusion makes the summed LUT over-predict more as depth grows. The mean error
    # stays under the 15 % bar while the deepest bin breaches it — exactly why we
    # bin by depth (R4.2) instead of trusting one aggregate number.
    pts = [_point(depth=2, measured=10.0, error=0.05),
           _point(depth=4, measured=10.0, error=0.10),
           _point(depth=8, measured=10.0, error=0.22)]
    rep = validate_additivity(pts, bar=0.15)
    assert rep.aggregate < 0.15          # a single-number DoD would PASS here
    assert rep.breached                  # ...but binning catches the deep regime
    assert rep.breaching_depths == [8]


def test_clean_additivity_is_not_breached():
    pts = [_point(depth=2, measured=10.0, error=0.03),
           _point(depth=8, measured=10.0, error=0.07)]
    rep = validate_additivity(pts, bar=0.15)
    assert not rep.breached
    assert rep.breaching_depths == []


def test_by_depth_averages_points_in_the_same_bin():
    pts = [_point(depth=4, measured=10.0, error=0.08),
           _point(depth=4, measured=10.0, error=0.12)]
    rep = validate_additivity(pts, bar=0.15)
    assert rep.by_depth[4] == pytest.approx(0.10)   # mean of 0.08 and 0.12
