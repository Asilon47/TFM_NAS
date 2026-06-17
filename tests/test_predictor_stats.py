"""Predictor-fidelity statistics: rank correlation + regression calibration.

These wrap ``scipy.stats``; the tests pin the *behaviour we rely on* (perfect monotone
-> rho = tau = 1, an exact line -> slope/intercept/R^2 recovered, ranking-invariance
under a positive affine transform), not scipy's internals. ``importorskip`` keeps the
suite green in a degraded env even though scipy is a declared dependency.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from search.predictor_stats import (  # noqa: E402
    error_metrics,
    fit_calibration,
    kendall,
    pearson,
    predictor_stats,
    spearman,
)


def test_spearman_kendall_perfect_monotone_is_one():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 31.0, 42.0, 99.0]   # strictly increasing but non-linear
    assert spearman(x, y)[0] == pytest.approx(1.0)
    assert kendall(x, y)[0] == pytest.approx(1.0)


def test_spearman_kendall_reversed_is_minus_one():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    assert spearman(x, y)[0] == pytest.approx(-1.0)
    assert kendall(x, y)[0] == pytest.approx(-1.0)


def test_pearson_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.0 * x + rng.normal(scale=0.3, size=50)
    assert pearson(x, y)[0] == pytest.approx(float(np.corrcoef(x, y)[0, 1]))


def test_fit_calibration_recovers_exact_line():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0 * xi + 1.0 for xi in x]     # y = 2x + 1
    fit = fit_calibration(x, y)
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r2 == pytest.approx(1.0)


def test_fit_calibration_through_origin_factor():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0 * xi for xi in x]            # y = 2x exactly -> mult_factor == 2
    assert fit_calibration(x, y).mult_factor == pytest.approx(2.0)


def test_error_metrics_on_known_overprediction():
    meas = np.array([2.0, 3.0, 4.0, 5.0])
    summ = meas * 1.10                    # summed is exactly 10% above measured
    m = error_metrics(summ, meas)
    assert m["mape"] == pytest.approx(0.10)
    assert m["bias"] == pytest.approx(0.10)     # signed; positive = over-predict
    assert m["rmse_ms"] == pytest.approx(float(np.sqrt(np.mean((summ - meas) ** 2))))


def test_pvalue_small_for_strong_correlation():
    rng = np.random.default_rng(1)
    x = np.arange(40.0)
    y = x + rng.normal(scale=0.5, size=40)
    assert spearman(x, y)[1] < 1e-6
    assert pearson(x, y)[1] < 1e-6


def test_pvalue_one_for_zero_correlation():
    x = [-2.0, -1.0, 1.0, 2.0]
    y = [1.0, -1.0, -1.0, 1.0]            # constructed so Pearson r == 0 exactly
    r, p = pearson(x, y)
    assert r == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_calibration_is_ranking_invariant():
    # a positive affine transform must not change the Spearman ranking
    rng = np.random.default_rng(2)
    x = rng.normal(size=30)
    y = rng.normal(size=30)
    assert spearman(3.0 * x + 7.0, y)[0] == pytest.approx(spearman(x, y)[0])


def test_predictor_stats_calibration_reduces_error():
    # measured = 0.9*summed (a clean over-prediction); the affine fit is exact here,
    # so calibration drives MAPE -> 0 while the ranking stays perfect.
    summ = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    meas = 0.9 * summ
    s = predictor_stats(summ, meas)
    assert s.n == 6
    assert s.spearman_rho == pytest.approx(1.0)
    assert s.mape == pytest.approx(1.0 / 0.9 - 1.0)   # raw over-predicts by 1/9
    assert s.fit.slope == pytest.approx(0.9)
    assert s.mape_calibrated < 1e-9
