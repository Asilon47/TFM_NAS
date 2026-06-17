"""How faithful is the summed-LUT cost predictor? — rank correlation + calibration.

``search.cost.cost`` predicts whole-net latency by *summing* per-block LUT latencies.
The additivity DoD (``search.validate_additivity``) already showed the sum **over-
predicts** by a near-constant ~8% (TensorRT fuses across block seams the per-block LUT
can't see). This module quantifies the predictor two complementary ways, from the
measured-vs-summed pairs in ``data/additivity_subnets.json``:

* **Ranking fidelity** (Spearman rho, Kendall tau-b): does the predictor *order* archs
  the way the device does? This is the metric that matters for search — BO ranks
  thousands of candidates, and a monotone bias leaves the ranking untouched.
* **Absolute calibration** (OLS ``measured = a*summed + b`` via ``scipy.stats.linregress``,
  plus a through-origin "fusion discount" factor): the affine fit that removes the ~8%
  bias so predicted milliseconds match the device — what the Phase-3 objective's
  ``lambda*latency`` term and the latency budget need in absolute units.

scipy.stats gives the correlations (with p-values) and the regression (with standard
errors) directly; only the through-origin factor and the error metrics are plain numpy.
Runs under ``.venv`` (CPU); needs ``scipy`` + ``numpy``, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def spearman(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Spearman rank-correlation rho and its two-sided p-value."""
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def kendall(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Kendall tau-b (tie-corrected) and its two-sided p-value."""
    res = stats.kendalltau(x, y)
    return float(res.statistic), float(res.pvalue)


def pearson(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Pearson linear-correlation r and its two-sided p-value."""
    res = stats.pearsonr(np.asarray(x, float), np.asarray(y, float))
    return float(res.statistic), float(res.pvalue)


@dataclass(frozen=True)
class CalibrationFit:
    """The ``measured ~= slope*summed + intercept`` calibration of the cost predictor.

    ``slope``/``intercept`` (+ their standard errors) and ``r2`` come from an OLS fit
    (``scipy.stats.linregress``). ``mult_factor`` is the separate through-origin least-
    squares factor (``measured = a*summed``) — the single interpretable "fusion discount"
    (expected ~0.92: the device runs ~8% faster than the per-block sum predicts).
    """
    slope: float
    intercept: float
    r2: float
    stderr: float
    intercept_stderr: float
    mult_factor: float


def fit_calibration(summed: ArrayLike, measured: ArrayLike) -> CalibrationFit:
    """OLS ``measured ~ summed`` (+ through-origin factor) — the predictor calibration."""
    x = np.asarray(summed, float)
    y = np.asarray(measured, float)
    lr = stats.linregress(x, y)
    mult = float(np.dot(x, y) / np.dot(x, x))   # least squares through the origin
    return CalibrationFit(
        slope=float(lr.slope),
        intercept=float(lr.intercept),
        r2=float(lr.rvalue) ** 2,
        stderr=float(lr.stderr),
        intercept_stderr=float(lr.intercept_stderr),
        mult_factor=mult,
    )


def error_metrics(predicted: ArrayLike, measured: ArrayLike) -> dict[str, float]:
    """Absolute-accuracy metrics of ``predicted`` against ``measured``.

    ``bias`` is the mean *signed* relative error (positive = over-predict, matching
    ``validate_additivity.relative_error``); ``mape`` is its absolute counterpart;
    ``rmse_ms`` is in the same units as the inputs (milliseconds).
    """
    pred = np.asarray(predicted, float)
    meas = np.asarray(measured, float)
    rel = (pred - meas) / meas
    return {
        "mape": float(np.mean(np.abs(rel))),
        "rmse_ms": float(np.sqrt(np.mean((pred - meas) ** 2))),
        "bias": float(np.mean(rel)),
    }


@dataclass(frozen=True)
class PredictorStats:
    """Full predictor-fidelity summary: ranking, calibration, and error before/after."""
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    fit: CalibrationFit
    mape: float
    rmse_ms: float
    bias: float
    mape_calibrated: float
    rmse_calibrated_ms: float


def predictor_stats(summed: ArrayLike, measured: ArrayLike) -> PredictorStats:
    """Assemble every fidelity statistic for the (summed, measured) latency pairs.

    The *calibrated* error metrics apply the affine fit (``slope*summed + intercept``)
    before comparing to ``measured`` — so a caller can show how much the calibration
    closes the raw ~8% gap.
    """
    x = np.asarray(summed, float)
    y = np.asarray(measured, float)
    r, r_p = pearson(x, y)
    rho, rho_p = spearman(x, y)
    tau, tau_p = kendall(x, y)
    fit = fit_calibration(x, y)
    raw = error_metrics(x, y)
    calibrated = error_metrics(fit.slope * x + fit.intercept, y)
    return PredictorStats(
        n=int(x.size),
        pearson_r=r,
        pearson_p=r_p,
        spearman_rho=rho,
        spearman_p=rho_p,
        kendall_tau=tau,
        kendall_p=tau_p,
        fit=fit,
        mape=raw["mape"],
        rmse_ms=raw["rmse_ms"],
        bias=raw["bias"],
        mape_calibrated=calibrated["mape"],
        rmse_calibrated_ms=calibrated["rmse_ms"],
    )
