"""D4 — the J(α) search objective: scalarization + memory penalty + λ calibration.

Resolves open decision **D4** (2026-06-27): how Phase-3 search trades accuracy against
latency and memory. The objective is::

    J(α) = acc_eff − λ·latency_ms
    acc_eff = acc − μ·max(0, resident_mem_mib − budget)²

**Search form = Pareto + hard latency ceiling.** CP 3.3 runs *multi-objective* BO over
``(acc_eff, latency_ms)`` bounded by a hard ceiling ``latency_ms ≤ T_max`` (the ε-constraint —
OFA's own "best accuracy under a latency budget" method). The soft μ² memory penalty is *folded
into the accuracy axis* (``acc_eff``) rather than kept as a separate objective, so the search
stays 2-D while honouring the penalty. The scalar ``J`` itself is used twice: as the random-weight
**ParEGO scalarization** that traces the front (reconciling the EI acquisition with the
hypervolume DoD), and as the **final-winner selector** (CP 3.5).

**λ is a selection knob, not a fixed search constant.** ParEGO samples the weight while searching,
so no single λ is committed up front. The deploy winner is picked by calibrating λ from two
reference models on a common iso-J contour (:func:`lambda_from_anchors`) and reporting a
*sensitivity sweep*, never one magic value. The numeric λ is deferred to CP 3.3 because it has
units of accuracy-per-ms and so needs the @640 deploy-latency scale (the owed 640-res LUT
re-sweep + the yolo11n-pose baseline anchor); this module leaves λ/μ as caller arguments and
hard-codes no deferred number.

**The memory term never binds in v1.** OFA-MBv3-w1.0 subnets are ≤24 MiB fp32 (tens of MiB at
fp16) against the 8 GB device, so ``max(0, resident − budget)`` is 0 for every subnet in the v1
space — :func:`mem_penalty` is identically zero here and only becomes active in Phase-7's
expanded space. Keeping it (rather than a hard filter) preserves one uniform ``J(α)`` across
Phases 3 and 7. ``budget`` defaults to a conservative 512 MiB resident reservation on the shared
8 GB; feed it the figure from :func:`search.cost.resident_mem_mib` (weights + peak working set),
not the weight-excluding ``peak_mem_mib`` alone.
"""
from __future__ import annotations

from dataclasses import dataclass

# Provisional, never binds for v1 (subnets are tens of MiB). A conservative resident-memory
# reservation for the model on the Jetson's shared 8 GB; μ is calibrated with λ at CP 3.3.
DEFAULT_BUDGET_MIB: float = 512.0


def fps_to_ms(fps: float) -> float:
    """A target frame rate → the per-frame latency ceiling in ms (``1000 / fps``).

    The FPS-anchored half of ``T_max`` (the other half is the measured baseline latency;
    the effective ceiling is the tighter of the two). Decidable without the Jetson.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return 1000.0 / fps


def mem_penalty(resident_mib: float, budget: float, mu: float) -> float:
    """The soft memory penalty ``μ·max(0, resident_mib − budget)²`` (0 below the budget).

    Quadratic above the budget so the cost ramps steeply once a subnet overflows; exactly
    zero at or below it (the boundary is feasible). Identically zero across the whole v1
    space — see the module docstring.
    """
    overflow = resident_mib - budget
    return mu * overflow * overflow if overflow > 0.0 else 0.0


def effective_accuracy(
    acc: float, resident_mib: float, *, mu: float, budget: float = DEFAULT_BUDGET_MIB
) -> float:
    """Accuracy with the memory penalty folded in: ``acc − μ·max(0, resident − budget)²``.

    This is the accuracy *objective* the Pareto search maximises (so the 2-D front stays
    ``(acc_eff, latency)``); for v1 subnets the penalty is 0 and ``acc_eff == acc``.
    """
    return acc - mem_penalty(resident_mib, budget, mu)


def scalarize(
    acc: float,
    latency_ms: float,
    resident_mib: float,
    *,
    lam: float,
    mu: float,
    budget: float = DEFAULT_BUDGET_MIB,
) -> float:
    """The scalar ``J(α) = acc_eff − λ·latency_ms`` (ParEGO weight + final-winner selector).

    ``lam`` is the accuracy-per-ms exchange rate (see :func:`lambda_from_anchors`); larger
    ``lam`` penalises latency harder. Monotonically decreasing in ``latency_ms`` for ``lam>0``.
    """
    return effective_accuracy(acc, resident_mib, mu=mu, budget=budget) - lam * latency_ms


def within_ceiling(latency_ms: float, t_max: float) -> bool:
    """The hard latency-ceiling predicate (``latency_ms ≤ t_max``); the boundary is feasible.

    ``t_max = min(baseline_latency, fps_to_ms(target_fps))`` — the tighter of the
    "must beat the deployed yolo11n-pose" bar and the frame-rate budget.
    """
    return latency_ms <= t_max


@dataclass(frozen=True)
class Anchor:
    """A reference model as a single (accuracy, latency) point for λ calibration."""

    acc: float
    latency_ms: float


def lambda_from_anchors(a: Anchor, b: Anchor) -> float:
    """λ that places both anchors on one iso-J contour: the two-point slope ``Δacc / Δlat``.

    Setting ``J(a) == J(b)`` in ``J = acc − λ·latency`` gives
    ``λ = (a.acc − b.acc) / (a.latency_ms − b.latency_ms)`` — the accuracy you'd trade for a
    ms, read off the line through two reference models (e.g. MobileNetV3-large vs
    EfficientNet-B0). Order-independent (numerator and denominator flip sign together).

    Design choices: equal latencies give no latency axis to trade against (a vertical line) —
    an undefined exchange rate, so we raise rather than divide by zero. The *signed* slope is
    returned as-is; a sensible trade-off pair (the slower model is the more accurate one)
    yields ``λ > 0``, whereas ``λ ≤ 0`` means one anchor Pareto-dominates the other and the
    pair is not a valid calibration — surfaced to the caller, not silently masked with ``abs``.
    """
    d_lat = a.latency_ms - b.latency_ms
    if d_lat == 0.0:
        raise ValueError(
            "anchors have equal latency; the accuracy/latency exchange rate is undefined"
        )
    return (a.acc - b.acc) / d_lat
