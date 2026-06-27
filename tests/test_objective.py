"""D4 contract: the J(α) search objective (search/objective.py).

J(α) = acc_eff − λ·latency, with acc_eff = acc − μ·max(0, resident_mem_mib − budget)².
Every function here is pure (no LUT, no pymoo, no torch), so the whole file runs in CI.
The memory penalty is exercised both directly and through search.cost.resident_mem_mib to
show it is ≡ 0 for the v1 space (subnets are tens of MiB vs the 512 MiB budget).
"""
import pytest

from search.objective import (
    DEFAULT_BUDGET_MIB,
    Anchor,
    effective_accuracy,
    fps_to_ms,
    lambda_from_anchors,
    mem_penalty,
    scalarize,
    within_ceiling,
)

# ---- memory penalty: 0 at/below budget, quadratic above ----

def test_mem_penalty_is_zero_at_and_below_budget():
    assert mem_penalty(100.0, budget=512.0, mu=0.04) == 0.0
    assert mem_penalty(512.0, budget=512.0, mu=0.04) == 0.0  # boundary is inside


def test_mem_penalty_is_quadratic_above_budget():
    # overflow of 2 MiB -> mu*2^2; doubling the overflow quadruples the penalty.
    assert mem_penalty(514.0, budget=512.0, mu=0.5) == pytest.approx(2.0)
    assert mem_penalty(516.0, budget=512.0, mu=0.5) == pytest.approx(8.0)


# ---- effective accuracy: acc minus the (usually zero) memory penalty ----

def test_effective_accuracy_unchanged_below_budget():
    assert effective_accuracy(0.80, 100.0, mu=0.04, budget=512.0) == pytest.approx(0.80)


def test_effective_accuracy_penalized_above_budget():
    # 514 MiB is 2 over a 512 budget: penalty = 0.5*2^2 = 2.0.
    assert effective_accuracy(5.0, 514.0, mu=0.5, budget=512.0) == pytest.approx(3.0)


def test_effective_accuracy_never_binds_for_a_v1_subnet():
    # Drive the resident figure through the real cost helper: a generous ~6M-param
    # fp16 subnet is tens of MiB, far under the default budget -> acc unchanged.
    from search.cost import resident_mem_mib

    cost = {"latency_ms": 3.0, "peak_mem_mib": 40.0, "params": 6_000_000, "flops": 0}
    resident = resident_mem_mib(cost, bytes_per_param=2)  # ~11.4 weight + 40 working set
    assert resident < DEFAULT_BUDGET_MIB
    assert effective_accuracy(0.73, resident, mu=0.04) == pytest.approx(0.73)


# ---- scalarization: the scalar J used for ParEGO + final selection ----

def test_scalarize_strictly_decreases_with_latency():
    slow = scalarize(0.80, 20.0, 0.0, lam=0.01, mu=0.04)
    fast = scalarize(0.80, 10.0, 0.0, lam=0.01, mu=0.04)
    assert fast > slow


def test_scalarize_equals_acc_eff_minus_lambda_latency():
    # acc 5.0, 2 MiB over budget (penalty 2.0 at mu=0.5) -> acc_eff 3.0; minus 0.01*10.
    j = scalarize(5.0, 10.0, 514.0, lam=0.01, mu=0.5, budget=512.0)
    assert j == pytest.approx(3.0 - 0.01 * 10.0)


# ---- hard latency ceiling: the ε-constraint predicate (boundary is inside) ----

def test_within_ceiling_boundary_is_inclusive():
    assert within_ceiling(16.7, 16.7) is True
    assert within_ceiling(16.0, 16.7) is True
    assert within_ceiling(16.71, 16.7) is False


# ---- fps -> latency ceiling ----

def test_fps_to_ms():
    assert fps_to_ms(60) == pytest.approx(1000.0 / 60.0)
    assert fps_to_ms(120) == pytest.approx(1000.0 / 120.0)


def test_fps_to_ms_rejects_nonpositive():
    with pytest.raises(ValueError):
        fps_to_ms(0)


# ---- two-anchor iso-J λ calibration ----

def test_lambda_from_anchors_is_the_two_point_slope():
    # iso-J of two reference models: a slower-but-more-accurate anchor vs a faster one.
    # λ = Δacc/Δlat so both lie on one J contour; a real trade-off -> λ > 0.
    slower = Anchor(acc=0.80, latency_ms=20.0)
    faster = Anchor(acc=0.70, latency_ms=10.0)
    lam = lambda_from_anchors(slower, faster)
    assert lam == pytest.approx((0.80 - 0.70) / (20.0 - 10.0))
    assert lam > 0.0


def test_lambda_from_anchors_is_order_independent():
    a = Anchor(acc=0.80, latency_ms=20.0)
    b = Anchor(acc=0.70, latency_ms=10.0)
    assert lambda_from_anchors(a, b) == pytest.approx(lambda_from_anchors(b, a))


def test_lambda_from_anchors_rejects_equal_latency():
    # equal latency -> no latency axis to trade against -> undefined exchange rate.
    with pytest.raises(ValueError):
        lambda_from_anchors(Anchor(acc=0.80, latency_ms=10.0), Anchor(acc=0.70, latency_ms=10.0))
