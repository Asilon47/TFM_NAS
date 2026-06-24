"""Ranking-quality metrics beyond Kendall-τ (CP 2.4 Tier-1B).

`precision_at_k` and `top1_regret` measure what a NAS search actually needs — *did the
ranker surface a near-best architecture?* — which the τ≥0.7 gate does not. The CP 2.4
investigation showed every zero-cost descriptor picks the true-best arch (regret 0) even
when its τ "fails", so these metrics are the fairer success criteria. Pure (no torch/ofa)
→ unit-tested under .venv / CI.
"""

import pytest

from eval.shortft import precision_at_k, top1_regret


def test_precision_at_k_perfect_agreement():
    full = [0.1, 0.2, 0.3, 0.4]
    proxy = [1.0, 2.0, 3.0, 4.0]  # identical order → both top-2 are {2,3}
    assert precision_at_k(proxy, full, k=2) == 1.0


def test_precision_at_k_partial_overlap():
    # full top-2 = {3,2}; proxy top-2 = {3,0}; overlap {3} → 1/2
    full = [0.1, 0.2, 0.3, 0.4]
    proxy = [0.9, 0.0, 0.1, 1.0]
    assert precision_at_k(proxy, full, k=2) == 0.5


def test_precision_at_k_k1_is_top1_hit():
    full = [0.1, 0.5, 0.2]
    proxy = [0.0, 9.0, 1.0]  # proxy best == full best (idx1) → 1.0
    assert precision_at_k(proxy, full, k=1) == 1.0


def test_precision_at_k_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        precision_at_k([1, 2, 3], [1, 2, 3], k=0)
    with pytest.raises(ValueError):
        precision_at_k([1, 2, 3], [1, 2, 3], k=4)


def test_precision_at_k_length_mismatch():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1, 2, 3], k=1)


def test_top1_regret_zero_when_best_picked():
    full = [0.1, 0.4, 0.3]
    proxy = [0.0, 1.0, 0.5]  # argmax → idx1 = true best → no regret
    assert top1_regret(proxy, full) == 0.0


def test_top1_regret_positive_when_suboptimal():
    full = [0.1, 0.4, 0.3]
    proxy = [9.0, 0.0, 0.5]  # argmax → idx0 (full 0.1); best is 0.4 → regret 0.3
    assert top1_regret(proxy, full) == pytest.approx(0.3)


def test_top1_regret_length_mismatch():
    with pytest.raises(ValueError):
        top1_regret([1, 2], [1, 2, 3])
