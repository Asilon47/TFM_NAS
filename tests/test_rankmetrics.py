"""Ranking-quality metrics beyond Kendall-τ (CP 2.4 Tier-1B).

`precision_at_k` and `top1_regret` measure what a NAS search actually needs — *did the
ranker surface a near-best architecture?* — which the τ≥0.7 gate does not. The CP 2.4
investigation showed every zero-cost descriptor picks the true-best arch (regret 0) even
when its τ "fails", so these metrics are the fairer success criteria. Pure (no torch/ofa)
→ unit-tested under .venv / CI.
"""

import pytest

from eval.shortft import RankVerdict, precision_at_k, rank_verdict, top1_regret


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


# --- rank_verdict: the search-relevant DoD gate (Spearman + top-1 regret) -------------------

def test_rank_verdict_passes_for_strong_ranker():
    full = [0.10, 0.20, 0.30, 0.40, 0.50]
    proxy = [1.0, 2.0, 3.0, 4.0, 5.0]  # identical order → ρ=1, regret 0
    v = rank_verdict(proxy, full, k=2)
    assert isinstance(v, RankVerdict)
    assert v.spearman == pytest.approx(1.0)
    assert v.top1_regret == 0.0
    assert v.n == 5 and v.k == 2
    assert v.passes


def test_rank_verdict_fails_for_noisy_proxy():
    # weak correlation AND picks a poor arch — the CP 2.4 failure shape
    full = [0.10, 0.50, 0.30, 0.40, 0.45]
    proxy = [0.9, 0.1, 0.2, 0.3, 0.4]  # argmax → idx0 (full 0.10): big regret, low ρ
    assert not rank_verdict(proxy, full, k=2).passes


def test_rank_verdict_regret_gate_decisive_when_correlation_passes():
    # ρ=0.8 (passes the 0.70 gate) but the proxy's #1 pick is only 2nd-best → regret 0.10
    full = [0.40, 0.50, 0.60, 0.70]
    proxy = [1.0, 2.0, 4.0, 3.0]
    v = rank_verdict(proxy, full, k=1)
    assert v.spearman == pytest.approx(0.8)
    assert v.top1_regret == pytest.approx(0.10)
    assert not rank_verdict(proxy, full, k=1, regret_tol=0.05).passes  # regret too high
    assert rank_verdict(proxy, full, k=1, regret_tol=0.15).passes      # both gates met


def test_rank_verdict_custom_spearman_gate():
    full = [0.40, 0.50, 0.60, 0.70]
    proxy = [1.0, 2.0, 4.0, 3.0]  # ρ=0.8
    assert not rank_verdict(proxy, full, k=1, spearman_gate=0.9, regret_tol=0.2).passes
    assert rank_verdict(proxy, full, k=1, spearman_gate=0.7, regret_tol=0.2).passes


def test_rank_verdict_length_mismatch():
    with pytest.raises(ValueError):
        rank_verdict([1, 2], [1, 2, 3])
