"""Tests for eval/shortft.py — the CP 2.4 pose fine-tune harness.

Only the *DoD logic* is exercised here (pure, scipy-only → runs under ``.venv`` / CI):

- :func:`rank_fidelity` — the proxy-rank gate (Kendall-τ ≥ 0.7) that decides whether the
  5-epoch proxy ranking is trustworthy enough to drive the search.
- :func:`reproducible` — the "twice within 0.5 %" precision check.

The actual fine-tune + eval (``short_finetune``) imports ultralytics/ofa lazily and needs a
GPU + dataset, so it is integration-smoked under ``.venv-nas`` (see ``eval/shortft.py``'s
``__main__`` and ``tests/test_grafted_pose_model.py``), not unit-tested here.
"""
import pytest

pytest.importorskip("scipy")  # the rank stats; present in .venv (CP 2.2)

from eval.shortft import KENDALL_TAU_GATE, rank_fidelity, reproducible  # noqa: E402

# --- rank_fidelity: proxy (5-epoch) vs full-train ranking agreement ----------

def test_rank_fidelity_perfect_concordance():
    full = [0.10, 0.20, 0.30, 0.40, 0.50]
    proxy = [0.11, 0.22, 0.29, 0.42, 0.48]  # same order, different values
    rf = rank_fidelity(proxy, full)
    assert rf.kendall_tau == pytest.approx(1.0)
    assert rf.spearman == pytest.approx(1.0)
    assert rf.n == 5
    assert rf.passes is True


def test_rank_fidelity_perfect_discordance_fails_gate():
    full = [0.10, 0.20, 0.30, 0.40]
    proxy = [0.40, 0.30, 0.20, 0.10]  # exactly reversed
    rf = rank_fidelity(proxy, full)
    assert rf.kendall_tau == pytest.approx(-1.0)
    assert rf.passes is False


def test_rank_fidelity_one_swap_is_below_gate():
    # proxy swaps the top two of four: 1 discordant pair / 6 → tau-b = (5-1)/6 ≈ 0.667 < 0.70.
    full = [0.10, 0.20, 0.30, 0.40]
    proxy = [0.10, 0.20, 0.40, 0.30]
    rf = rank_fidelity(proxy, full)
    assert rf.kendall_tau == pytest.approx(2 / 3, abs=1e-9)  # (5-1 concordant-discordant)/6 pairs
    assert rf.passes is False


def test_rank_fidelity_gate_threshold_is_0_7():
    assert KENDALL_TAU_GATE == 0.7


def test_rank_fidelity_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        rank_fidelity([0.1, 0.2, 0.3], [0.1, 0.2])


def test_rank_fidelity_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least"):
        rank_fidelity([0.1], [0.2])


# --- reproducible: the "twice within 0.5 %" precision DoD --------------------

def test_reproducible_within_tolerance():
    assert reproducible(0.500, 0.5039) is True  # 0.39 mAP points apart


def test_reproducible_outside_tolerance():
    assert reproducible(0.500, 0.5101) is False  # 1.01 mAP points apart


def test_reproducible_exactly_equal():
    assert reproducible(0.617, 0.617) is True


def test_reproducible_custom_tolerance():
    assert reproducible(0.50, 0.59, abs_tol=0.10) is True
    assert reproducible(0.50, 0.59, abs_tol=0.05) is False
