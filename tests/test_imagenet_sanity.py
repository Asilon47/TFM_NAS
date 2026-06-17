"""CP 1.4 ImageNet-sanity harness tests.

The *pure* decision layer (arch construction, predictor-scale normalization,
binomial CI, pass/fail verdict) is deterministic and runs under ``.venv`` with no
``ofa``/``torchvision``/ImageNet — that is what these tests cover. The OFA + ImageNet
orchestration (``measure_topk`` / ``predict_topk`` / ``run_sanity``) needs ``.venv-nas``
and a real dataset, so it is exercised by the runbook, not in CI (mirrors how the
ofa-dependent and LUT-file tests skip when their inputs are absent).
"""
import math

import pytest

from catalog.ofa_mbv3 import MAX_DEPTH
from eval.imagenet_sanity import (
    binomial_ci95,
    build_net_config,
    canonical_archs,
    normalize_to_percent,
    random_archs,
    rank_pass,
    rank_summary,
    require_imagenet_layout,
    resolve_archs,
    val_sample_size,
)

N_SLOTS = 5 * MAX_DEPTH  # 20 ks/e slots, 5 depth entries


# --- canonical / resolved archs ----------------------------------------------

def test_canonical_archs_are_the_space_corners():
    archs = canonical_archs()
    assert archs["max"] == {"ks": [7] * N_SLOTS, "e": [6] * N_SLOTS, "d": [4] * 5}
    assert archs["min"] == {"ks": [3] * N_SLOTS, "e": [3] * N_SLOTS, "d": [2] * 5}


def test_resolve_archs_preserves_order_and_labels():
    resolved = resolve_archs(["max", "min"], seed=0)
    assert [label for label, _ in resolved] == ["max", "min"]
    assert resolved[0][1] == canonical_archs()["max"]


def test_resolve_archs_random_is_seed_deterministic_and_well_shaped():
    a = resolve_archs(["random"], seed=0)
    b = resolve_archs(["random"], seed=0)
    assert a == b                                          # same seed -> same arch
    (_, arch), = a
    assert len(arch["ks"]) == len(arch["e"]) == N_SLOTS
    assert len(arch["d"]) == 5
    assert resolve_archs(["random"], seed=1) != a          # different seed -> differs


def test_resolve_archs_rejects_unknown_label():
    with pytest.raises(ValueError, match="nope"):
        resolve_archs(["nope"], seed=0)


# --- build_net_config (inject resolution, enforce OFA's length contract) ------

def test_build_net_config_injects_resolution():
    cfg = build_net_config({"ks": [7] * N_SLOTS, "e": [6] * N_SLOTS, "d": [4] * 5}, 224)
    assert cfg["r"] == [224]
    assert cfg["ks"] == [7] * N_SLOTS and cfg["d"] == [4] * 5   # arch fields preserved


def test_build_net_config_rejects_wrong_lengths():
    with pytest.raises(ValueError, match="20"):
        build_net_config({"ks": [7] * 19, "e": [6] * N_SLOTS, "d": [4] * 5}, 224)
    with pytest.raises(ValueError, match="5"):
        build_net_config({"ks": [7] * N_SLOTS, "e": [6] * N_SLOTS, "d": [4] * 4}, 224)


# --- predictor-scale normalization -------------------------------------------

def test_normalize_to_percent_scales_fraction_but_leaves_percent():
    assert normalize_to_percent(0.766) == pytest.approx(76.6)   # fraction -> percent
    assert normalize_to_percent(76.6) == pytest.approx(76.6)    # already percent
    assert normalize_to_percent(1.0) == pytest.approx(100.0)    # boundary = fraction


# --- binomial 95% CI on a measured top-1 -------------------------------------

def test_binomial_ci95_matches_closed_form():
    # p=0.5, n=100 -> 1.96 * sqrt(.25/100) * 100 = 9.8 pp
    assert binomial_ci95(50.0, 100) == pytest.approx(9.8, abs=1e-6)


def test_binomial_ci95_shrinks_with_n():
    assert binomial_ci95(77.0, 10_000) < binomial_ci95(77.0, 2_000)


def test_binomial_ci95_is_inf_without_samples():
    assert math.isinf(binomial_ci95(77.0, 0))


# --- ImageNet layout validation (loud failure, no silent bogus number) -------

def test_require_imagenet_layout_accepts_train_and_val(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    assert require_imagenet_layout(tmp_path) == tmp_path


def test_require_imagenet_layout_rejects_missing_val(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="val"):
        require_imagenet_layout(tmp_path)


def test_require_imagenet_layout_rejects_missing_train(tmp_path):
    (tmp_path / "val").mkdir()
    with pytest.raises(FileNotFoundError, match="train"):
        require_imagenet_layout(tmp_path)


# --- val-subset sizing (feeds the binomial CI) -------------------------------

def test_val_sample_size_zero_means_all():
    assert val_sample_size(0, 5000) == 5000
    assert val_sample_size(-1, 5000) == 5000      # guard against negatives


def test_val_sample_size_caps_request_at_available():
    assert val_sample_size(2000, 5000) == 2000
    assert val_sample_size(10_000, 5000) == 5000  # never more than exist


# --- rank-fidelity: a set of archs + the Spearman gate ------------------------
# The accuracy predictor is a *ranking* model (constant ~6.3pp absolute offset, verified on
# the first run); CP 1.4 therefore gates on rank correlation across a spread of archs, with
# the OLS affine fit reported as scale evidence. These layers stay .venv-pure.

def test_random_archs_are_distinct_seed_deterministic_and_well_shaped():
    a = random_archs(5, seed=0)
    assert [label for label, _ in a] == ["rand0", "rand1", "rand2", "rand3", "rand4"]
    for _, arch in a:
        assert len(arch["ks"]) == len(arch["e"]) == N_SLOTS
        assert len(arch["d"]) == 5
    archs = [arch for _, arch in a]
    assert any(arch != archs[0] for arch in archs[1:])     # distinct draws (seed + i)
    assert random_archs(5, seed=0) == a                    # reproducible for a fixed seed
    assert random_archs(5, seed=1) != a                    # different seed -> different set


def test_random_archs_first_draw_matches_base_seed():
    # rand0 is exactly random_arch_dict(Random(seed)) — the same draw resolve_archs('random')
    # makes, so a single-random run and the rank run agree on their first interior arch.
    import random as _random

    from search.arch_to_blocks import random_arch_dict
    (_, first), = random_archs(1, seed=7)
    assert first == random_arch_dict(_random.Random(7))


def test_rank_pass_threshold_boundary():
    assert rank_pass(0.90, threshold=0.85) is True
    assert rank_pass(0.85, threshold=0.85) is True         # boundary is inclusive
    assert rank_pass(0.80, threshold=0.85) is False


def test_rank_pass_nan_is_failure():
    assert rank_pass(float("nan"), threshold=0.85) is False


def test_rank_summary_affine_with_constant_offset_passes():
    # measured = predicted - 6.3 exactly: perfect rank fidelity AND a clean affine fit
    # (slope 1, intercept -6.3) — the structure the real predictor showed by hand.
    predicted = [80.0, 78.0, 83.0, 77.0, 81.0, 75.0]
    measured = [p - 6.3 for p in predicted]
    s = rank_summary(measured, predicted, threshold=0.85)
    assert s["spearman_rho"] == pytest.approx(1.0)
    assert s["slope"] == pytest.approx(1.0, abs=1e-6)
    assert s["intercept"] == pytest.approx(-6.3, abs=1e-6)
    assert s["r2"] == pytest.approx(1.0, abs=1e-6)
    assert s["mape_calibrated"] < s["mape"]                # calibration closes the offset
    assert s["passed"] is True


def test_rank_summary_rank_reversed_fails_the_gate():
    predicted = [80.0, 78.0, 83.0, 77.0, 81.0, 75.0]
    measured = [-p for p in predicted]                     # perfectly anti-correlated
    s = rank_summary(measured, predicted, threshold=0.85)
    assert s["spearman_rho"] == pytest.approx(-1.0)
    assert s["passed"] is False
