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
    is_diagnostic,
    normalize_to_percent,
    overall_pass,
    require_imagenet_layout,
    resolve_archs,
    val_sample_size,
    verdict,
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


# --- verdict ------------------------------------------------------------------

def test_verdict_within_bar():
    v = verdict(76.0, 76.5, bar=1.5)
    assert v["gap"] == pytest.approx(-0.5)
    assert v["within_bar"] is True


def test_verdict_outside_bar():
    v = verdict(74.0, 77.0, bar=1.5)
    assert v["abs_gap"] == pytest.approx(3.0)
    assert v["within_bar"] is False


def test_verdict_within_noise_band_when_ci_covers_gap():
    # 2.0pp gap busts the 1.5 bar but is consistent with a ±2.5pp measurement CI.
    v = verdict(76.0, 74.0, bar=1.5, ci=2.5)
    assert v["within_bar"] is False
    assert v["within_noise"] is True


def test_verdict_without_ci_reports_no_noise_band():
    assert verdict(76.0, 76.0, bar=1.5)["within_noise"] is None


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


# --- DoD gate: corners are diagnostic, interior (sampled) archs gate ----------

def test_is_diagnostic_flags_corners_only():
    assert is_diagnostic("max") and is_diagnostic("min")
    assert not is_diagnostic("random")


def test_overall_pass_ignores_diagnostic_corner_failures():
    # The predictor extrapolates at the corners; a corner busting the bar is not a weight
    # bug. With the interior 'sampled' arch inside the bar, the DoD passes.
    results = [
        {"label": "max", "diagnostic": True, "within_bar": False},
        {"label": "random", "diagnostic": False, "within_bar": True},
    ]
    assert overall_pass(results) is True


def test_overall_pass_fails_when_interior_arch_busts_bar():
    results = [
        {"label": "max", "diagnostic": True, "within_bar": True},
        {"label": "random", "diagnostic": False, "within_bar": False},
    ]
    assert overall_pass(results) is False


def test_overall_pass_falls_back_to_corners_when_no_interior():
    assert overall_pass([{"label": "max", "diagnostic": True, "within_bar": False}]) is False
    assert overall_pass([{"label": "min", "diagnostic": True, "within_bar": True}]) is True
