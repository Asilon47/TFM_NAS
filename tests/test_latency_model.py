"""Track 2 — search/latency_model.py: the fit math (pure) + ONNX feature extraction."""
import json

import numpy as np
import pytest

from search.latency_model import (
    FEATURES,
    PHYSICAL_GRAFT_FP16,
    PHYSICAL_GRAFT_FP32,
    ROOT,
    act_limit_for_ms,
    collect_points,
    fit_physical,
    fit_ridge,
    is_graft_e2e_point,
    loo_mape,
    predict_ms,
    resolve_onnx,
)


def _linear_points(n=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 20, size=(n, len(FEATURES)))
    w_true = np.array([0.5, 0.2, 0.05, 0.8])
    y = 2.0 + x @ w_true
    return x, y, w_true


def test_fit_ridge_recovers_exact_linear():
    x, y, w_true = _linear_points()
    m = fit_ridge(x, y, lam=1e-6)
    assert np.allclose(m["coef"], w_true, atol=1e-3)
    assert m["intercept"] == pytest.approx(2.0, abs=1e-2)
    feats = dict(zip(FEATURES, x[0], strict=True))
    assert predict_ms(m, feats) == pytest.approx(y[0], rel=1e-3)


def test_loo_mape_near_zero_on_noiseless_data():
    x, y, _ = _linear_points()
    assert loo_mape(x, y, lam=1e-6) < 0.01


def test_collect_points_filters_unlocked_clocks(tmp_path):
    good = {"name": "m_640", "precision": "fp32", "clocks_locked": True, "power_mode": "0",
            "latency_ms": {"mean": 9.0}}
    bad = {**good, "name": "hot_640", "clocks_locked": False}
    (tmp_path / "a.json").write_text(json.dumps(good))
    (tmp_path / "b.json").write_text(json.dumps(bad))
    rows = collect_points(tmp_path, root=tmp_path)  # no ONNX roots under tmp → skip note
    assert [r["name"] for r in rows] == ["m_640"]
    assert rows[0]["skipped"] == "no ONNX resolved"


def test_resolve_onnx_strips_fp16_and_aliases(tmp_path):
    d = tmp_path / "data" / "e2e"
    d.mkdir(parents=True)
    (d / "yolo11n_pose_640.onnx").write_bytes(b"x")
    assert resolve_onnx("baseline_recheck_640_fp16", root=tmp_path).name == \
        "yolo11n_pose_640.onnx"
    assert resolve_onnx("nothing_here_640", root=tmp_path) is None


def test_extract_features_on_tiny_conv_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    w = helper.make_tensor("w", TensorProto.FLOAT, [8, 3, 3, 3],
                           np.zeros(8 * 3 * 3 * 3, dtype=np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16, 16])],
        initializer=[w])
    path = tmp_path / "tiny.onnx"
    onnx.save(helper.make_model(graph), str(path))

    from search.latency_model import extract_onnx_features

    f = extract_onnx_features(path)
    assert f["n_convs"] == 1
    assert f["param_mbytes"] == pytest.approx(8 * 3 * 9 * 4 / 2**20)
    # MACs = out_numel(1*8*16*16) * in_c(3) * 3 * 3; FLOPs = 2×
    assert f["conv_gflops"] == pytest.approx(2 * 2048 * 27 / 1e9)
    assert f["act_mbytes"] == pytest.approx(2048 * 4 / 2**20)  # the y output tensor
    assert f["coverage"] == 1.0


# --- graft-family physical fit (winner-v2-OFA, Track 0) --------------------------------------

def _line_points(slope=0.02, intercept=1.2, n=8):
    return [{"name": f"graft_p{i}_640", "ms": intercept + slope * a, "act_mbytes": float(a)}
            for i, a in enumerate(np.linspace(100, 600, n))]


def test_fit_physical_recovers_line():
    fit = fit_physical(_line_points())
    assert fit["slope"] == pytest.approx(0.02, rel=1e-6)
    assert fit["intercept"] == pytest.approx(1.2, rel=1e-6)
    assert fit["loo_mape"] < 1e-9
    assert fit["max_err_pct"] < 1e-6
    assert fit["n"] == 8 and len(fit["points"]) == 8


def test_fit_physical_needs_three_points():
    with pytest.raises(ValueError, match="need >=3"):
        fit_physical(_line_points(n=2))


def test_is_graft_e2e_point_filter():
    assert is_graft_e2e_point("graft_r50_gtay_640")
    assert is_graft_e2e_point("graft_halp_9p0_640_fp16")
    assert is_graft_e2e_point("winner_v1_e2e_640")
    assert is_graft_e2e_point("fallback_idx3_e2e_640")       # Stage-0 fallback topology
    assert not is_graft_e2e_point("winner_v1_backbone_640")  # partial net
    assert not is_graft_e2e_point("addv_idx0_backbone_640")
    assert not is_graft_e2e_point("dense_w22_640")
    assert not is_graft_e2e_point("baseline_recheck_640")


def test_act_limit_for_ms_roundtrips_the_fence():
    for model in (PHYSICAL_GRAFT_FP16, PHYSICAL_GRAFT_FP32):
        act = act_limit_for_ms(7.2, model)
        assert model["intercept"] + model["slope"] * act == pytest.approx(7.2, abs=1e-9)
    # sanity: the fp16 7.2 ms fence sits near the planned ~292 MB
    assert act_limit_for_ms(7.2, PHYSICAL_GRAFT_FP16) == pytest.approx(292.1, abs=0.5)


def test_graft_constants_match_tracked_fit_json():
    """The pinned PHYSICAL_GRAFT_* dicts must equal the tracked fit artifact."""
    tracked = json.loads((ROOT / "search" / "graft_latency_fit.json").read_text())
    for prec, const in (("fp32", PHYSICAL_GRAFT_FP32), ("fp16", PHYSICAL_GRAFT_FP16)):
        fit = tracked["fits"][prec]
        assert fit["slope"] == pytest.approx(const["slope"], rel=1e-6)
        assert fit["intercept"] == pytest.approx(const["intercept"], rel=1e-6)
        assert fit["n"] == const["n"]
        assert fit["loo_mape"] == pytest.approx(const["loo_mape"], abs=1e-4)


# --- dense-family physical fits (NAS-born beat-n program, 2026-07-18) -------------------------

def test_is_dense_e2e_point_filter():
    from search.latency_model import is_dense_e2e_point, is_dense_pruned_point

    assert is_dense_e2e_point("dense_w22_640")
    assert is_dense_e2e_point("densenas_s39_640_fp16")
    assert is_dense_e2e_point("prune_base_r20_640")
    assert is_dense_e2e_point("baseline_recheck_640")
    assert not is_dense_e2e_point("graft_r50_gtay_640")
    assert not is_dense_e2e_point("winner_v1_e2e_640")
    assert not is_dense_e2e_point("yolo11s_pose_640_fp16")   # the audit's one suspect row
    assert is_dense_pruned_point("prune_base_r45_640")
    assert not is_dense_pruned_point("dense_ctrl_n_640")


def test_dense_constants_match_tracked_fit_json():
    """The pinned PHYSICAL_DENSE_PRUNED_* dicts must equal the tracked fit artifact."""
    from search.latency_model import PHYSICAL_DENSE_PRUNED_FP16, PHYSICAL_DENSE_PRUNED_FP32

    tracked = json.loads((ROOT / "search" / "dense_latency_fit.json").read_text())
    for prec, const in (("fp32", PHYSICAL_DENSE_PRUNED_FP32),
                        ("fp16", PHYSICAL_DENSE_PRUNED_FP16)):
        fit = tracked["subfamilies"]["pruned"]["fits"][prec]
        assert fit["slope"] == pytest.approx(const["slope"], rel=1e-6)
        assert fit["intercept"] == pytest.approx(const["intercept"], rel=1e-6)
        assert fit["n"] == const["n"]
        assert fit["loo_mape"] == pytest.approx(const["loo_mape"], abs=1e-4)


def test_dense_fit_report_splits_subfamilies(monkeypatch):
    """The pruned/scaled currencies fit separately; grafts are excluded from all three."""
    pts = ([{"name": f"prune_base_r{i}_640", "ms": 0.5 + 0.05 * a, "act_mbytes": float(a),
             "precision": "fp32"} for i, a in enumerate((160, 180, 200))]
           + [{"name": f"dense_w{i}_640", "ms": 3.0 + 0.02 * a, "act_mbytes": float(a),
               "precision": "fp32"} for i, a in enumerate((400, 500, 600))]
           + [{"name": "graft_r50_gtay_640", "ms": 10.2, "act_mbytes": 300.0,
               "precision": "fp32"}])
    import search.latency_model as lm

    monkeypatch.setattr(lm, "collect_points", lambda d, root=None: pts)
    rep = lm.dense_fit_report("unused")
    assert rep["subfamilies"]["pruned"]["fits"]["fp32"]["slope"] == pytest.approx(
        0.05, rel=1e-6)
    assert rep["subfamilies"]["scaled"]["fits"]["fp32"]["slope"] == pytest.approx(
        0.02, rel=1e-6)
    assert rep["subfamilies"]["pooled"]["n_points"] == 6
    assert "graft_r50_gtay_640" in rep["subfamilies"]["pooled"]["excluded"]


def test_graft_fit_report_filters_and_fits(monkeypatch):
    pts = _line_points(n=6) + [
        {"name": "winner_v1_backbone_640", "ms": 9.9, "act_mbytes": 300.0},
        {"name": "dense_w22_640", "ms": 11.5, "act_mbytes": 500.0},
        {"name": "graft_skipped_640", "skipped": "no ONNX resolved"},
    ]
    for p in pts:
        p.setdefault("precision", "fp32")
    import search.latency_model as lm

    monkeypatch.setattr(lm, "collect_points", lambda d, root=None: pts)
    rep = lm.graft_fit_report("unused")
    assert rep["n_points"] == 6
    assert set(rep["excluded"]) == {"winner_v1_backbone_640", "dense_w22_640"}
    assert rep["fits"]["fp32"]["slope"] == pytest.approx(0.02, rel=1e-6)
