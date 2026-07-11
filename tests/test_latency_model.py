"""Track 2 — search/latency_model.py: the fit math (pure) + ONNX feature extraction."""
import json

import numpy as np
import pytest

from search.latency_model import (
    FEATURES,
    collect_points,
    fit_ridge,
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
