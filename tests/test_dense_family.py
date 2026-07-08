"""search/dense_family.py — pure parts of the Phase-3c wave-1 scaling search."""
import pytest

from search.dense_family import (
    WAVE1,
    assemble_wave_report,
    fixed_data_yaml_dict,
    scaled_yaml,
    wave_tags,
)


def test_wave1_shape() -> None:
    tags = wave_tags()
    assert "ctrl_n" in tags                                  # the recipe control is mandatory
    assert len(tags) == len(set(tags)) == len(WAVE1)
    # yolo11n-pose's own scale is the control triple
    ctrl = next(c for c in WAVE1 if c[0] == "ctrl_n")
    assert ctrl[1:] == (0.50, 0.25, 1024)


def test_wave_tags_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        wave_tags([("a", 0.5, 0.25, 1024), ("a", 0.33, 0.25, 1024)])


def test_scaled_yaml_pins_single_scale() -> None:
    base: dict = {"nc": 80, "kpt_shape": [17, 3],
                  "scales": {"n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024]}, "backbone": []}
    out = scaled_yaml(base, 0.33, 0.20, 1024)
    assert out["scales"] == {"n": [0.33, 0.20, 1024]}        # only our triple survives
    assert base["scales"]["s"] == [0.5, 0.5, 1024]           # input not mutated
    assert out["backbone"] is base["backbone"]               # graph untouched


def test_scaled_yaml_guards() -> None:
    with pytest.raises(ValueError, match="suspicious"):
        scaled_yaml({}, 2.0, 0.25, 1024)
    with pytest.raises(ValueError, match="max_channels"):
        scaled_yaml({}, 0.5, 0.25, 32)


def test_fixed_data_yaml_reroots_path(tmp_path) -> None:
    data = {"path": "/kaggle/input/datasets/someone/dataset/dataset", "train": "images/train"}
    out = fixed_data_yaml_dict(data, tmp_path)
    assert out["path"] == str(tmp_path)
    assert out["train"] == "images/train"
    assert data["path"].startswith("/kaggle")                # input not mutated


def test_assemble_wave_report_sorts_and_caveats() -> None:
    rows = [{"tag": "big", "params": 3_000_000, "map": 0.85},
            {"tag": "small", "params": 1_000_000, "map": 0.80}]
    rep = assemble_wave_report(rows, anchors={"yolo11n_pretrained": 0.877})
    assert [r["tag"] for r in rep["rows"]] == ["small", "big"]
    assert rep["control_tag"] == "ctrl_n"
    assert "single-seed" in rep["note"] and "de-noise" in rep["note"]
    assert rep["anchors"]["yolo11n_pretrained"] == 0.877


def test_wave2_is_width_only_and_unique() -> None:
    from search.dense_family import WAVE2, WAVES
    tags = wave_tags(WAVE2)
    assert len(tags) == len(set(tags)) == len(WAVE2)
    assert all(d == 0.50 for _, d, _, _ in WAVE2)          # depth fixed (degenerate below n)
    assert WAVES["1"] and WAVES["2"] == WAVE2
    widths = sorted(w for _, _, w, _ in WAVE2)
    assert widths[0] < 0.15 and widths[-1] > 0.25          # extends below and above wave-1
