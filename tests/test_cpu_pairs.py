"""Contract tests for the Jetson<->ONNX pair map.

Jetson row names do not match ONNX filenames and cannot be derived (baseline_recheck_640 is
yolo11n_pose_640.onnx; dense_ctrl_n_640 is dense_w25_640.onnx). A mis-pair would produce a
corrupt-but-plausible Spearman, so the map is declared data and tested as data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lut.orchestrate.cpu_pairs import (
    CANARY,
    GRAFT_FAMILIES,
    PAIRS,
    REFERENCE_FAMILIES,
    resolve_pairs,
)

MODELS = Path(__file__).resolve().parents[1] / "models"
E2E = Path(__file__).resolve().parents[1] / "data" / "e2e"


def test_pair_count() -> None:
    # 1 anchor + 1 baseline + 7 dense + 3 dense_nas + 3 graft + 7 graft_pruned + 7 prune
    assert len(PAIRS) == 29


def test_jetson_names_unique() -> None:
    names = [p.jetson_name for p in PAIRS]
    assert len(set(names)) == len(names)


def test_onnx_paths_unique() -> None:
    """Two rows mapped to one ONNX would silently duplicate a point in the correlation."""
    paths = [p.onnx for p in PAIRS]
    assert len(set(paths)) == len(paths)


def test_no_backbone_only_models() -> None:
    """Backbone rows are a different network scope -- mixing them is the retired-claim error."""
    for p in PAIRS:
        assert "backbone" not in p.jetson_name, p.jetson_name
        assert "_bb_" not in p.onnx, p.onnx


def test_canary_is_in_the_map() -> None:
    assert CANARY in {p.jetson_name for p in PAIRS}


def test_families_are_known() -> None:
    known = {"baseline", "anchor", "dense", "dense_nas", "prune", "graft", "graft_pruned"}
    assert {p.family for p in PAIRS} <= known


def test_reference_set_size_and_range() -> None:
    """The OLS reference is dense+prune+baseline = 15 models (anchor excluded: leverage)."""
    ref = [p for p in PAIRS if p.family in REFERENCE_FAMILIES]
    assert len(ref) == 15
    assert "anchor" not in REFERENCE_FAMILIES


def test_graft_families() -> None:
    grafts = [p for p in PAIRS if p.family in GRAFT_FAMILIES]
    assert len(grafts) == 10


@pytest.mark.skipif(not MODELS.exists(), reason="models/ is gitignored; absent in CI")
def test_every_onnx_exists() -> None:
    resolved = resolve_pairs(MODELS.parent)
    assert len(resolved) == 29
    for _, path in resolved:
        assert path.is_file(), path


@pytest.mark.skipif(not E2E.exists(), reason="data/ is gitignored; absent in CI")
def test_every_jetson_row_exists() -> None:
    for p in PAIRS:
        assert (E2E / f"{p.jetson_name}.json").is_file(), p.jetson_name


def test_resolve_pairs_raises_listing_all_misses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        resolve_pairs(tmp_path)
    # Must list misses, not fail on the first -- a short pair list weakens rho silently.
    msg = str(exc.value)
    assert "29 of 29" in msg
    for pair in PAIRS:
        assert pair.onnx in msg
