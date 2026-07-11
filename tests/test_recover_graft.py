"""Guard for the graft prune+train ladder (prune/recover_graft.py) — pure bits only.

The prune/train wiring needs .venv-nas (OFA + ultralytics + torch-pruning) and is exercised by the
module's own CPU smoke; this pins the unpruned anchor so CI stays green (all heavy imports live
inside graft_prune_train_ladder / main).
"""
import json

import pytest

from prune import recover_graft


def test_unpruned_anchor_is_the_full_ft_map():
    # winner-v1's full-FT mAP (full_finetune.json, 2026-07-05) — the ladder's delta reference.
    assert 0.0 < recover_graft.UNPRUNED_GRAFT_ANCHOR_MAP < 1.0
    assert recover_graft.UNPRUNED_GRAFT_ANCHOR_MAP == 0.841


def test_load_candidate_arch_selects_by_index(tmp_path):
    """The G1 probe selects fallback topologies by INDEX — d=[2,2,4,3,2] appears twice in the
    real top-12, so depth lists are not unique keys."""
    cand = tmp_path / "denoise_candidates.json"
    arch3 = {"ks": [3] * 20, "e": [4] * 20, "d": [2, 2, 4, 3, 2]}
    cand.write_text(json.dumps({"candidates": [
        {"arch": {"d": [9]}}, {"arch": {"d": [8]}}, {"arch": {"d": [7]}}, {"arch": arch3},
    ]}))
    assert recover_graft.load_candidate_arch(cand, 3) == arch3
    with pytest.raises(ValueError, match="out of range"):
        recover_graft.load_candidate_arch(cand, 4)


def test_ladder_vocab_reexported():
    # kaggle/run.py imports run_tag/TECHNIQUES via this module — keep the surface.
    assert recover_graft.run_tag(0.60, technique="global_taylor") == "r60_gtay"
    assert "uniform" in recover_graft.TECHNIQUES
