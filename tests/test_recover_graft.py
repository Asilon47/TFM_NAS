"""Guard for the graft prune+train ladder (prune/recover_graft.py) — pure bits only.

The prune/train wiring needs .venv-nas (OFA + ultralytics + torch-pruning) and is exercised by the
module's own CPU smoke; this pins the unpruned anchor so CI stays green (all heavy imports live
inside graft_prune_train_ladder / main).
"""
from prune import recover_graft


def test_unpruned_anchor_is_the_full_ft_map():
    # winner-v1's full-FT mAP (full_finetune.json, 2026-07-05) — the ladder's delta reference.
    assert 0.0 < recover_graft.UNPRUNED_GRAFT_ANCHOR_MAP < 1.0
    assert recover_graft.UNPRUNED_GRAFT_ANCHOR_MAP == 0.841
