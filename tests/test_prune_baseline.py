"""prune/prune_baseline.py — pure parts of the CP 6.2-B control-arm ladder."""
import pytest

from prune.prune_baseline import assemble_ladder_report, ladder_plan


def test_ladder_plan_canonicalizes() -> None:
    assert ladder_plan([0.45, 0.15, 0.30, 0.15]) == [0.15, 0.30, 0.45]


def test_ladder_plan_guards() -> None:
    with pytest.raises(ValueError, match="empty"):
        ladder_plan([])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ladder_plan([0.15, 1.0])


def test_assemble_ladder_report_deltas_and_best() -> None:
    donor = {"map": 0.877, "params": 2_900_000}
    rows = [
        {"ratio": 0.30, "params": 2_000_000, "map": 0.850},
        {"ratio": 0.15, "params": 2_400_000, "map": 0.870},
    ]
    rep = assemble_ladder_report(donor, rows)
    assert [r["ratio"] for r in rep["rows"]] == [0.15, 0.30]          # sorted
    assert rep["rows"][0]["delta_map_vs_donor"] == pytest.approx(-0.007)
    assert rep["best_row_ratio"] == 0.15
    assert "measured-only" in rep["note"]


def test_assemble_ladder_report_needs_donor_map() -> None:
    with pytest.raises(ValueError, match="donor"):
        assemble_ladder_report({}, [])
