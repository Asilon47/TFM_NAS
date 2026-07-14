"""Contention experiment — the pure ratio/gap analysis (device parts need the Jetson)."""
from lut.orchestrate.contention_bench import analyze


def test_analyze_flags_widening_gap():
    # roofline case: under load the memory-bound graft degrades MORE than the baseline,
    # so the graft/baseline ratio RISES (Δratio > 0) vs the unloaded control.
    rows = {
        "none": {"baseline": 7.7, "graft": 12.4},   # ratio 1.61
        "dram": {"baseline": 9.0, "graft": 17.0},   # ratio 1.89 -> widened
    }
    table = {r["condition"]: r for r in analyze(rows, "baseline", "graft")}
    assert table["none"]["ratio"] == 12.4 / 7.7
    assert table["dram"]["ratio"] > table["none"]["ratio"]
    assert table["dram"]["ratio_delta_vs_control"] > 0          # gap widened
    assert (table["dram"]["graft_slowdown_vs_control"]
            > table["dram"]["baseline_slowdown_vs_control"])


def test_analyze_flags_equalizing_gap():
    # the user's hypothesis: under load the ratio FALLS toward 1 (Δratio < 0).
    rows = {
        "none": {"baseline": 7.7, "graft": 12.4},
        "cpu":  {"baseline": 11.0, "graft": 14.0},   # ratio 1.27 -> narrowed
    }
    table = {r["condition"]: r for r in analyze(rows, "baseline", "graft")}
    assert table["cpu"]["ratio_delta_vs_control"] < 0
    assert table["cpu"]["gap_ms"] < table["none"]["gap_ms"]


def test_analyze_handles_missing_rows():
    rows = {"none": {"baseline": 7.7}}   # graft missing
    table = {r["condition"]: r for r in analyze(rows, "baseline", "graft")}
    assert table["none"]["ratio"] is None and table["none"]["gap_ms"] is None
