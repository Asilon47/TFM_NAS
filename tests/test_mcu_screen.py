"""mcu/mcu_screen.py — Stage-1 screen sampler (pure parts)."""
from mcu.mcu_screen import NECKS, RESES, sample_screen


def test_sample_screen_is_deterministic_and_stratified() -> None:
    a = sample_screen(18, seed=0)
    b = sample_screen(18, seed=0)
    assert a == b                                        # seeded
    cells = {(c["neck"], c["imgsz"]) for c in a}
    assert cells == {(n, r) for n in NECKS for r in RESES}   # every cell covered at n=18


def test_sample_screen_respects_the_ofa_contract() -> None:
    for c in sample_screen(9, seed=3):
        assert len(c["arch"]["ks"]) == 20 and len(c["arch"]["e"]) == 20
        assert len(c["arch"]["d"]) == 5
        assert all(d in (2, 3, 4) for d in c["arch"]["d"])
        assert len(c["spec"]["stage_ratios"]) == 5
        assert 0.0 < c["spec"]["rest_ratio"] <= 0.45
        assert c["neck"] in NECKS and c["imgsz"] in RESES


def test_sample_screen_focus_restricts_cells() -> None:
    """Wave-2 focus: only the requested neck/res cells appear, still seeded + stratified."""
    cands = sample_screen(12, seed=1, necks=["topdown"], reses=[160, 192])
    assert {(c["neck"], c["imgsz"]) for c in cands} == {("topdown", 160), ("topdown", 192)}
    assert len(cands) == 12
    assert cands == sample_screen(12, seed=1, necks=["topdown"], reses=[160, 192])
