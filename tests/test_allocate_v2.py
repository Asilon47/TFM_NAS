"""Winner-v2-OFA Track 1/1b — prune/allocate_v2.py pure math (predictor, ranking, screen).

The honest-build path (CPU prune → ONNX → act bytes) needs .venv-nas and is exercised by the
module's own runs; everything here is dependency-free.
"""
import pytest

from prune.allocate_v2 import (
    ACT_BIN_MB,
    D_PATTERNS,
    E_PATTERNS,
    N_STAGES,
    REST_BASE,
    REST_RUNGS,
    STAGE_RUNGS,
    enumerate_specs,
    expand_per_stage,
    pick_minact,
    predict_act,
    rank_candidates,
    screen_grid,
    spec_payload,
)

SENS = {"base_act": 500.0, "stage": [120.0, 60.0, 50.0, 40.0, 20.0], "rest": 65.0}


def test_predict_act_monotone_in_every_knob():
    base = predict_act([0.0] * N_STAGES, REST_BASE, SENS)
    assert base == pytest.approx(SENS["base_act"])
    for s in range(N_STAGES):
        ratios = [0.0] * N_STAGES
        ratios[s] = 0.5
        assert predict_act(ratios, REST_BASE, SENS) < base
    assert predict_act([0.0] * N_STAGES, 0.6, SENS) < base
    # raising any single knob never raises predicted act
    lo = predict_act([0.2] * N_STAGES, 0.3, SENS)
    hi = predict_act([0.2, 0.2, 0.3, 0.2, 0.2], 0.3, SENS)
    assert hi < lo


def test_enumerate_specs_covers_the_grid():
    grid = enumerate_specs()
    assert len(grid) == len(STAGE_RUNGS) ** N_STAGES * len(REST_RUNGS)
    assert all(rr >= REST_BASE > 0.0 for _, rr in grid[:100])


def _mk(sr, rr, act):
    return {"stage_ratios": tuple(sr), "rest_ratio": rr, "act": act}


def test_rank_candidates_respects_fence_and_prefers_balance():
    mk = _mk
    over = mk([0.1] * 5, 0.1, 300.0)                    # over the fence → dropped
    balanced = mk([0.4] * 5, 0.2, 290.0)
    concentrated = mk([0.7, 0.7, 0.1, 0.1, 0.1], 0.2, 290.5)   # same act bin, lumpier
    smaller = mk([0.5] * 5, 0.2, 250.0)                 # feasible but wastes budget
    ranked = rank_candidates([over, smaller, concentrated, balanced], act_max=292.0)
    assert over not in ranked
    assert ranked[0] == balanced          # same act bin as concentrated → balance wins
    assert ranked[1] == concentrated
    assert ranked[-1] == smaller          # lower act bin ranks last
    assert abs(balanced["act"] - concentrated["act"]) < ACT_BIN_MB


def test_rank_candidates_breaks_ties_on_rest_then_mean():
    mk = _mk
    a = mk([0.4] * 5, 0.1, 290.0)
    b = mk([0.4] * 5, 0.3, 290.0)
    c = mk([0.4, 0.4, 0.4, 0.4, 0.3], 0.1, 290.0)   # same max, lower mean
    ranked = rank_candidates([b, a, c], act_max=300.0)
    assert ranked[0] == c and ranked[1] == a and ranked[2] == b


def test_spec_payload_carries_the_legacy_keys():
    """recover_graft.py indexes these five — regression guard for spec consumption."""
    p = spec_payload([0.4] * 5, 0.2, act_honest=290.0, act_predicted=288.5,
                     params_after=800_000, fence_fp16_ms=7.2, act_max=292.1)
    for k in ("stage_ratios", "rest_ratio", "predicted_fp32_ms", "fp16_estimate_ms",
              "target_fp32_ms"):
        assert k in p, k
    assert p["rest_ratio"] == 0.2
    assert p["fence"]["fp16_target_ms"] == 7.2
    # predictions follow the pinned graft fit (fp16 ≈ 1.20 + 0.0205·act)
    assert p["fp16_estimate_ms"] == pytest.approx(1.1999 + 0.020542 * 290.0, abs=0.01)
    assert p["predicted_fp32_ms"] == pytest.approx(0.7876 + 0.031618 * 290.0, abs=0.01)


def test_expand_per_stage():
    assert expand_per_stage([3, 3, 4, 4, 6]) == [3] * 4 + [3] * 4 + [4] * 4 + [4] * 4 + [6] * 4
    with pytest.raises(ValueError, match="per-stage"):
        expand_per_stage([3, 3])


def test_screen_grid_shapes():
    ks = [5] * 20
    grid = screen_grid(ks)
    assert len(grid) == len(D_PATTERNS) * len(E_PATTERNS)
    for cand in grid:
        assert cand["arch"]["ks"] == ks
        assert len(cand["arch"]["e"]) == 20
        assert len(cand["arch"]["d"]) == N_STAGES
        assert cand["tag"].startswith("minact_d")
    assert len({c["tag"] for c in grid}) == len(grid)   # tags unique


def test_pick_minact_prefers_depth_then_act():
    rows = [
        {"tag": "a", "d": [2, 2, 2, 2, 2], "act_mbytes": 260.0},
        {"tag": "b", "d": [2, 2, 2, 3, 3], "act_mbytes": 285.0},   # deepest under fence
        {"tag": "c", "d": [2, 2, 2, 3, 3], "act_mbytes": 275.0},   # same depth, less act
        {"tag": "d", "d": [2, 2, 4, 3, 3], "act_mbytes": 400.0},   # over the fence
    ]
    assert pick_minact(rows, act_max=292.0)["tag"] == "b"
    with pytest.raises(ValueError, match="no screened topology"):
        pick_minact(rows, act_max=100.0)


def test_pick_probe_pairs_when_nothing_fits():
    from prune.allocate_v2 import pick_probe

    rows = [
        {"tag": "lean", "d": [2, 2, 2, 2, 2], "act_mbytes": 340.0},
        {"tag": "deep_lean", "d": [2, 2, 4, 3, 3], "act_mbytes": 404.0},  # max dsum, min act
        {"tag": "deep_rich", "d": [2, 2, 4, 3, 3], "act_mbytes": 455.0},
        {"tag": "fits", "d": [2, 2, 2, 2, 3], "act_mbytes": 285.0},
    ]
    # something fits → pure pick, no pairing
    probe, needs_pair = pick_probe(rows, act_max=292.0)
    assert probe["tag"] == "fits" and needs_pair is False
    # nothing fits (the measured 2026-07-13 case: floor 340 > 292) → pair the deepest,
    # lightest-prune candidate
    probe, needs_pair = pick_probe(rows[:3], act_max=292.0)
    assert probe["tag"] == "deep_lean" and needs_pair is True
