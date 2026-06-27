"""CP 3.2 contract: NSGA-II evolutionary search over (depth_sum, latency_ms).

DoD (PROJECT_PLAN.md:219): the frontier has >= 10 non-dominated points.

The pure helpers (objective wiring, non-dominated dedup) are tested without pymoo
or the LUT so they run in CI. The end-to-end NSGA-II run is gated on pymoo +
a local LUT and marked slow. evaluate_objectives needs the LUT (latency) but not
pymoo; run_search lazy-imports pymoo, so importing this module never requires it.
"""
import pytest

from catalog.ofa_mbv3 import KS, MAX_DEPTH, STAGES, D, E
from search.cost import CostError
from search.cost_preview import nondominated_indices
from search.evolution import _nondominated_dedup, evaluate_objectives, run_search

N_SLOTS = len(STAGES) * MAX_DEPTH
N_STAGES = len(STAGES)


# ---- pure: non-dominated dedup (no pymoo, no LUT, always runs) ----

def test_nondominated_dedup_keeps_skyline_and_dedups():
    # minimize both axes. Trade-off skyline = {(1,3),(2,2),(3,1)};
    # (2,3) and (4,4) are dominated; the duplicate (2,2) collapses to one.
    objs = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (2.0, 3.0), (4.0, 4.0), (2.0, 2.0)]
    idx = _nondominated_dedup(objs)
    assert sorted(objs[i] for i in idx) == [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]


def test_nondominated_dedup_matches_costpreview_util():
    # cross-check against the existing tested skyline util (also minimize-both).
    objs = [(1.0, 5.0), (2.0, 2.0), (5.0, 1.0), (3.0, 3.0), (2.0, 2.0), (0.5, 9.0)]
    mine = {objs[i] for i in _nondominated_dedup(objs)}
    ref_idx = nondominated_indices([o[0] for o in objs], [o[1] for o in objs])
    assert mine == {objs[i] for i in ref_idx}


# ---- objective wiring (needs LUT for latency, no pymoo) ----

def test_evaluate_objectives_depth_sign_and_latency(lut_path):
    from lut.loader import load_lut

    lut = load_lut(lut_path, precision="fp32")
    deep = {"ks": [3] * N_SLOTS, "e": [3] * N_SLOTS, "d": [D[-1]] * N_STAGES}    # all depth 4
    shallow = {"ks": [3] * N_SLOTS, "e": [3] * N_SLOTS, "d": [D[0]] * N_STAGES}  # all depth 2
    try:
        f_deep = evaluate_objectives(deep, lut)
        f_shallow = evaluate_objectives(shallow, lut)
    except CostError:
        pytest.skip("LUT partial — some blocks missing")
    # f1 = -depth_sum (maximize depth -> minimize negative): deeper is more negative.
    assert f_deep[0] == -float(sum(deep["d"]))
    assert f_deep[0] < f_shallow[0]
    # f2 = latency_ms: positive, and more blocks -> higher latency.
    assert f_deep[1] > 0 and f_shallow[1] > 0
    assert f_deep[1] > f_shallow[1]


# ---- end-to-end NSGA-II (pymoo + LUT, slow) ----

@pytest.mark.slow
def test_nsga2_frontier_has_at_least_10_points(lut_path):
    pytest.importorskip("pymoo")
    from lut.loader import load_lut

    lut = load_lut(lut_path, precision="fp32")
    try:
        result = run_search(lut, pop_size=40, n_gen=40, seed=0)
    except CostError:
        pytest.skip("LUT partial — some blocks missing")
    assert result.n_nondominated >= 10
    # every reported point is genuinely non-dominated (in minimize form) and distinct.
    objs = [(-fp.depth_sum, fp.latency_ms) for fp in result.frontier]
    assert len(_nondominated_dedup(objs)) == len(objs)


@pytest.mark.slow
def test_run_search_is_seed_reproducible(lut_path):
    pytest.importorskip("pymoo")
    from lut.loader import load_lut

    lut = load_lut(lut_path, precision="fp32")
    try:
        a = run_search(lut, pop_size=20, n_gen=15, seed=7)
        b = run_search(lut, pop_size=20, n_gen=15, seed=7)
    except CostError:
        pytest.skip("LUT partial — some blocks missing")
    fa = sorted((fp.depth_sum, round(fp.latency_ms, 6)) for fp in a.frontier)
    fb = sorted((fp.depth_sum, round(fp.latency_ms, 6)) for fp in b.frontier)
    assert fa == fb


@pytest.mark.slow
def test_default_budget_reaches_true_pareto_front(lut_path):
    """The default budget (pop=150) converges to the GLOBAL front: every point is
    at min ks/e (ks=3, e=3 on all active blocks) — the true min-latency arch at its
    depth. pop=50 finds >=10 non-dominated points but leaves ~9 of them ~1.5% above
    optimal, so this locks the population fix (see run_search docstring)."""
    pytest.importorskip("pymoo")
    from lut.loader import load_lut

    lut = load_lut(lut_path, precision="fp32")
    try:
        result = run_search(lut, seed=0)  # defaults: pop_size=150, n_gen=200
    except CostError:
        pytest.skip("LUT partial — some blocks missing")
    assert result.n_nondominated >= 10
    for fp in result.frontier:
        a = fp.arch
        active_all_min = all(
            a["ks"][MAX_DEPTH * s + j] == KS[0] and a["e"][MAX_DEPTH * s + j] == E[0]
            for s, d in enumerate(a["d"])
            for j in range(d)
        )
        assert active_all_min, f"depth {fp.depth_sum} point is not at min ks/e (under-converged)"
