"""CP 3.3 — pure helpers of the BO search (numpy-free, no GPU / botorch).

These exercise the parts of ``search.bo`` that drive the loop and score the DoD
without the GP: the ParEGO scalarization, the 2-D Pareto hypervolume (the DoD
metric), candidate generation under the hard latency ceiling, and the
hypervolume-vs-random verdict. The BoTorch GP fit + ``run_bo`` driver are
integration-smoked separately (they need the surrogate stack).

The synthetic LUT gives every catalog MBConv row a constant 0.01 ms latency, so a
subnet's predicted latency is ``0.01 * n_blocks`` — monotone in depth, which makes
the ceiling filter (and its boundary) deterministic to test.
"""
import random

import pytest

from catalog.sweep import iter_sweep
from search.arch_to_blocks import _random_arch_dict, validate_arch_dict
from search.bo import (
    BoVerdict,
    bo_verdict,
    candidate_pool,
    feasible,
    hypervolume_2d,
    mutate_arch,
    nondominated_indices,
    parego_weights,
    pareto_hypervolume,
    tchebycheff_scalarize,
)
from search.space import canonical, encode


def _row(mean: float, key: str) -> dict:
    return {
        "row_key": key, "block": "mbconv", "cfg": {}, "input_shape": [1, 3, 8, 8],
        "precision": "fp16",
        "latency_ms": {"mean": mean, "std": 0.0, "p50": mean, "p95": mean, "n": 1},
        "peak_mem_mib": 1.0, "params": 100, "flops": 1000,
        "achieved_bw_gbps": 0.0, "trt_version": "10.3.0", "power_mode": "0",
        "jetpack": None, "timestamp": "2026-01-01T00:00:00Z",
    }


@pytest.fixture(scope="module")
def synth_lut() -> dict:
    """Every @224 MBConv catalog row at a constant 0.01 ms (latency = 0.01*n_blocks)."""
    return {k: _row(0.01, k) for _, _, _, k in iter_sweep(["mbconv"])}


# ---- ParEGO scalarization ----------------------------------------------------

def test_parego_weights_lie_on_the_simplex():
    rng = random.Random(0)
    for _ in range(50):
        w = parego_weights(rng, n_obj=2)
        assert len(w) == 2
        assert all(wi >= 0.0 for wi in w)
        assert sum(w) == pytest.approx(1.0)


def test_parego_weights_are_seed_deterministic_but_vary():
    assert parego_weights(random.Random(1)) == parego_weights(random.Random(1))
    assert parego_weights(random.Random(1)) != parego_weights(random.Random(2))


def test_tchebycheff_is_zero_at_the_ideal():
    # normalized costs (0 = best) with any weights -> 0 at the ideal point
    assert tchebycheff_scalarize([0.0, 0.0], [0.5, 0.5]) == pytest.approx(0.0)


def test_tchebycheff_takes_the_weighted_max_plus_augmentation():
    # max(0.5*0.2, 0.5*0.8) + 0.05*(0.1+0.4) = 0.4 + 0.025
    assert tchebycheff_scalarize([0.2, 0.8], [0.5, 0.5], rho=0.05) == pytest.approx(0.425)


def test_tchebycheff_prefers_lower_costs():
    w = [0.5, 0.5]
    assert tchebycheff_scalarize([0.1, 0.1], w) < tchebycheff_scalarize([0.9, 0.9], w)


# ---- non-domination + hypervolume (the DoD metric) ---------------------------

def test_nondominated_drops_dominated_points():
    pts = [(1.0, 4.0), (3.0, 2.0), (2.0, 1.0)]  # maximize both
    assert set(nondominated_indices(pts)) == {0, 1}  # (2,1) dominated by (3,2)


def test_nondominated_keeps_a_single_point():
    assert nondominated_indices([(2.0, 2.0)]) == [0]


def test_hypervolume_single_point_is_the_rectangle():
    assert hypervolume_2d([(2.0, 3.0)], ref=(0.0, 0.0)) == pytest.approx(6.0)


def test_hypervolume_two_points_is_the_rectangle_union():
    # union of [0,3]x[0,2] (6) and [0,1]x[0,4] (4) minus overlap [0,1]x[0,2] (2) = 8
    assert hypervolume_2d([(1.0, 4.0), (3.0, 2.0)], ref=(0.0, 0.0)) == pytest.approx(8.0)


def test_hypervolume_ignores_dominated_points():
    base = hypervolume_2d([(1.0, 4.0), (3.0, 2.0)], ref=(0.0, 0.0))
    with_dom = hypervolume_2d([(1.0, 4.0), (3.0, 2.0), (2.0, 1.0)], ref=(0.0, 0.0))
    assert with_dom == pytest.approx(base)


def test_hypervolume_grows_with_a_nondominated_point():
    base = hypervolume_2d([(3.0, 2.0)], ref=(0.0, 0.0))
    grown = hypervolume_2d([(3.0, 2.0), (1.0, 4.0)], ref=(0.0, 0.0))
    assert grown > base


def test_pareto_hypervolume_maximizes_acc_minimizes_latency():
    # acc up / latency down both improve -> dominate; ref = (worst acc, worst lat)
    evals = [(0.80, 10.0), (0.70, 5.0)]  # (acc_eff, latency_ms)
    hv = pareto_hypervolume(evals, ref_acc=0.0, ref_lat=20.0)
    # transform: maximize (acc, -lat) vs ref (0, -20); both points nondominated
    assert hv == pytest.approx(hypervolume_2d([(0.80, -10.0), (0.70, -5.0)], ref=(0.0, -20.0)))
    # a strictly better front (higher acc at same latency) has larger HV
    better = pareto_hypervolume([(0.85, 10.0), (0.75, 5.0)], ref_acc=0.0, ref_lat=20.0)
    assert better > hv


# ---- candidate generation under the latency ceiling --------------------------

def test_feasible_respects_the_ceiling(synth_lut):
    arch = _random_arch_dict(random.Random(0))
    n_blocks = 1 + sum(arch["d"])
    lat = 0.01 * n_blocks
    assert feasible(arch, synth_lut, t_max=lat + 1e-9)        # at the bound: in
    assert not feasible(arch, synth_lut, t_max=lat - 1e-3)    # just under: out


def test_mutate_arch_stays_valid_and_changes_canonically():
    rng = random.Random(0)
    arch = _random_arch_dict(rng)
    changed = 0
    for _ in range(20):
        mut = mutate_arch(arch, rng, n_edits=2)
        validate_arch_dict(mut)  # never emits an invalid arch
        if canonical(encode(mut)) != canonical(encode(arch)):
            changed += 1
    assert changed > 0  # mutation actually explores


def test_candidate_pool_is_feasible_deduped_and_excludes_evaluated(synth_lut):
    rng = random.Random(0)
    t_max = 0.15  # filters out the deepest archs (latency up to 0.21)
    seen_arch = _random_arch_dict(random.Random(99))
    evaluated = {tuple(canonical(encode(seen_arch)))}
    pool = candidate_pool(synth_lut, t_max=t_max, rng=rng, evaluated=evaluated, size=40)

    assert pool, "pool should not be empty"
    canon = [tuple(canonical(encode(a))) for a in pool]
    assert len(canon) == len(set(canon))                      # deduped
    assert all(c not in evaluated for c in canon)             # excludes evaluated
    for a in pool:
        validate_arch_dict(a)
        assert feasible(a, synth_lut, t_max=t_max)            # honors the ceiling


# ---- the DoD verdict: BO hypervolume beats random search ---------------------

def test_bo_verdict_passes_when_bo_dominates_with_separation():
    v = bo_verdict(bo_hvs=[1.00, 1.02, 0.98], rs_hvs=[0.70, 0.72, 0.68])
    assert isinstance(v, BoVerdict)
    assert v.passes


def test_bo_verdict_fails_when_dispersions_overlap():
    v = bo_verdict(bo_hvs=[1.00, 0.60], rs_hvs=[0.95, 0.55])
    assert not v.passes
