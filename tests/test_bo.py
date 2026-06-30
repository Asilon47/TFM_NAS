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
import json
import random
import time

import pytest

from catalog.sweep import iter_sweep
from search.arch_to_blocks import _random_arch_dict, validate_arch_dict
from search.bo import (
    BoVerdict,
    assign_seeds_to_gpus,
    bo_verdict,
    candidate_pool,
    depth_sum_accuracy,
    feasible,
    hypervolume_2d,
    main,
    merge_bo_outputs,
    mutate_arch,
    nondominated_indices,
    parego_weights,
    pareto_hypervolume,
    random_search_control,
    run_bo,
    seed_remaining_evals,
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


# ---- run_bo integration (needs the BoTorch surrogate; CI-skips if absent) -----

def test_run_bo_drives_the_gp_loop_to_a_frontier(synth_lut):
    """The full ParEGO -> MixedSingleTaskGP -> qLogEI -> discrete-pool loop runs."""
    pytest.importorskip("botorch")
    bo = run_bo(depth_sum_accuracy, synth_lut, budget=8, n_init=4, seed=0,
                t_max=0.18, res=224)
    assert bo.n_evals == 8
    assert bo.complete is True  # spent the full budget within (no) deadline
    assert bo.hypervolume > 0.0
    assert bo.frontier  # at least one non-dominated point
    # every evaluated arch honored the hard latency ceiling
    assert all(e["latency_ms"] <= 0.18 + 1e-9 for e in bo.evals)


def test_run_bo_resume_skips_already_evaluated(synth_lut, tmp_path):
    """A re-run with the same cache loads prior evals and never re-evaluates them."""
    pytest.importorskip("botorch")
    cache = tmp_path / "bo_cache.jsonl"
    calls = {"n": 0}

    def counting_eval(arch):
        calls["n"] += 1
        return depth_sum_accuracy(arch)

    run_bo(counting_eval, synth_lut, budget=6, n_init=3, seed=0,
           t_max=0.18, res=224, cache_path=cache)
    first = calls["n"]
    assert first == 6 and cache.exists()

    # second run with the same cache: all 6 are already done -> zero new eval calls
    bo2 = run_bo(counting_eval, synth_lut, budget=6, n_init=3, seed=0,
                 t_max=0.18, res=224, cache_path=cache)
    assert calls["n"] == first      # no re-evaluation
    assert bo2.n_evals == 6         # the cached evals populate the run


def test_merge_bo_outputs_concatenates_seeds_and_recomputes_verdict():
    """Per-worker outputs (disjoint seeds) combine into one, verdict over ALL seeds.

    This is what lets the DoD fan its seeds across multiple GPUs and rejoin them:
    each worker writes a slice, the merge recomputes the across-seed verdict.
    """
    def run(seed, bo, rs):
        return {"seed": seed, "bo_hv": bo, "rs_hv": rs, "bo_frontier": []}

    def payload(runs):
        return {"passes": False, "n_seeds": len(runs), "bo_hv_mean": 0.0,
                "bo_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0,
                "t_max_ms": 16.7, "res": 640, "budget": 50, "structural": False,
                "runs": runs}

    a = payload([run(0, 1.00, 0.70), run(1, 1.02, 0.72), run(2, 0.98, 0.68)])
    b = payload([run(3, 1.01, 0.69), run(4, 0.99, 0.71)])
    merged = merge_bo_outputs([b, a])  # out of order -> must sort by seed

    assert [r["seed"] for r in merged["runs"]] == [0, 1, 2, 3, 4]
    assert merged["n_seeds"] == 5
    expect = bo_verdict([1.00, 1.02, 0.98, 1.01, 0.99],
                        [0.70, 0.72, 0.68, 0.69, 0.71])
    assert merged["passes"] == expect.passes
    assert merged["bo_hv_mean"] == pytest.approx(expect.bo_hv_mean)
    assert merged["rs_hv_mean"] == pytest.approx(expect.rs_hv_mean)
    assert (merged["res"], merged["budget"]) == (640, 50)  # metadata carried through


def test_merge_cli_short_circuits_before_the_lut_and_writes_combined(tmp_path):
    """`search.bo --merge a.json b.json --out m.json` needs no LUT/GPU."""
    def part(path, runs):
        path.write_text(json.dumps({
            "passes": False, "n_seeds": len(runs), "bo_hv_mean": 0.0,
            "bo_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0,
            "t_max_ms": 16.7, "res": 224, "budget": 8, "structural": False,
            "runs": runs}))

    p0, p1 = tmp_path / "part0.json", tmp_path / "part1.json"
    part(p0, [{"seed": 0, "bo_hv": 1.0, "rs_hv": 0.7, "bo_frontier": []},
              {"seed": 1, "bo_hv": 1.02, "rs_hv": 0.72, "bo_frontier": []}])
    part(p1, [{"seed": 2, "bo_hv": 0.98, "rs_hv": 0.68, "bo_frontier": []}])
    out = tmp_path / "merged.json"

    rc = main(["--merge", str(p0), str(p1), "--out", str(out)])
    assert rc == 0
    merged = json.loads(out.read_text())
    assert [r["seed"] for r in merged["runs"]] == [0, 1, 2]
    assert merged["n_seeds"] == 3


# ---- cross-session resume: cache the random control, bound a session, track done -

def test_random_search_control_resume_skips_already_evaluated(synth_lut, tmp_path):
    """The random-search control caches + resumes too, so the DoD's baseline half
    survives a Kaggle session boundary instead of restarting from zero."""
    cache = tmp_path / "rs_cache.jsonl"
    calls = {"n": 0}

    def counting_eval(arch):
        calls["n"] += 1
        return depth_sum_accuracy(arch)

    rs1 = random_search_control(counting_eval, synth_lut, budget=6, seed=1000,
                                t_max=0.18, res=224, cache_path=cache)
    first = calls["n"]
    assert first == 6 and cache.exists() and rs1.n_evals == 6

    rs2 = random_search_control(counting_eval, synth_lut, budget=6, seed=1000,
                                t_max=0.18, res=224, cache_path=cache)
    assert calls["n"] == first      # nothing re-evaluated
    assert rs2.n_evals == 6         # the cache repopulates the run


def test_load_eval_cache_tolerates_a_truncated_final_line(tmp_path):
    """A session killed mid-append leaves a partial last line; resume must not choke."""
    from search.bo import _load_eval_cache
    rec = {"arch": _random_arch_dict(random.Random(0)), "acc": 0.5,
           "latency_ms": 0.1, "acc_eff": 0.5}
    cache = tmp_path / "c.jsonl"
    cache.write_text(json.dumps(rec) + "\n" + '{"arch": {"d": [2, 3')  # truncated write
    evals, done = _load_eval_cache(cache)
    assert len(evals) == 1 and len(done) == 1


def test_load_acc_memo_averages_duplicate_archs(tmp_path):
    """The same canonical backbone seen under several seeds collapses to one mean acc."""
    from search.bo import load_acc_memo
    arch = _random_arch_dict(random.Random(1))
    memo_file = tmp_path / "memo.json"
    memo_file.write_text(json.dumps([
        {"arch": arch, "acc": 0.40}, {"arch": arch, "acc": 0.60},  # same arch, two seeds
        {"arch": _random_arch_dict(random.Random(2)), "acc": 0.30},
    ]))
    memo = load_acc_memo(memo_file)
    assert len(memo) == 2
    assert memo[tuple(canonical(encode(arch)))] == pytest.approx(0.50)  # (0.40 + 0.60) / 2


def test_load_acc_memo_missing_file_is_empty(tmp_path):
    from search.bo import load_acc_memo
    assert load_acc_memo(None) == {}
    assert load_acc_memo(tmp_path / "absent.json") == {}


def test_memoized_evaluator_hits_memo_and_falls_through_on_miss():
    """A hit returns the memo value without calling the oracle; a miss delegates."""
    from search.bo import memoized_evaluator
    known = _random_arch_dict(random.Random(3))
    unknown = _random_arch_dict(random.Random(4))
    memo = {tuple(canonical(encode(known))): 0.77}
    calls = []
    def oracle(arch):
        calls.append(arch)
        return 0.11
    hits: list[int] = []
    ev = memoized_evaluator(oracle, memo, hits=hits)
    assert ev(known) == 0.77        # served from memo
    assert calls == []              # oracle untouched
    assert ev(unknown) == 0.11      # fell through to the oracle
    assert calls == [unknown]
    assert sum(hits) == 1


def test_run_bo_stops_at_the_deadline_and_reports_incomplete(synth_lut):
    """A past deadline makes run_bo return at once with complete=False, so the caller
    knows the seed must be resumed next session."""
    pytest.importorskip("botorch")
    bo = run_bo(depth_sum_accuracy, synth_lut, budget=50, n_init=4, seed=0,
                t_max=0.18, res=224, deadline=time.monotonic() - 1.0)
    assert bo.n_evals < 50
    assert bo.complete is False


def test_random_search_control_stops_at_the_deadline_and_reports_incomplete(synth_lut):
    rs = random_search_control(depth_sum_accuracy, synth_lut, budget=50, seed=1000,
                               t_max=0.18, res=224, deadline=time.monotonic() - 1.0)
    assert rs.n_evals < 50
    assert rs.complete is False


def test_merge_reports_incomplete_when_any_seed_is_unfinished():
    """A multi-session DoD is valid only once every seed has spent its full budget;
    the merge surfaces that as ``complete`` so a runner knows when to stop resuming."""
    def run(seed, complete):
        return {"seed": seed, "bo_hv": 1.0, "rs_hv": 0.7, "bo_frontier": [],
                "complete": complete}

    def payload(runs):
        return {"passes": False, "n_seeds": len(runs), "bo_hv_mean": 0.0,
                "bo_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0,
                "t_max_ms": 16.7, "res": 224, "budget": 50, "structural": False,
                "runs": runs}

    done = merge_bo_outputs([payload([run(0, True), run(1, True)])])
    assert done["complete"] is True
    partial = merge_bo_outputs([payload([run(0, True), run(1, False)])])
    assert partial["complete"] is False


# ---- dynamic seed -> GPU rebalancing across resumed sessions -----------------

def test_seeds_to_run_prefers_explicit_list_else_contiguous_range():
    from search.bo import _seeds_to_run
    assert _seeds_to_run("2,4", 0, 5) == [2, 4]          # explicit list wins
    assert _seeds_to_run(None, 3, 2) == [3, 4]           # else the contiguous range
    assert _seeds_to_run("", 0, 3) == [0, 1, 2]          # empty string -> range fallback
    assert _seeds_to_run(" 1 , 0 ", 0, 9) == [1, 0]      # space-tolerant, order preserved


def test_seed_remaining_evals_counts_both_shards(tmp_path):
    base = tmp_path / "cp33_bo_cache_r224"

    def shard(s, suffix, n):
        (tmp_path / f"cp33_bo_cache_r224.seed{s}.{suffix}.jsonl").write_text(
            "".join(f'{{"i":{i}}}\n' for i in range(n)))

    shard(0, "bo", 50)
    shard(0, "rs", 6)
    assert seed_remaining_evals(base, 0, budget=50) == 44   # (50-50) + (50-6)
    assert seed_remaining_evals(base, 9, budget=50) == 100  # no shards -> owes full budget
    shard(1, "bo", 50)
    shard(1, "rs", 50)
    assert seed_remaining_evals(base, 1, budget=50) == 0    # both full -> done
    shard(2, "bo", 60)
    shard(2, "rs", 60)                                      # over-budget clamps at 0
    assert seed_remaining_evals(base, 2, budget=50) == 0


def test_assign_seeds_lpt_balances_remaining_work_and_covers_every_seed():
    # remaining evals per seed (seeds 0,3 partially done); LPT balances LOAD, not count
    rem = {0: 44, 1: 100, 2: 100, 3: 44, 4: 100}
    a = assign_seeds_to_gpus(rem, ngpu=2)
    assert sorted(s for g in a for s in g) == [0, 1, 2, 3, 4]      # every seed exactly once
    loads = [sum(rem[s] for s in g) for g in a]
    assert max(loads) - min(loads) <= max(rem.values())           # balanced within one job
    # a naive 3/2 round-robin by count would give 244 vs 144; LPT must do better
    assert max(loads) - min(loads) < 244 - 144
    # all-done seeds are still placed so the merge sees every seed
    a2 = assign_seeds_to_gpus({0: 0, 1: 0, 2: 0}, ngpu=2)
    assert sorted(s for g in a2 for s in g) == [0, 1, 2]
