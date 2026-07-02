"""CP 3.4 — TPE fallback (Optuna MOTPE) search, the CP 3.3 dominance test re-run.

``search.tpe`` reuses ``search.bo``'s method-agnostic machinery (hypervolume, the
hard-ceiling feasibility filter, the random-search control, the across-seed verdict,
the resumable cache) and swaps only the *proposer*: BO's ``MixedSingleTaskGP + qLogEI``
becomes Optuna's multi-objective ``TPESampler`` (MOTPE). These tests exercise the
arch<->Optuna-params bijection (pure) and the ``run_tpe`` driver (optuna-gated,
CI-skips if absent), mirroring ``tests/test_bo.py``.

The synthetic LUT gives every catalog MBConv row a constant 0.01 ms latency, so a
subnet's predicted latency is ``0.01 * n_blocks`` — monotone in depth, which makes
the ceiling filter deterministic to test (identical fixture to ``test_bo``).
"""
import json
import random
import time

import pytest

from search.arch_to_blocks import _random_arch_dict
from search.bo import depth_sum_accuracy, random_search_control
from search.space import VECTOR_LEN, canonical, encode
from search.tpe import _arch_to_params, _params_to_arch, run_tpe


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
    from catalog.sweep import iter_sweep
    return {k: _row(0.01, k) for _, _, _, k in iter_sweep(["mbconv"])}


# ---- arch <-> Optuna params bijection (pure; no optuna import needed) ----------

def test_arch_params_roundtrip_is_lossless():
    """``_params_to_arch(_arch_to_params(a)) == a`` — the flat-index param mapping is
    a bijection, so a cached arch replays into an Optuna trial verbatim on resume."""
    rng = random.Random(0)
    for _ in range(100):
        arch = _random_arch_dict(rng)
        assert _params_to_arch(_arch_to_params(arch)) == arch


def test_arch_to_params_covers_every_slot():
    """One param per surrogate-vector slot (the full ks|e|d layout), so nothing is
    silently dropped from the TPE search space."""
    params = _arch_to_params(_random_arch_dict(random.Random(1)))
    assert len(params) == VECTOR_LEN
    assert all(isinstance(v, int) for v in params.values())


# ---- run_tpe driver (needs Optuna; CI-skips if absent) ------------------------

def test_run_tpe_drives_the_loop_to_a_frontier(synth_lut):
    """The full MOTPE ask/tell loop runs and honors the hard latency ceiling."""
    pytest.importorskip("optuna")
    tpe = run_tpe(depth_sum_accuracy, synth_lut, budget=8, n_init=4, seed=0,
                  t_max=0.18, res=224)
    assert tpe.n_evals == 8
    assert tpe.complete is True
    assert tpe.hypervolume > 0.0
    assert tpe.frontier                                    # >=1 non-dominated point
    assert all(e["latency_ms"] <= 0.18 + 1e-9 for e in tpe.evals)   # ceiling honored


def test_run_tpe_is_seed_deterministic(synth_lut):
    """Same seed -> same evaluated archs and hypervolume (MOTPE is seeded)."""
    pytest.importorskip("optuna")
    a = run_tpe(depth_sum_accuracy, synth_lut, budget=8, n_init=4, seed=3, t_max=0.20)
    b = run_tpe(depth_sum_accuracy, synth_lut, budget=8, n_init=4, seed=3, t_max=0.20)
    assert a.hypervolume == pytest.approx(b.hypervolume)
    assert [e["arch"] for e in a.evals] == [e["arch"] for e in b.evals]


def test_run_tpe_resume_skips_already_evaluated(synth_lut, tmp_path):
    """A re-run with the same cache loads prior evals and never re-evaluates them —
    the cross-session / cross-backend resume the campaign depends on."""
    pytest.importorskip("optuna")
    cache = tmp_path / "tpe_cache.jsonl"
    calls = {"n": 0}

    def counting_eval(arch):
        calls["n"] += 1
        return depth_sum_accuracy(arch)

    run_tpe(counting_eval, synth_lut, budget=6, n_init=3, seed=0,
            t_max=0.18, res=224, cache_path=cache)
    first = calls["n"]
    assert first == 6 and cache.exists()

    tpe2 = run_tpe(counting_eval, synth_lut, budget=6, n_init=3, seed=0,
                   t_max=0.18, res=224, cache_path=cache)
    assert calls["n"] == first      # nothing re-evaluated
    assert tpe2.n_evals == 6        # the cached evals repopulate the run


def test_run_tpe_stops_at_the_deadline_and_reports_incomplete(synth_lut):
    """A past deadline returns at once with complete=False so the seed is resumed."""
    pytest.importorskip("optuna")
    tpe = run_tpe(depth_sum_accuracy, synth_lut, budget=50, n_init=4, seed=0,
                  t_max=0.18, res=224, deadline=time.monotonic() - 1.0)
    assert tpe.n_evals < 50
    assert tpe.complete is False


def test_run_tpe_beats_random_on_the_structural_smoke(synth_lut):
    """CP 3.4 DoD (spirit): MOTPE's Pareto hypervolume beats the *same-budget,
    same-oracle* random-search control. On the depth_sum objective random over-samples
    the middle (sum d ~ 15) and rarely reaches the front's corners, while MOTPE actively
    pushes to both (max acc / min latency), so its frontier is wider -> higher HV.
    Non-binding ceiling (t_max above the deepest arch) isolates search quality."""
    pytest.importorskip("optuna")
    budget, n_init, t_max = 24, 6, 0.25
    tpe_hvs, rs_hvs = [], []
    for s in range(4):
        tpe = run_tpe(depth_sum_accuracy, synth_lut, budget=budget, n_init=n_init,
                      seed=s, t_max=t_max, res=224)
        rs = random_search_control(depth_sum_accuracy, synth_lut, budget=budget,
                                   seed=1000 + s, t_max=t_max, res=224)
        assert tpe.complete and tpe.n_evals == budget
        tpe_hvs.append(tpe.hypervolume)
        rs_hvs.append(rs.hypervolume)
    assert sum(tpe_hvs) / len(tpe_hvs) > sum(rs_hvs) / len(rs_hvs)


def test_run_tpe_evals_are_canonically_deduped(synth_lut):
    """Every recorded eval is a distinct canonical arch — the budget buys distinct
    architectures, not re-scored duplicates (the HV must not be padded by repeats)."""
    pytest.importorskip("optuna")
    tpe = run_tpe(depth_sum_accuracy, synth_lut, budget=12, n_init=4, seed=1, t_max=0.20)
    keys = [tuple(canonical(encode(e["arch"]))) for e in tpe.evals]
    assert len(keys) == len(set(keys))


# ---- Kaggle dual-GPU fan-out: method-aware remaining-work + per-worker merge --------

def test_seed_remaining_evals_is_method_aware(tmp_path):
    """A TPE session must count its OWN '.tpe.' shards (plus the shared '.rs.'), not the
    CP 3.3 '.bo.' shards — otherwise the dual-GPU rebalancer thinks TPE is already done."""
    from search.bo import seed_remaining_evals
    base = tmp_path / "cp33_bo_cache_r640"

    def shard(s, suffix, n):
        (tmp_path / f"cp33_bo_cache_r640.seed{s}.{suffix}.jsonl").write_text(
            "".join(f'{{"i":{i}}}\n' for i in range(n)))

    shard(0, "bo", 50)     # a finished CP 3.3 BO seed ...
    shard(0, "tpe", 30)    # ... but TPE is only 30/50 ...
    shard(0, "rs", 50)     # ... with the random control reused (full)
    assert seed_remaining_evals(base, 0, 50, method="tpe") == 20   # (50-30) + (50-50)
    assert seed_remaining_evals(base, 0, 50) == 0                  # default 'bo': both full


def test_merge_tpe_outputs_concatenates_seeds_and_recomputes_verdict():
    """Per-worker TPE outputs (disjoint seeds) combine into one, verdict over ALL seeds —
    the dual-T4 fan-out's rejoin, mirroring merge_bo_outputs but on 'tpe_hv' keys."""
    from search.bo import bo_verdict
    from search.tpe import merge_tpe_outputs

    def run(seed, tpe, rs):
        return {"seed": seed, "tpe_hv": tpe, "rs_hv": rs, "tpe_frontier": []}

    def payload(runs):
        return {"method": "tpe", "passes": False, "n_seeds": len(runs), "tpe_hv_mean": 0.0,
                "tpe_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0, "t_max_ms": 12.75,
                "res": 640, "budget": 50, "structural": False, "runs": runs}

    a = payload([run(0, 1.00, 0.70), run(1, 1.02, 0.72), run(2, 0.98, 0.68)])
    b = payload([run(3, 1.01, 0.69), run(4, 0.99, 0.71)])
    merged = merge_tpe_outputs([b, a])  # out of order -> must sort by seed

    assert [r["seed"] for r in merged["runs"]] == [0, 1, 2, 3, 4]
    assert merged["n_seeds"] == 5
    expect = bo_verdict([1.00, 1.02, 0.98, 1.01, 0.99], [0.70, 0.72, 0.68, 0.69, 0.71])
    assert merged["passes"] == expect.passes
    assert merged["tpe_hv_mean"] == pytest.approx(expect.bo_hv_mean)
    assert (merged["res"], merged["budget"]) == (640, 50)  # metadata carried through


def test_merge_cli_short_circuits_and_writes_combined_tpe(tmp_path):
    """`search.tpe --merge a.json b.json --out m.json` needs no LUT/GPU (parity w/ bo)."""
    from search.tpe import main

    def part(path, runs):
        path.write_text(json.dumps({
            "method": "tpe", "passes": False, "n_seeds": len(runs), "tpe_hv_mean": 0.0,
            "tpe_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0, "t_max_ms": 12.75,
            "res": 640, "budget": 50, "structural": False, "runs": runs}))

    p0, p1 = tmp_path / "part0.json", tmp_path / "part1.json"
    part(p0, [{"seed": 0, "tpe_hv": 1.0, "rs_hv": 0.7, "tpe_frontier": []},
              {"seed": 1, "tpe_hv": 1.02, "rs_hv": 0.72, "tpe_frontier": []}])
    part(p1, [{"seed": 2, "tpe_hv": 0.98, "rs_hv": 0.68, "tpe_frontier": []}])
    out = tmp_path / "merged.json"

    rc = main(["--merge", str(p0), str(p1), "--out", str(out)])
    assert rc == 0
    merged = json.loads(out.read_text())
    assert [r["seed"] for r in merged["runs"]] == [0, 1, 2]
    assert merged["n_seeds"] == 3
