"""CP 3.4 — TPE fallback (Optuna MOTPE), the CP 3.3 dominance test re-run.

The methodological robustness check that follows CP 3.3: swap the Bayesian-optimization
proposer for a **Tree-structured Parzen Estimator** and re-run the *same* DoD — a
5-seed Pareto **hypervolume** that beats a same-budget random-search control
(`PROJECT_PLAN.md` CP 3.4: "Same dominance test as CP 3.3"). BO winning over *both*
TPE and random is the "why a GP surrogate" evidence for the thesis.

This module is deliberately thin: every piece that defines the *comparison* —
:func:`~search.bo.pareto_hypervolume` (the HV metric), :func:`~search.bo.feasible`
(the hard latency ceiling), :func:`~search.bo.random_search_control` (the control TPE
must beat), :func:`~search.bo.bo_verdict` (the across-seed verdict), the resumable
cache, and the warm-head accuracy oracle — is **reused verbatim from** :mod:`search.bo`,
so any hypervolume gap is attributable purely to the proposer. The only new logic is the
MOTPE ask/tell loop below.

**Why MOTPE, not ParEGO+TPE.** BO refits its GP from scratch each step, so ParEGO's
per-step random re-scalarization is free; Optuna's TPE is *incremental* (it models the
accumulated params->value history), so a shifting scalarization would poison that model.
Optuna's ``TPESampler`` with two ``directions`` is genuine multi-objective TPE (MOTPE,
Ozaki et al. 2020) — **not** NSGA-II — and traces the Pareto front directly, giving a
fair, well-implemented fallback whose HV is directly comparable to BO's.

Run the CPU structural smoke (no GPU; accuracy = depth_sum prior)::

    python -m search.tpe --structural --seeds 3 --budget 24
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from catalog.contracts import ArchDict, LutRow

# The comparison machinery is shared with BO so the two methods stay like-for-like.
from search.bo import (
    BoRun,
    _acc_eff_of,
    _frontier,
    _load_eval_cache,
    _seed_label,
    _seeds_to_run,
    bo_verdict,
    depth_sum_accuracy,
    feasible,
    pareto_hypervolume,
)
from search.objective import DEFAULT_BUDGET_MIB, fps_to_ms
from search.space import AXIS_CARDINALITIES, VECTOR_LEN, canonical, decode, encode

ROOT = Path(__file__).resolve().parents[1]

_Evaluator = Callable[[ArchDict], float]  # the expensive accuracy oracle (arch -> mAP)
_PARAM_NAMES: list[str] = [f"x{i}" for i in range(VECTOR_LEN)]


# ---- arch <-> Optuna params (pure; the flat category-index vector as named ints) ----

def _arch_to_params(arch: ArchDict) -> dict[str, int]:
    """OFA arch -> ``{"x0": idx, ...}`` over the length-``VECTOR_LEN`` category vector.

    Reuses :func:`search.space.encode` (the lossless CP 3.1 bijection), so a cached arch
    replays into an Optuna trial verbatim on resume. One named int per surrogate slot.
    """
    return {name: int(v) for name, v in zip(_PARAM_NAMES, encode(arch), strict=True)}


def _params_to_arch(params: Mapping[str, int]) -> ArchDict:
    """Inverse of :func:`_arch_to_params` (``search.space.decode`` of the ordered vector)."""
    return decode([params[name] for name in _PARAM_NAMES])


# ---- the MOTPE driver (lazy optuna; GPU only for the real accuracy oracle) ----------

def run_tpe(
    evaluate_fn: _Evaluator,
    lut: dict[str, LutRow],
    *,
    budget: int = 50,
    n_init: int = 20,
    seed: int = 0,
    t_max: float = 16.7,
    res: int = 224,
    ref_acc: float = 0.0,
    ref_lat: float | None = None,
    mu: float = 0.0,
    budget_mib: float = DEFAULT_BUDGET_MIB,
    bytes_per_param: int = 2,
    seed_archs: Sequence[ArchDict] = (),
    cache_path: Path | None = None,
    deadline: float | None = None,
) -> BoRun:
    """Multi-objective TPE over ``(acc_eff, latency)`` under ``latency <= t_max``.

    Mirrors :func:`search.bo.run_bo` exactly (same signature, same ``BoRun`` result, same
    resumable cache, same reference point) so the CP 3.4 verdict is comparable to CP 3.3's.
    ``evaluate_fn(arch) -> accuracy`` is the expensive oracle (warm-head proxy on GPU;
    :func:`~search.bo.depth_sum_accuracy` for the CPU smoke); latency is exact from ``lut``.

    Each iteration ``ask``s Optuna's MOTPE sampler for an arch, rejects it (``PRUNED``,
    no oracle call) if it breaks the ceiling or repeats a canonical arch, else scores it
    and ``tell``s the sampler ``(acc_eff, latency)``. Optuna is imported here so the pure
    helpers above stay importable without it (parity with ``run_bo``'s lazy botorch).
    """
    import time
    import warnings

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ref_lat = t_max if ref_lat is None else ref_lat

    evals, done = _load_eval_cache(cache_path)
    # canonical-key -> (acc_eff, latency) for O(1) dup/resume replay into the sampler
    done_vals: dict[tuple[int, ...], tuple[float, float]] = {
        tuple(canonical(encode(e["arch"]))): (e["acc_eff"], e["latency_ms"]) for e in evals
    }
    stopped_for_time = False

    # Independent MOTPE (non-experimental): two directions => multi-objective TPE, not
    # NSGA-II. n_startup_trials random draws seed the density model before it takes over.
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init)
    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        name: optuna.distributions.IntDistribution(0, AXIS_CARDINALITIES[i] - 1)
        for i, name in enumerate(_PARAM_NAMES)
    }
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)

    # Warm-start the sampler with cached evals (partial resume) so MOTPE models the prior
    # history instead of cold-starting on the remaining budget.
    for e in evals:
        study.add_trial(optuna.trial.create_trial(
            params=_arch_to_params(e["arch"]),
            distributions=distributions,
            values=[e["acc_eff"], e["latency_ms"]],
        ))
    # Queue the CP 3.2 NSGA-II frontier as the initial design (BO uses these as seeds too).
    for arch in seed_archs:
        study.enqueue_trial(_arch_to_params(arch), skip_if_exists=True)

    def record(arch: ArchDict, acc: float, dt: float) -> tuple[float, float]:
        key = tuple(canonical(encode(arch)))
        acc_eff, latency = _acc_eff_of(
            arch, acc, lut, res=res, mu=mu, budget_mib=budget_mib,
            bytes_per_param=bytes_per_param)
        rec = {"arch": arch, "acc": acc, "latency_ms": latency, "acc_eff": acc_eff}
        evals.append(rec)
        done.add(key)
        done_vals[key] = (acc_eff, latency)
        if cache_path is not None:
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[eval] seed={_seed_label(seed, cache_path)} tpe {len(evals)}/{budget} "
              f"acc={acc:.4f} lat={latency:.2f}ms ({dt:.1f}s)", flush=True)
        return acc_eff, latency

    attempts = 0
    max_attempts = budget * 200  # bound so an over-tight ceiling can't loop forever
    while len(evals) < budget and attempts < max_attempts:
        if deadline is not None and time.monotonic() >= deadline:
            stopped_for_time = True
            break
        attempts += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # optuna sampling chatter
            trial = study.ask(distributions)
        arch = _params_to_arch(trial.params)
        if not feasible(arch, lut, t_max, res=res):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # infeasible: no eval
            continue
        key = tuple(canonical(encode(arch)))
        if key in done:
            # canonical dup (resume replay or a MOTPE re-proposal): feed the sampler the
            # known objective so it still learns, but don't spend an oracle eval on it
            study.tell(trial, list(done_vals[key]))
            continue
        t0 = time.monotonic()
        acc = evaluate_fn(arch)
        dt = time.monotonic() - t0
        acc_eff, latency = record(arch, acc, dt)
        study.tell(trial, [acc_eff, latency])

    hv = pareto_hypervolume([(e["acc_eff"], e["latency_ms"]) for e in evals],
                            ref_acc=ref_acc, ref_lat=ref_lat)
    return BoRun(evals=evals, frontier=_frontier(evals), hypervolume=hv,
                 n_evals=len(evals), complete=not stopped_for_time)


def merge_tpe_outputs(payloads: Sequence[dict]) -> dict:
    """Combine per-worker CP 3.4 outputs (disjoint seeds) into one, recomputing the
    across-seed verdict over EVERY seed — the dual-GPU rejoin. Mirror of
    :func:`search.bo.merge_bo_outputs` on ``tpe_hv`` keys; run metadata carried from the
    first payload."""
    if not payloads:
        raise ValueError("need at least one payload to merge")
    runs = sorted((r for p in payloads for r in p["runs"]), key=lambda r: r["seed"])
    v = bo_verdict([r["tpe_hv"] for r in runs], [r["rs_hv"] for r in runs])
    base = payloads[0]
    return {
        "method": "tpe", "passes": v.passes, "n_seeds": v.n_seeds,
        "tpe_hv_mean": v.bo_hv_mean, "tpe_hv_std": v.bo_hv_std,
        "rs_hv_mean": v.rs_hv_mean, "rs_hv_std": v.rs_hv_std,
        "t_max_ms": base["t_max_ms"], "res": base["res"], "budget": base["budget"],
        "structural": base["structural"],
        "complete": all(r.get("complete", True) for r in runs),
        "runs": runs,
    }


# ---- CLI: structural CPU smoke / timed calibration / real warm-head search ----------

def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    from lut.loader import load_lut
    from search.bo import (
        _build_real_evaluator,
        _load_nsga_frontier,
        load_acc_memo,
        memoized_evaluator,
        random_search_control,
    )

    p = argparse.ArgumentParser(description="CP 3.4 — TPE fallback (Optuna MOTPE)")
    p.add_argument("--structural", action="store_true",
                   help="no-GPU smoke: accuracy = depth_sum prior (CP 3.2 style)")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="time N real warm-head evals (per-eval wall-clock) and exit")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-list", default=None, metavar="S,S,...",
                   help="explicit comma-separated seed indices (overrides --seed-start/--seeds)")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--t-max-ms", type=float, default=fps_to_ms(60),
                   help="hard latency ceiling (default 60 FPS -> 16.7 ms)")
    p.add_argument("--res", type=int, default=224,
                   help="LUT resolution: 224 or 640 (the pose deploy sweep)")
    p.add_argument("--lut", type=Path, default=ROOT / "data" / "lut.jsonl")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--proxy-epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--head-weights", type=Path, default=None)
    p.add_argument("--freeze-head", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "cp34_tpe.json")
    p.add_argument("--cache", type=Path, default=None,
                   help="cache base; TPE shards are '.seed{s}.tpe.jsonl', and the random "
                        "control reuses the CP 3.3 '.seed{s}.rs.jsonl' shards if present")
    p.add_argument("--acc-memo", type=Path, default=None, metavar="JSON",
                   help="prior {arch, acc} measurements consulted before the GPU oracle "
                        "(shared by TPE and random; does NOT bias the dominance test)")
    p.add_argument("--deadline-s", type=int, default=None, metavar="SEC",
                   help="stop starting new evals after SEC seconds and write partial state")
    p.add_argument("--merge", nargs="+", type=Path, default=None, metavar="PART",
                   help="merge per-worker output JSONs into --out and exit (no LUT/GPU)")
    a = p.parse_args(argv)

    if a.merge:
        merged = merge_tpe_outputs([json.loads(f.read_text()) for f in a.merge])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(merged, indent=2))
        print(f"merged {len(a.merge)} part(s) -> {a.out} ({merged['n_seeds']} seeds, "
              f"{'DoD PASS' if merged['passes'] else 'DoD FAIL'})")
        return 0

    lut = load_lut(a.lut, precision=a.precision)
    seed_archs = _load_nsga_frontier()
    evaluate = depth_sum_accuracy if a.structural else _build_real_evaluator(a)
    if a.acc_memo:
        memo = load_acc_memo(a.acc_memo)
        evaluate = memoized_evaluator(evaluate, memo)
        print(f"[acc-memo] {len(memo)} prior measurement(s) loaded from {a.acc_memo} "
              "(free on a hit; shared by TPE and random search)")

    if a.calibrate:
        from search.arch_to_blocks import random_arch_dict
        rng = random.Random(0)
        archs = [arch for arch in (random_arch_dict(rng) for _ in range(a.calibrate * 5))
                 if feasible(arch, lut, a.t_max_ms, res=a.res)][:a.calibrate]
        times = []
        for arch in archs:
            t0 = time.perf_counter()
            evaluate(arch)
            times.append(time.perf_counter() - t0)
        per = sum(times) / len(times)
        gpu_h = 5 * (2 * a.budget - a.n_init) * per / 3600
        print(f"calibration: {len(times)} evals, mean {per:.1f}s/eval "
              f"-> 5-seed budget {a.budget} ~= {gpu_h:.1f} GPU-h")
        return 0

    deadline = time.monotonic() + a.deadline_s if a.deadline_s else None
    tpe_hvs, rs_hvs, runs = [], [], []
    for s in _seeds_to_run(a.seed_list, a.seed_start, a.seeds):
        tpe_cache = a.cache.with_suffix(f".seed{s}.tpe.jsonl") if a.cache else None
        rs_cache = a.cache.with_suffix(f".seed{s}.rs.jsonl") if a.cache else None
        tpe = run_tpe(evaluate, lut, budget=a.budget, n_init=a.n_init, seed=s,
                      t_max=a.t_max_ms, res=a.res, seed_archs=seed_archs,
                      cache_path=tpe_cache, deadline=deadline)
        rs = random_search_control(evaluate, lut, budget=a.budget, seed=1000 + s,
                                   t_max=a.t_max_ms, res=a.res,
                                   cache_path=rs_cache, deadline=deadline)
        tpe_hvs.append(tpe.hypervolume)
        rs_hvs.append(rs.hypervolume)
        done = tpe.complete and rs.complete
        runs.append({"seed": s, "tpe_hv": tpe.hypervolume, "rs_hv": rs.hypervolume,
                     "complete": done, "tpe_frontier": tpe.frontier})
        print(f"seed {s}: TPE HV={tpe.hypervolume:.4f}  random HV={rs.hypervolume:.4f}"
              f"  {'complete' if done else 'PARTIAL (resume next session)'}")

    verdict = bo_verdict(tpe_hvs, rs_hvs)  # method-agnostic: 'bo_*' fields hold TPE here
    all_complete = all(r["complete"] for r in runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "method": "tpe", "passes": verdict.passes, "n_seeds": verdict.n_seeds,
        "tpe_hv_mean": verdict.bo_hv_mean, "tpe_hv_std": verdict.bo_hv_std,
        "rs_hv_mean": verdict.rs_hv_mean, "rs_hv_std": verdict.rs_hv_std,
        "t_max_ms": a.t_max_ms, "res": a.res, "budget": a.budget,
        "structural": a.structural, "complete": all_complete, "runs": runs,
    }, indent=2))
    print(f"TPE HV {verdict.bo_hv_mean:.4f}±{verdict.bo_hv_std:.4f} vs "
          f"random {verdict.rs_hv_mean:.4f}±{verdict.rs_hv_std:.4f} -> "
          f"{'DoD PASS' if verdict.passes else 'DoD FAIL'} "
          f"({'COMPLETE' if all_complete else 'PARTIAL — re-run to resume'})")
    print(f"wrote {a.out}")
    return 0 if verdict.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
