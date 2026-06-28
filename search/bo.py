"""CP 3.3 — Bayesian Optimization over the OFA pose search space.

The expensive half of Phase 3: a GP surrogate proposes architectures, each scored
by the **warm-head 5-epoch proxy** (``eval.shortft.short_finetune``, the closed
CP 2.4 accuracy signal) for accuracy and the **LUT** (``search.cost.cost``) for
latency. The objective is D4's ``J(α)`` — multi-objective ``(acc_eff, latency)``
under a hard ceiling ``latency ≤ T_max`` — explored by **ParEGO** (random-weight
scalarizations trace the Pareto front). The DoD: over ≥5 seeds, BO's Pareto
**hypervolume** beats a same-budget random-search control with non-overlapping
across-seed dispersion (`PROJECT_PLAN.md` Phase-3 protocol).

This module splits like ``search.evolution``: the pure helpers below (ParEGO
scalarization, hypervolume, candidate generation, the verdict) are numpy/stdlib
only and unit-tested in ``.venv``/CI (``tests/test_bo.py``); the heavy
``run_bo`` driver lazy-imports ``botorch``/``gpytorch``/``torch`` and the GPU
accuracy oracle. Latency is *deterministic* (exact from the LUT), so only
accuracy is modeled by the GP — the ceiling pre-filters candidates and ParEGO
scalarizes the GP-accuracy against the known latency.

Run the CPU structural smoke (no GPU, no botorch needed beyond the driver)::

    python -m search.bo --structural --seeds 3 --budget 20
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from catalog.contracts import ArchDict, LutRow
from catalog.ofa_mbv3 import KS, MAX_DEPTH, STAGES, D, E
from search.arch_to_blocks import random_arch_dict, validate_arch_dict
from search.cost import cost, resident_mem_mib
from search.objective import DEFAULT_BUDGET_MIB, effective_accuracy, fps_to_ms
from search.space import N_SLOTS, canonical, encode

ROOT = Path(__file__).resolve().parents[1]

# An evaluation point on the two search objectives: (acc_eff ↑, latency_ms ↓).
ParetoPoint = tuple[float, float]


# ---- ParEGO scalarization ----------------------------------------------------

def parego_weights(rng: random.Random, n_obj: int = 2) -> tuple[float, ...]:
    """A weight vector drawn uniformly from the ``n_obj`` simplex (ParEGO, D4).

    Normalised Exp(1) draws are a Dirichlet(1,…,1) sample — i.e. uniform over the
    simplex — so successive BO iterations sweep the whole accuracy/latency
    trade-off rather than a fixed direction. Deterministic given ``rng``.
    """
    raw = [rng.expovariate(1.0) for _ in range(n_obj)]
    total = sum(raw)
    return tuple(r / total for r in raw)


def tchebycheff_scalarize(
    costs: Sequence[float], weights: Sequence[float], *, rho: float = 0.05
) -> float:
    """Augmented Tchebycheff scalarization of normalised ``costs`` (lower = better).

    ``costs`` are objectives to **minimise**, normalised so 0 is the ideal; the
    augmented form ``max_i(w_i c_i) + rho·Σ w_i c_i`` (Knowles 2006) recovers the
    concave parts of the front a plain weighted sum cannot. ParEGO minimises this.
    """
    weighted = [w * c for w, c in zip(weights, costs, strict=True)]
    return max(weighted) + rho * sum(weighted)


# ---- non-domination + 2-D hypervolume (the DoD metric) -----------------------

def nondominated_indices(points: Sequence[Sequence[float]]) -> list[int]:
    """Indices of the non-dominated set when **maximising every** coordinate.

    ``b`` dominates ``a`` iff ``b`` is ≥ on all axes and > on at least one. Equal
    points never dominate each other, so duplicates are all kept (harmless for the
    hypervolume sweep, which adds zero area for a repeat).
    """
    keep: list[int] = []
    for i, pi in enumerate(points):
        dominated = any(
            all(pj[d] >= pi[d] for d in range(len(pi)))
            and any(pj[d] > pi[d] for d in range(len(pi)))
            for j, pj in enumerate(points)
            if j != i
        )
        if not dominated:
            keep.append(i)
    return keep


def hypervolume_2d(points: Sequence[ParetoPoint], ref: ParetoPoint) -> float:
    """Dominated hypervolume of a 2-D point set, **maximising both** axes.

    ``ref`` is the lower-left reference (worse than every kept point). The
    non-dominated front is a decreasing staircase; sweeping it in descending-x
    order and summing ``(x − ref_x)·(y − prev_y)`` (``prev_y`` from ``ref_y``)
    gives the union area of the per-point rectangles.
    """
    nd = [points[i] for i in nondominated_indices(points)]
    nd = [(x, y) for x, y in nd if x > ref[0] and y > ref[1]]
    nd.sort(key=lambda p: p[0], reverse=True)
    hv = 0.0
    prev_y = ref[1]
    for x, y in nd:
        if y <= prev_y:  # already covered by a wider (higher-x) rectangle
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return hv


def pareto_hypervolume(
    evals: Sequence[ParetoPoint], *, ref_acc: float, ref_lat: float
) -> float:
    """Hypervolume of ``(acc_eff ↑, latency_ms ↓)`` evals vs a fixed reference.

    Latency is flipped to a maximisation axis (``-latency``) so the generic
    :func:`hypervolume_2d` applies; ``ref`` = (worst acc, worst latency).
    """
    pts = [(acc, -lat) for acc, lat in evals]
    return hypervolume_2d(pts, ref=(ref_acc, -ref_lat))


# ---- candidate generation under the hard latency ceiling ---------------------

def feasible(arch: ArchDict, lut: dict[str, LutRow], t_max: float, *, res: int = 224) -> bool:
    """True iff the subnet's LUT-predicted latency is within the hard ceiling."""
    return cost(arch, lut, res=res)["latency_ms"] <= t_max


def mutate_arch(arch: ArchDict, rng: random.Random, *, n_edits: int = 1) -> ArchDict:
    """A neighbour of ``arch``: re-roll ``n_edits`` random ks/e/d genes.

    Local moves are how BO refines around promising points (and ParEGO diversifies
    the candidate pool). Edits stay in the OFA choice sets, so the result is always
    a valid arch.
    """
    n_slots = len(STAGES) * MAX_DEPTH
    mut: ArchDict = {
        "ks": list(arch["ks"]), "e": list(arch["e"]), "d": list(arch["d"]),
    }
    for _ in range(n_edits):
        axis = rng.choice(("ks", "e", "d"))
        if axis == "d":
            mut["d"][rng.randrange(len(STAGES))] = rng.choice(D)
        elif axis == "ks":
            mut["ks"][rng.randrange(n_slots)] = rng.choice(KS)
        else:
            mut["e"][rng.randrange(n_slots)] = rng.choice(E)
    return mut


def candidate_pool(
    lut: dict[str, LutRow],
    *,
    t_max: float,
    rng: random.Random,
    res: int = 224,
    evaluated: Sequence[tuple[int, ...]] = (),
    seeds: Sequence[ArchDict] = (),
    size: int = 64,
) -> list[ArchDict]:
    """A deduped, feasible pool of candidate archs for the acquisition step.

    Built from ``seeds`` (e.g. the NSGA-II frontier or current incumbents), their
    mutations, and fresh random archs, then filtered to ``latency ≤ t_max`` and
    de-duplicated by canonical encoding (quotienting out depth-inactive don't-cares),
    excluding anything already in ``evaluated``. Stops once ``size`` candidates are
    found or the random budget is spent.
    """
    excluded = set(evaluated)
    pool: list[ArchDict] = []

    def consider(arch: ArchDict) -> None:
        key = tuple(canonical(encode(arch)))
        if key in excluded:
            return
        if not feasible(arch, lut, t_max, res=res):
            return
        excluded.add(key)
        pool.append(arch)

    for seed in seeds:                       # the incumbents themselves …
        consider(seed)
        for _ in range(2):                   # … and a couple of local neighbours
            consider(mutate_arch(seed, rng, n_edits=1))
            if len(pool) >= size:
                return pool
    # top up with random exploration (bounded attempts so an over-tight ceiling
    # cannot loop forever)
    for _ in range(size * 50):
        if len(pool) >= size:
            break
        consider(random_arch_dict(rng))
    return pool


# ---- the DoD verdict: BO hypervolume beats the random-search control ---------

@dataclass(frozen=True)
class BoVerdict:
    """CP 3.3 DoD: BO's Pareto hypervolume beats same-budget random search.

    ``passes`` encodes the *dominance-across-seeds* statement the Phase-3 protocol
    requires (never a single-run comparison): the BO across-seed band
    ``[mean − std, mean + std]`` must sit entirely above the random-search band.
    """

    bo_hv_mean: float
    bo_hv_std: float
    rs_hv_mean: float
    rs_hv_std: float
    n_seeds: int

    @property
    def passes(self) -> bool:
        return (self.bo_hv_mean - self.bo_hv_std) > (self.rs_hv_mean + self.rs_hv_std)


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n  # population std (across-seed spread)
    return mean, var**0.5


def bo_verdict(bo_hvs: Sequence[float], rs_hvs: Sequence[float]) -> BoVerdict:
    """Assemble the across-seed hypervolume verdict from per-seed HV lists."""
    if not bo_hvs or not rs_hvs:
        raise ValueError("need at least one seed of BO and random-search hypervolume")
    bo_mean, bo_std = _mean_std(bo_hvs)
    rs_mean, rs_std = _mean_std(rs_hvs)
    return BoVerdict(
        bo_hv_mean=bo_mean, bo_hv_std=bo_std,
        rs_hv_mean=rs_mean, rs_hv_std=rs_std,
        n_seeds=min(len(bo_hvs), len(rs_hvs)),
    )


def merge_bo_outputs(payloads: Sequence[dict]) -> dict:
    """Combine per-worker CP 3.3 outputs (disjoint seeds) into one payload, recomputing
    the across-seed verdict over EVERY seed. This is what lets the DoD fan its seeds
    across GPUs (one worker per device) and rejoin them; run metadata (t_max/res/budget)
    is carried from the first payload."""
    if not payloads:
        raise ValueError("need at least one payload to merge")
    runs = sorted((r for p in payloads for r in p["runs"]), key=lambda r: r["seed"])
    v = bo_verdict([r["bo_hv"] for r in runs], [r["rs_hv"] for r in runs])
    base = payloads[0]
    return {
        "passes": v.passes, "n_seeds": v.n_seeds,
        "bo_hv_mean": v.bo_hv_mean, "bo_hv_std": v.bo_hv_std,
        "rs_hv_mean": v.rs_hv_mean, "rs_hv_std": v.rs_hv_std,
        "t_max_ms": base["t_max_ms"], "res": base["res"], "budget": base["budget"],
        "structural": base["structural"], "runs": runs,
    }


# Structural accuracy stub (CPU smoke / random-search control init): depth_sum is
# the zero-cost capacity prior (ρ≈0.84 vs real pose mAP, CP 2.4) — lets the whole
# BO+hypervolume machinery run end-to-end with no GPU, exactly as CP 3.2 does.
def depth_sum_accuracy(arch: ArchDict) -> float:
    """A no-GPU accuracy surrogate = Σ active depths (the zero-cost prior)."""
    validate_arch_dict(arch)
    return float(sum(arch["d"]))


_Evaluator = Callable[[ArchDict], float]


# ---- the BO driver (lazy botorch/torch; GPU only for the real accuracy oracle) ----

@dataclass(frozen=True)
class BoRun:
    """One seed's outcome: the evaluated archs, their non-dominated front, and HV."""

    evals: list[dict]       # [{"arch", "acc", "latency_ms", "acc_eff"}], in eval order
    frontier: list[dict]    # the non-dominated subset, ascending latency
    hypervolume: float
    n_evals: int


def _acc_eff_of(arch: ArchDict, acc: float, lut: dict[str, LutRow], *, res: int,
                mu: float, budget_mib: float, bytes_per_param: int) -> tuple[float, float]:
    """(acc_eff, latency_ms) for one arch — folds the soft μ² memory term into accuracy."""
    c = cost(arch, lut, res=res)
    resident = resident_mem_mib(c, bytes_per_param)
    return effective_accuracy(acc, resident, mu=mu, budget=budget_mib), c["latency_ms"]


def _frontier(evals: Sequence[dict]) -> list[dict]:
    """Non-dominated (acc_eff ↑, latency ↓) subset, ascending latency."""
    pts = [(e["acc_eff"], -e["latency_ms"]) for e in evals]
    keep = [evals[i] for i in nondominated_indices(pts)]
    return sorted(keep, key=lambda e: e["latency_ms"])


def _normalized_costs(
    accs: Sequence[float], lats: Sequence[float]
) -> list[tuple[float, float]]:
    """Per-point ``(acc_cost, lat_cost)`` in ``[0, 1]`` (0 = best) for the scalarization.

    acc is maximised → cost rises as acc falls; latency is minimised → cost rises as
    latency rises. A degenerate (constant) objective contributes zero cost.
    """
    amax, amin = max(accs), min(accs)
    lmax, lmin = max(lats), min(lats)
    arange = (amax - amin) or 1.0
    lrange = (lmax - lmin) or 1.0
    return [((amax - a) / arange, (lat - lmin) / lrange)
            for a, lat in zip(accs, lats, strict=True)]


def _encode_for_gp(arch: ArchDict) -> list[float]:
    """Canonical arch → GP feature vector: ks/e as category labels, depth ∈ [0,1].

    The first ``2*N_SLOTS`` dims are the (masked) ks/e categories — INACTIVE(-1) is
    relabelled 3 so every label is a non-negative integer for the CategoricalKernel;
    the trailing 5 are the ordinal depths, scaled to ``[0,1]`` for the Matérn kernel.
    """
    vec = canonical(encode(arch))
    feats: list[float] = []
    for i, v in enumerate(vec):
        if i < 2 * N_SLOTS:
            feats.append(float(v if v >= 0 else 3))        # ks/e category label
        else:
            feats.append(v / (len(D) - 1))                  # depth 0..2 -> [0,1]
    return feats


def _load_eval_cache(path: Path | None) -> tuple[list[dict], set[tuple[int, ...]]]:
    """Resume: read a JSONL eval cache into (evals, canonical-key set)."""
    evals: list[dict] = []
    done: set[tuple[int, ...]] = set()
    if path is None or not path.exists():
        return evals, done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        evals.append(rec)
        done.add(tuple(canonical(encode(rec["arch"]))))
    return evals, done


def run_bo(
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
    pool_size: int = 128,
    cache_path: Path | None = None,
) -> BoRun:
    """ParEGO Bayesian optimization of ``(acc_eff, latency)`` under ``latency ≤ t_max``.

    ``evaluate_fn(arch) -> accuracy`` is the expensive oracle (the warm-head proxy on
    GPU; :func:`depth_sum_accuracy` for the CPU smoke). Latency is exact from ``lut``.
    Each BO step draws ParEGO weights, refits a ``MixedSingleTaskGP`` on the scalarised
    observations, and picks the discrete feasible candidate maximising ``qLogEI``.
    Resumable: every eval is appended to ``cache_path`` and skipped on a re-run.

    botorch/gpytorch/torch are imported here so the pure helpers above stay importable
    without the surrogate stack.
    """
    import warnings

    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MixedSingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = random.Random(seed)
    torch.manual_seed(seed)
    ref_lat = t_max if ref_lat is None else ref_lat
    cat_dims = list(range(2 * N_SLOTS))

    evals, done = _load_eval_cache(cache_path)

    def record(arch: ArchDict) -> None:
        key = tuple(canonical(encode(arch)))
        if key in done:
            return
        acc = evaluate_fn(arch)
        acc_eff, latency = _acc_eff_of(
            arch, acc, lut, res=res, mu=mu, budget_mib=budget_mib,
            bytes_per_param=bytes_per_param)
        rec = {"arch": arch, "acc": acc, "latency_ms": latency, "acc_eff": acc_eff}
        evals.append(rec)
        done.add(key)
        if cache_path is not None:
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # --- initial design: NSGA-II seeds first, then random feasible archs ---------
    for arch in candidate_pool(lut, t_max=t_max, rng=rng, res=res,
                               evaluated=list(done), seeds=seed_archs, size=n_init):
        if len(evals) >= min(n_init, budget):
            break
        record(arch)
    _attempts = 0
    while len(evals) < min(n_init, budget) and _attempts < n_init * 200:
        _attempts += 1
        cand = random_arch_dict(rng)
        if feasible(cand, lut, t_max, res=res):
            record(cand)

    # --- BO loop: ParEGO scalarization -> GP -> qLogEI over the feasible pool -----
    while len(evals) < budget:
        weights = parego_weights(rng)
        costs = _normalized_costs([e["acc_eff"] for e in evals],
                                  [e["latency_ms"] for e in evals])
        # maximize -g (g = augmented Tchebycheff cost, lower is better)
        y = [-tchebycheff_scalarize(c, weights) for c in costs]
        train_x = torch.tensor([_encode_for_gp(e["arch"]) for e in evals],
                               dtype=torch.double)
        train_y = torch.tensor(y, dtype=torch.double).unsqueeze(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # GP fit input/standardize chatter
            gp = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acq = qLogExpectedImprovement(gp, best_f=train_y.max())

            pool = candidate_pool(lut, t_max=t_max, rng=rng, res=res,
                                  evaluated=list(done),
                                  seeds=[e["arch"] for e in _frontier(evals)],
                                  size=pool_size)
            if not pool:
                break  # feasible space exhausted (degenerate / over-tight ceiling)
            cand_x = torch.tensor([_encode_for_gp(a) for a in pool], dtype=torch.double)
            scores = acq(cand_x.unsqueeze(1))           # (n_pool, 1, d) -> (n_pool,)
        record(pool[int(scores.argmax())])

    hv = pareto_hypervolume([(e["acc_eff"], e["latency_ms"]) for e in evals],
                            ref_acc=ref_acc, ref_lat=ref_lat)
    return BoRun(evals=evals, frontier=_frontier(evals), hypervolume=hv,
                 n_evals=len(evals))


def random_search_control(
    evaluate_fn: _Evaluator,
    lut: dict[str, LutRow],
    *,
    budget: int,
    seed: int,
    t_max: float,
    res: int = 224,
    ref_acc: float = 0.0,
    ref_lat: float | None = None,
    mu: float = 0.0,
    budget_mib: float = DEFAULT_BUDGET_MIB,
    bytes_per_param: int = 2,
) -> BoRun:
    """The same-budget random-search baseline the DoD compares BO against.

    Samples ``budget`` feasible archs uniformly and scores them with the *same*
    ``evaluate_fn`` and reference point — so a hypervolume gap is attributable to the
    GP-guided search, not to a different objective or budget.
    """
    rng = random.Random(seed)
    ref_lat = t_max if ref_lat is None else ref_lat
    evals: list[dict] = []
    attempts = 0
    while len(evals) < budget and attempts < budget * 200:
        attempts += 1
        arch = random_arch_dict(rng)
        if not feasible(arch, lut, t_max, res=res):
            continue
        acc = evaluate_fn(arch)
        acc_eff, latency = _acc_eff_of(
            arch, acc, lut, res=res, mu=mu, budget_mib=budget_mib,
            bytes_per_param=bytes_per_param)
        evals.append({"arch": arch, "acc": acc, "latency_ms": latency, "acc_eff": acc_eff})
    hv = pareto_hypervolume([(e["acc_eff"], e["latency_ms"]) for e in evals],
                            ref_acc=ref_acc, ref_lat=ref_lat)
    return BoRun(evals=evals, frontier=_frontier(evals), hypervolume=hv,
                 n_evals=len(evals))


# ---- CLI: structural CPU smoke / timed calibration / real warm-head search ----

def _build_real_evaluator(args) -> _Evaluator:
    """The GPU accuracy oracle: warm-head 5-epoch proxy (CP 2.4) per arch.

    Lazy — loads the OFA supernet + ultralytics once, then fine-tunes each sampled
    backbone under the frozen gate head. Needs ``.venv-nas``/Kaggle (GPU, ofa,
    ultralytics, the dataset). Imported only when a real run is requested.
    """
    from eval.shortft import short_finetune
    from supernet.sampler import load_supernet

    supernet = load_supernet()

    def evaluate(arch: ArchDict) -> float:
        return short_finetune(
            dict(arch), epochs=args.proxy_epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, supernet=supernet,
            head_weights=args.head_weights, freeze_head=args.freeze_head,
        )["map"]

    return evaluate


def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    from lut.loader import load_lut

    p = argparse.ArgumentParser(description="CP 3.3 — ParEGO Bayesian optimization")
    p.add_argument("--structural", action="store_true",
                   help="no-GPU smoke: accuracy = depth_sum prior (CP 3.2 style)")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="time N real warm-head evals (per-eval wall-clock) and exit")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed index; a worker runs [seed_start, seed_start+seeds) "
                        "so seeds can be fanned across GPUs and merged")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--t-max-ms", type=float, default=fps_to_ms(60),
                   help="hard latency ceiling (default 60 FPS -> 16.7 ms)")
    p.add_argument("--res", type=int, default=224,
                   help="LUT resolution: 224 (measured now) or 640 (owed pose sweep)")
    p.add_argument("--lut", type=Path, default=ROOT / "data" / "lut.jsonl")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--proxy-epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--head-weights", type=Path, default=None)
    p.add_argument("--freeze-head", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "cp33_bo.json")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--merge", nargs="+", type=Path, default=None, metavar="PART",
                   help="merge per-worker output JSONs into --out and exit (no LUT/GPU)")
    a = p.parse_args(argv)

    if a.merge:
        merged = merge_bo_outputs([json.loads(f.read_text()) for f in a.merge])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(merged, indent=2))
        print(f"merged {len(a.merge)} part(s) -> {a.out} ({merged['n_seeds']} seeds, "
              f"{'DoD PASS' if merged['passes'] else 'DoD FAIL'})")
        return 0

    lut = load_lut(a.lut, precision=a.precision)
    seed_archs = _load_nsga_frontier()
    evaluate = depth_sum_accuracy if a.structural else _build_real_evaluator(a)

    if a.calibrate:
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

    bo_hvs, rs_hvs, runs = [], [], []
    for s in range(a.seed_start, a.seed_start + a.seeds):
        cache = (a.cache.with_suffix(f".seed{s}.jsonl") if a.cache else None)
        bo = run_bo(evaluate, lut, budget=a.budget, n_init=a.n_init, seed=s,
                    t_max=a.t_max_ms, res=a.res, seed_archs=seed_archs, cache_path=cache)
        rs = random_search_control(evaluate, lut, budget=a.budget, seed=1000 + s,
                                   t_max=a.t_max_ms, res=a.res)
        bo_hvs.append(bo.hypervolume)
        rs_hvs.append(rs.hypervolume)
        runs.append({"seed": s, "bo_hv": bo.hypervolume, "rs_hv": rs.hypervolume,
                     "bo_frontier": bo.frontier})
        print(f"seed {s}: BO HV={bo.hypervolume:.4f}  random HV={rs.hypervolume:.4f}")

    verdict = bo_verdict(bo_hvs, rs_hvs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "passes": verdict.passes, "n_seeds": verdict.n_seeds,
        "bo_hv_mean": verdict.bo_hv_mean, "bo_hv_std": verdict.bo_hv_std,
        "rs_hv_mean": verdict.rs_hv_mean, "rs_hv_std": verdict.rs_hv_std,
        "t_max_ms": a.t_max_ms, "res": a.res, "budget": a.budget,
        "structural": a.structural, "runs": runs,
    }, indent=2))
    print(f"BO HV {verdict.bo_hv_mean:.4f}±{verdict.bo_hv_std:.4f} vs "
          f"random {verdict.rs_hv_mean:.4f}±{verdict.rs_hv_std:.4f} -> "
          f"{'DoD PASS' if verdict.passes else 'DoD FAIL'}")
    print(f"wrote {a.out}")
    return 0 if verdict.passes else 1


def _load_nsga_frontier(path: Path | None = None) -> list[ArchDict]:
    """The CP 3.2 NSGA-II frontier archs as BO warm-start seeds (empty if absent)."""
    path = ROOT / "data" / "phase3_nsga2_frontier.json" if path is None else path
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [fp["arch"] for fp in payload.get("frontier", [])]


if __name__ == "__main__":
    raise SystemExit(main())
