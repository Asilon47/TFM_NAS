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

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from catalog.contracts import ArchDict, LutRow
from catalog.ofa_mbv3 import KS, MAX_DEPTH, STAGES, D, E
from search.arch_to_blocks import random_arch_dict, validate_arch_dict
from search.cost import cost
from search.space import canonical, encode

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


# Structural accuracy stub (CPU smoke / random-search control init): depth_sum is
# the zero-cost capacity prior (ρ≈0.84 vs real pose mAP, CP 2.4) — lets the whole
# BO+hypervolume machinery run end-to-end with no GPU, exactly as CP 3.2 does.
def depth_sum_accuracy(arch: ArchDict) -> float:
    """A no-GPU accuracy surrogate = Σ active depths (the zero-cost prior)."""
    validate_arch_dict(arch)
    return float(sum(arch["d"]))


_Evaluator = Callable[[ArchDict], float]
