"""CP 3.2 — NSGA-II evolutionary search over (depth_sum, latency_ms), via pymoo.

The cheap, CPU-only Phase-3 baseline: maximize ``depth_sum`` (a zero-cost capacity
prior, ρ≈0.84 vs real pose mAP) and minimize ``latency_ms`` (the Jetson LUT cost),
producing a Pareto frontier. **DoD: ≥ 10 non-dominated points.** No GPU / Colab /
Jetson — the fp32 LUT (`data/lut.jsonl`) is read locally.

The GA searches the CP 3.1 length-45 **integer category-index** vector
(`search.space`); every axis has cardinality 3, so pymoo sees a uniform integer
box ``[0, 2]``. Each candidate is decoded to an OFA arch_dict and scored by
:func:`evaluate_objectives`. pymoo is imported lazily inside :func:`run_search`,
so importing this module never requires it (mirrors `eval/shortft.py`'s lazy
torch — keeps the pure helpers testable in `.venv`/CI without pymoo).

The (depth_sum, latency) front is the analytic "depth staircase" — the min-latency
config (ks/e at their smallest) at each achievable depth 10→20 — so it is
intentionally thin; accuracy-richness comes from the CP 3.3 BO over real proxy mAP.
This module's value is the reusable NSGA-II machinery (re-run on the enriched space
at CP 7.2).

Run the DoD::

    python -m search.evolution
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from catalog.contracts import ArchDict, LutRow
from search.cost import cost
from search.space import VECTOR_LEN, decode


def evaluate_objectives(arch_dict: ArchDict, lut: dict[str, LutRow]) -> tuple[float, float]:
    """The two objectives to **minimize**.

    ``f1 = -depth_sum`` (maximize the capacity prior → minimize its negative);
    ``f2 = latency_ms`` (summed LUT cost, ranking-default args). Raises ``CostError``
    (from :func:`search.cost.cost`) if a block has no LUT row.
    """
    f1 = -float(sum(arch_dict["d"]))
    f2 = cost(arch_dict, lut)["latency_ms"]
    return (f1, f2)


def _nondominated_dedup(objs: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated (minimize **both**) set, one per distinct tuple.

    ``b`` dominates ``a`` iff ``b`` is ≤ on both objectives and < on at least one.
    Equal tuples never dominate each other; the first occurrence is kept (so the
    returned indices carry no duplicate objective values).
    """
    keep: list[int] = []
    seen: set[tuple[float, float]] = set()
    for i, oi in enumerate(objs):
        if oi in seen:
            continue
        dominated = any(
            oj[0] <= oi[0] and oj[1] <= oi[1] and (oj[0] < oi[0] or oj[1] < oi[1])
            for j, oj in enumerate(objs)
            if j != i
        )
        if not dominated:
            keep.append(i)
            seen.add(oi)
    return keep


@dataclass(frozen=True)
class FrontierPoint:
    """One non-dominated subnet: its arch and the two real (positive) objectives."""

    arch: ArchDict
    depth_sum: int
    latency_ms: float


@dataclass(frozen=True)
class SearchResult:
    frontier: list[FrontierPoint]  # non-dominated, deduped, sorted by ascending latency
    n_nondominated: int
    n_evals: int  # distinct genotypes scored (memoized)


def run_search(
    lut: dict[str, LutRow], *, pop_size: int = 50, n_gen: int = 100, seed: int = 0
) -> SearchResult:
    """Run NSGA-II and return the (depth_sum, latency) Pareto frontier.

    pymoo is imported here, not at module top. The integer recipe is the documented
    pymoo one: ``IntegerRandomSampling`` + ``SBX``/``PM`` with ``RoundingRepair``.
    Objectives are memoized per genotype so identical archs are scored once.
    """
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize

    memo: dict[tuple[int, ...], tuple[float, float]] = {}

    def objectives_for(vec: tuple[int, ...]) -> tuple[float, float]:
        cached = memo.get(vec)
        if cached is None:
            cached = evaluate_objectives(decode(list(vec)), lut)
            memo[vec] = cached
        return cached

    class _PoseSearchProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=VECTOR_LEN, n_obj=2, xl=0, xu=2, vtype=int)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = list(objectives_for(tuple(int(round(v)) for v in x)))

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0 / VECTOR_LEN, eta=20.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    res = minimize(_PoseSearchProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    f_arr = np.atleast_2d(res.F)
    x_arr = np.atleast_2d(res.X)
    objs = [(float(f_arr[i, 0]), float(f_arr[i, 1])) for i in range(f_arr.shape[0])]
    frontier: list[FrontierPoint] = []
    for i in _nondominated_dedup(objs):
        arch = decode([int(round(v)) for v in x_arr[i]])
        frontier.append(
            FrontierPoint(arch=arch, depth_sum=int(sum(arch["d"])), latency_ms=objs[i][1])
        )
    frontier.sort(key=lambda fp: fp.latency_ms)
    return SearchResult(frontier=frontier, n_nondominated=len(frontier), n_evals=len(memo))


def _write_frontier(result: SearchResult, path: Path) -> None:
    payload = {
        "n_nondominated": result.n_nondominated,
        "n_evals": result.n_evals,
        "frontier": [
            {"arch": fp.arch, "depth_sum": fp.depth_sum, "latency_ms": fp.latency_ms}
            for fp in result.frontier
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _dod_smoke_test(
    pop_size: int = 50,
    n_gen: int = 100,
    seed: int = 0,
    lut_path: str = "data/lut.jsonl",
    out: str = "data/phase3_nsga2_frontier.json",
) -> None:
    """CP 3.2 DoD: a NSGA-II run yields ≥ 10 non-dominated points."""
    from lut.loader import load_lut

    lut = load_lut(Path(lut_path), precision="fp32")
    result = run_search(lut, pop_size=pop_size, n_gen=n_gen, seed=seed)
    print(
        f"frontier: {result.n_nondominated} non-dominated points "
        f"({result.n_evals} unique archs evaluated)"
    )
    for fp in result.frontier:
        print(f"  depth_sum={fp.depth_sum:2d}  latency_ms={fp.latency_ms:.4f}")
    _write_frontier(result, Path(out))
    print(f"wrote {out}")
    ok = result.n_nondominated >= 10
    print("DoD PASS" if ok else "DoD FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _dod_smoke_test()
