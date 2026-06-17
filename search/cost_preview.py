"""Offline cost-landscape preview over the finished Jetson LUT.

Once the real per-block latency LUT is complete (``data/lut.jsonl``), the whole
search space's *cost geometry* is computable on a laptop — no Jetson, no CUDA, no
fine-tune. This module samples archs, composes each one's cost from the LUT
(``search.cost.cost``), and reports:

**The headline — was the measured LUT worth building?** A FLOPs (or params) proxy
is free; the Jetson sweep was not. If measured latency is *rank-correlated* with
FLOPs across the space, a proxy would have ranked archs just as well and the LUT
bought little. If they *diverge* (the usual story on a real accelerator — memory-
bound depthwise convs, SE overhead, fusion quirks), the LUT is justified and the
search will surface archs a FLOPs proxy would mis-rank. We quantify that with the
Spearman and Kendall-tau rank correlation of ``flops``-vs-``latency`` and
``params``-vs-``latency``, plus the latency spread *within* fixed-FLOPs bins.

It also reports the cost dynamic range the search must navigate, the min/max-cost
corners of the space, and a **cost-only non-dominated set** over
``(latency, params)`` / ``(latency, flops)`` — a *structural* preview of the
frontier, NOT the deployable Pareto front, which needs the accuracy axis from
CP 2.4 (still gated on CUDA + dataset decision D1).

Runs under ``.venv`` (CPU). numpy + pandas only — no scipy/matplotlib; the
per-arch table is written to CSV for plotting elsewhere.

    .venv/bin/python -m search.cost_preview --n 2000
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from catalog.contracts import ArchDict, LutRow
from catalog.ofa_mbv3 import KS, MAX_DEPTH, E
from lut.loader import load_lut
from search.arch_to_blocks import arch_to_blocks, random_arch_dict
from search.cost import CostError, cost, resident_mem_mib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUT = ROOT / "data" / "lut.jsonl"
DEFAULT_CSV = ROOT / "data" / "cost_preview.csv"

# Bytes per weight at a given precision — for the resident-memory projection.
_BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "int8": 1}


def bytes_per_param(precision: str | None) -> int:
    """Weight byte-width for ``precision`` (defaults to fp32 = 4)."""
    return _BYTES_PER_PARAM.get(precision or "fp32", 4)


# --- rank statistics (numpy/pandas only; no scipy) ---------------------------

def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties get average ranks). NaN if either is constant."""
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b (tie-corrected), vectorised per row — O(n^2) time, O(n) memory.

    tau-b = (concordant - discordant) / sqrt((n0 - n1)(n0 - n2)) where n0 is the
    total pair count, n1/n2 the pairs tied on x / on y. Robust to the integer ties
    in ``flops``/``params``; latency rarely ties.
    """
    xa = np.asarray(x, float)
    ya = np.asarray(y, float)
    n = len(xa)
    if n < 2:
        return float("nan")
    nc = nd = n1 = n2 = 0
    for i in range(n - 1):
        dx = np.sign(xa[i + 1:] - xa[i])
        dy = np.sign(ya[i + 1:] - ya[i])
        prod = dx * dy
        nc += int(np.count_nonzero(prod > 0))
        nd += int(np.count_nonzero(prod < 0))
        n1 += int(np.count_nonzero(dx == 0))
        n2 += int(np.count_nonzero(dy == 0))
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return (nc - nd) / denom if denom > 0 else float("nan")


def nondominated_indices(xs: Sequence[float], ys: Sequence[float]) -> list[int]:
    """Indices of the non-dominated (Pareto-min) set minimising both ``xs`` and ``ys``.

    A point is dominated when another is <= on both axes and < on at least one.
    Sort by x (ties by y), sweep keeping the running-min y — the staircase that
    survives is the skyline.
    """
    xa = np.asarray(xs, float)
    ya = np.asarray(ys, float)
    order = np.lexsort((ya, xa))  # primary: xa asc; secondary: ya asc
    keep: list[int] = []
    best_y = math.inf
    for i in order:
        if ya[i] < best_y:
            keep.append(int(i))
            best_y = ya[i]
    return sorted(keep)


# --- sampling + composition --------------------------------------------------

def cost_row(arch: ArchDict, lut: dict[str, LutRow], precision: str | None) -> dict:
    """One arch's costed record (depth + the four cost fields + resident memory)."""
    c = cost(arch, lut)
    return {
        "depth": len(arch_to_blocks(arch)),
        "latency_ms": c["latency_ms"],
        "params": c["params"],
        "flops": c["flops"],
        "peak_mem_mib": c["peak_mem_mib"],
        "resident_mem_mib": resident_mem_mib(c, bytes_per_param(precision)),
    }


def sample_costs(
    lut: dict[str, LutRow], n: int, *, seed: int = 0, precision: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Cost ``n`` random archs against ``lut``; return (DataFrame, n_coverage_gaps).

    A coverage gap (a block absent from the precision-filtered LUT) is counted and
    skipped, never silently costed as zero. With the complete 2710-row LUT and
    OFA-reachable-only archs this is 0 (guaranteed by arch_to_blocks' DoD).
    """
    rng = random.Random(seed)
    records: list[dict] = []
    gaps = 0
    for _ in range(n):
        arch = random_arch_dict(rng)
        try:
            records.append(cost_row(arch, lut, precision))
        except CostError:
            gaps += 1
    return pd.DataFrame.from_records(records), gaps


def _corner(d: int, e: int, k: int) -> ArchDict:
    """A space corner: every stage depth ``d``, every block expand ``e`` / kernel ``k``."""
    n = 5 * MAX_DEPTH
    return {"ks": [k] * n, "e": [e] * n, "d": [d] * 5}


# --- reporting ---------------------------------------------------------------

def _fmt_quantiles(s: pd.Series) -> str:
    q = s.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
    return (f"min={q[0.0]:.4g}  p5={q[0.05]:.4g}  median={q[0.5]:.4g}  "
            f"p95={q[0.95]:.4g}  max={q[1.0]:.4g}")


def summarize(df: pd.DataFrame, lut: dict[str, LutRow], precision: str | None) -> dict:
    """Print the cost-landscape report; return the headline numbers as a dict."""
    lat = df["latency_ms"].to_numpy()
    flops = df["flops"].to_numpy()
    params = df["params"].to_numpy()

    rank = {
        "spearman_flops_latency": spearman(flops, lat),
        "kendall_flops_latency": kendall_tau(flops, lat),
        "spearman_params_latency": spearman(params, lat),
        "kendall_params_latency": kendall_tau(params, lat),
    }

    print(f"\nCost-landscape preview — {len(df)} archs, precision={precision or 'fp32'}")
    print("=" * 72)

    print("\n[1] Proxy-vs-measured rank agreement (does the LUT beat a FLOPs proxy?)")
    print(f"    FLOPs  ~ latency : Spearman={rank['spearman_flops_latency']:+.3f}  "
          f"Kendall-tau={rank['kendall_flops_latency']:+.3f}")
    print(f"    params ~ latency : Spearman={rank['spearman_params_latency']:+.3f}  "
          f"Kendall-tau={rank['kendall_params_latency']:+.3f}")
    print("    (1.0 => a proxy ranks archs identically; lower => the measured LUT "
          "captures\n     ordering a proxy misses, i.e. it was worth collecting.)")

    # Latency spread within FLOPs deciles — the divergence a single proxy can't see.
    print("\n[2] Latency spread within FLOPs deciles (proxy-blind variation)")
    worst_ratio = 0.0
    n_bins = min(10, int(df["flops"].nunique()))
    if n_bins >= 2:
        df2 = df.copy()
        df2["flops_bin"] = pd.qcut(df2["flops"], n_bins, labels=False, duplicates="drop")
        for dec, g in df2.groupby("flops_bin"):
            lo, hi = g["latency_ms"].min(), g["latency_ms"].max()
            ratio = hi / lo if lo > 0 else float("nan")
            worst_ratio = max(worst_ratio, ratio if math.isfinite(ratio) else 0.0)
            print(f"    bin {int(dec):2d}: n={len(g):4d}  latency {lo:.4g}..{hi:.4g} ms  "
                  f"(max/min={ratio:.2f}x)")
        print(f"    => at near-equal FLOPs, measured latency varies up to {worst_ratio:.2f}x.")
    else:
        print("    (not enough distinct FLOPs values to bin)")

    print("\n[3] Cost dynamic range (the span the search will navigate)")
    for col, unit in [("latency_ms", "ms"), ("params", ""), ("flops", ""),
                      ("peak_mem_mib", "MiB"), ("resident_mem_mib", "MiB")]:
        print(f"    {col:17s} {unit:3s}: {_fmt_quantiles(df[col])}")

    print("\n[4] Cost corners of the space (depth + cost extremes)")
    for label, arch in [("min-cost (d=2,e=3,k=3)", _corner(2, min(E), min(KS))),
                        ("max-cost (d=4,e=6,k=7)", _corner(4, max(E), max(KS)))]:
        try:
            c = cost(arch, lut)
            print(f"    {label}: depth={len(arch_to_blocks(arch))}  "
                  f"latency={c['latency_ms']:.4g} ms  params={c['params']:,}  "
                  f"flops={c['flops']:,}")
        except CostError as exc:
            print(f"    {label}: coverage gap — {exc}")

    print("\n[5] Cost-only non-dominated set (STRUCTURAL preview — not the real front)")
    for ax, col in [("(latency, params)", "params"), ("(latency, flops)", "flops")]:
        nd = nondominated_indices(lat, df[col].to_numpy())
        print(f"    {ax}: {len(nd)} / {len(df)} archs non-dominated")
    print("    NB: the deployable Pareto front needs accuracy (CP 2.4); this is the "
          "cost\n     skyline only — every point here is just 'cheap', not 'good'.")

    return {"n": len(df), "rank": rank, "flops_decile_latency_spread": worst_ratio}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="LUT path")
    p.add_argument("--precision", default="fp32",
                   help="precision filter (real sweep=fp32, dummy=fp16)")
    p.add_argument("--n", type=int, default=2000, help="archs to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="per-arch CSV output (use '' to skip)")
    args = p.parse_args(argv)

    lut = load_lut(args.lut, precision=args.precision)
    print(f"Loaded {len(lut)} LUT rows from {args.lut} (precision={args.precision})")

    df, gaps = sample_costs(lut, args.n, seed=args.seed, precision=args.precision)
    if gaps:
        print(f"WARNING: {gaps}/{args.n} archs had a LUT coverage gap (skipped). "
              "Expected 0 against a complete LUT — finish the sweep if non-zero.")
    if df.empty:
        print("No archs costed (every sample hit a coverage gap). Is the LUT complete?")
        return 1

    summarize(df, lut, args.precision)

    if str(args.csv):
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nPer-arch table -> {args.csv}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
