"""Stage-0 follow-up — the @640 additivity study: why deployment runs slower than the LUT sum.

Stage 0 found ONE data point (winner-v1: measured backbone = 1.236 × the raw LUT sum, where
the @224 fit was 0.934×). This module turns the diverse-candidate backbone probe (8 archs
spanning 6.8–27.1 ms sums + the winner) into the answer to three questions:

1. **Is the gap constant (calibration-able) or arch-dependent?** — per-arch ratios + an OLS
   affine fit ``measured ≈ slope·sum + intercept`` at 640 (the honest successor to
   ``data/latency_calibration.json``'s @224 fit).
2. **Does the search's *ranking* survive?** — Spearman ρ between measured and summed
   latencies over the probe set (the load-bearing question: the whole Phase-3 frontier was
   ranked by sums).
3. **Is the error consistent with the DRAM-bound hypothesis?** — the relative error is
   correlated against depth_sum and against the early-stage depth (``d[0]+d[1]``, the
   160×160/80×80 blocks whose activations dominate memory traffic on the 62.5 GB/s Nano).

Pairing rule (uniform across probe sources): each bench row ``data/e2e/<name>.json`` has an
export sidecar ``<name>.meta.json`` whose provenance carries the arch's LUT sum —
``cached_latency_ms`` for de-noise candidates, else the ``latency_ms`` of the record at
``provenance.source`` (corner archs, winner.json).

CLI (laptop, ``.venv``)::

    python -m search.additivity640            # globs data/e2e/*_backbone_640.json
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "e2e"


def pair_from_files(row_path: Path) -> dict | None:
    """One probe pair {name, sum, measured, arch} from a bench row + its export sidecar."""
    row = json.loads(row_path.read_text())
    meta_path = row_path.with_name(row_path.stem + ".meta.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if not meta.get("backbone_only"):
        return None  # e2e rows carry the head; the additivity question is backbone-only
    prov = meta.get("provenance", {})
    lut_sum = prov.get("cached_latency_ms")
    if lut_sum is None and prov.get("source"):
        src = json.loads(Path(prov["source"]).read_text())
        lut_sum = src.get("latency_ms")
    if lut_sum is None:
        return None
    return {
        "name": row.get("name", row_path.stem),
        "sum_ms": float(lut_sum),
        "measured_ms": float(row["latency_ms"]["mean"]),
        "arch_d": meta["arch"]["d"],
        "power_mode": row.get("power_mode"),
        "clocks_locked": row.get("clocks_locked"),
    }


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Plain OLS fit ``y ≈ slope·x + intercept`` + R² (no scipy needed for the fit itself)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def analyze(pairs: Sequence[dict]) -> dict:
    """The @640 additivity verdict over the probe pairs (pure; ≥3 pairs required)."""
    if len(pairs) < 3:
        raise ValueError(f"need >=3 probe pairs for a fit, got {len(pairs)}")
    for p in pairs:
        if not p.get("clocks_locked"):
            raise ValueError(f"{p['name']} was measured with unlocked clocks — not comparable")

    sums = [p["sum_ms"] for p in pairs]
    meas = [p["measured_ms"] for p in pairs]
    ratios = [m / s for m, s in zip(meas, sums, strict=True)]
    slope, intercept, r2 = _ols(sums, meas)

    from scipy.stats import pearsonr, spearmanr  # scipy is a pinned .venv dep

    rank = float(spearmanr(sums, meas).statistic)

    # DRAM-bound probes: relative error vs total depth and vs early-stage depth (d0+d1 —
    # the 160×160/80×80 blocks whose activations dominate memory traffic at 640).
    rel_err = [r - 1.0 for r in ratios]
    depth_sum = [float(sum(p["arch_d"])) for p in pairs]
    early_depth = [float(p["arch_d"][0] + p["arch_d"][1]) for p in pairs]

    def _corr(xs: list[float]) -> float | None:
        return None if len(set(xs)) < 2 else float(pearsonr(xs, rel_err).statistic)

    per_arch = sorted(
        ({**p, "ratio": r} for p, r in zip(pairs, ratios, strict=True)),
        key=lambda p: p["sum_ms"])
    return {
        "n": len(pairs),
        "per_arch": per_arch,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_mean": sum(ratios) / len(ratios),
        "fit": {"slope": slope, "intercept": intercept, "r2": r2,
                "note": "measured ~= slope*sum + intercept @640 fp32 — the honest successor "
                        "to the @224 fit (0.934x); apply for absolute prediction only, "
                        "never edit the @224 data/latency_calibration.json"},
        "spearman_measured_vs_sum": rank,
        "rel_err_vs_depth_sum_pearson": _corr(depth_sum),
        "rel_err_vs_early_depth_pearson": _corr(early_depth),
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory of bench rows + export sidecars")
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR / "additivity640_report.json")
    a = ap.parse_args(argv)

    pairs = [p for f in sorted(a.dir.glob("*_backbone_640.json"))
             if (p := pair_from_files(f)) is not None]
    report = analyze(pairs)
    a.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"@640 additivity probe — {report['n']} archs (fp32, mode 0):")
    for p in report["per_arch"]:
        print(f"  {p['name']:32s} d={p['arch_d']}  sum={p['sum_ms']:7.3f}  "
              f"measured={p['measured_ms']:7.3f}  ratio={p['ratio']:.3f}")
    fit = report["fit"]
    print(f"fit: measured = {fit['slope']:.3f}*sum + {fit['intercept']:+.3f}  (R2={fit['r2']:.4f})")
    print(f"ratio range [{report['ratio_min']:.3f}, {report['ratio_max']:.3f}]  "
          f"Spearman(measured, sum) = {report['spearman_measured_vs_sum']:.3f}")
    print(f"rel-err correlations: depth_sum {report['rel_err_vs_depth_sum_pearson']}, "
          f"early-depth(d0+d1) {report['rel_err_vs_early_depth_pearson']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
