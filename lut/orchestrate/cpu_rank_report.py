"""Does the Orin's architecture ranking survive on x86? -- the cross-device rank check report.

Joins the CPU rows written by ``cpu_bench`` against the measured Jetson fp32 rows in
``data/e2e/`` and answers two questions per thread config:

1. **Spearman/Kendall, CPU vs Jetson** -- does the ordering hold?
2. **The graft penalty curve** -- for each config, fit OLS of latency on params across the
   reference families (dense + prune + baseline, 15 models, 0.9-3.9M params) and report each
   graft's *residual*: the ms it costs beyond what its size predicts. A residual that GROWS
   with thread count means grafts lose more as bandwidth pressure rises (memory-bound
   generalises); one that is flat and positive everywhere points instead at a runtime/kernel
   effect, not bandwidth.

The anchor (yolo11s, 9.7M) is excluded from the fit: at 2.5x the next-largest reference model
it would dominate the slope as a leverage point. It still enters the Spearman.

**Caveat carried into every claim:** ORT != TensorRT. Graph optimisation and kernel selection
differ, so part of any rank delta is *runtime*, not *device*; this bench cannot separate them.

Run::

    python -m lut.orchestrate.cpu_rank_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from scipy.stats import kendalltau, spearmanr

from lut.orchestrate.cpu_bench import root_dir
from lut.orchestrate.cpu_pairs import GRAFT_FAMILIES, PAIRS, REFERENCE_FAMILIES


def count_params(onnx_path: Path) -> int:
    """Total initialiser elements -- the ONNX is the truth, not README's rounded '~2.4M'."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
    return total


def load_jetson(e2e_dir: Path) -> dict[str, float]:
    """name -> fp32 p50 ms. fp16 is skipped: README records +-20% TRT build variance."""
    out: dict[str, float] = {}
    for path in sorted(e2e_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "latency_ms" not in row:
            continue
        if row.get("precision") != "fp32":
            continue
        out[str(row["name"])] = float(row["latency_ms"]["p50"])
    return out


def load_cpu(cpu_dir: Path) -> dict[tuple[str, str], float]:
    """(name, config) -> p50 ms."""
    out: dict[tuple[str, str], float] = {}
    for path in sorted(cpu_dir.glob("*__*.json")):
        row = json.loads(path.read_text())
        out[(str(row["name"]), str(row["config"]))] = float(row["latency_ms"]["p50"])
    return out


def rank_stats(cpu: dict[str, float], jetson: dict[str, float]) -> dict[str, float]:
    """Spearman + Kendall tau-b over the names present in both."""
    names = sorted(set(cpu) & set(jetson))
    if len(names) < 3:
        raise ValueError(f"need at least 3 paired models, got {len(names)}")
    x = [cpu[n] for n in names]
    y = [jetson[n] for n in names]
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "n": len(names),
    }


def fit_reference(
    params: dict[str, float], lat: dict[str, float], families: dict[str, str]
) -> tuple[float, float]:
    """OLS latency ~ params over REFERENCE_FAMILIES only. Returns (slope, intercept)."""
    names = sorted(n for n in set(params) & set(lat) if families.get(n) in REFERENCE_FAMILIES)
    if len(names) < 2:
        raise ValueError(f"need at least 2 reference models to fit, got {len(names)}")
    slope, intercept = np.polyfit([params[n] for n in names], [lat[n] for n in names], 1)
    return float(slope), float(intercept)


def residuals(
    params: dict[str, float],
    lat: dict[str, float],
    slope: float,
    intercept: float,
    names: list[str],
) -> dict[str, float]:
    """ms above the reference line -- the graft penalty at matched params."""
    return {n: lat[n] - (slope * params[n] + intercept) for n in names if n in lat}


def build_report(
    cpu: dict[tuple[str, str], float],
    jetson: dict[str, float],
    params: dict[str, float],
    families: dict[str, str],
    configs: list[str],
) -> dict[str, object]:
    """The full report: rank stats + penalty curve + the scatter behind them, per config."""
    per_config: dict[str, object] = {}
    for cfg in configs:
        lat = {name: v for (name, c), v in cpu.items() if c == cfg}
        if len(lat) < 3:
            continue
        slope, intercept = fit_reference(params, lat, families)
        graft_names = sorted(n for n in lat if families.get(n) in GRAFT_FAMILIES)
        res = residuals(params, lat, slope, intercept, graft_names)
        per_config[cfg] = {
            "rank_vs_jetson": rank_stats(lat, jetson),
            "reference_fit": {"slope_ms_per_mparam": slope, "intercept_ms": intercept},
            "graft_residual_ms": res,
            "graft_residual_mean_ms": float(np.mean(list(res.values()))) if res else 0.0,
            "scatter": {
                n: {
                    "params_m": params.get(n, 0.0),
                    "latency_ms": lat[n],
                    "family": families.get(n, "?"),
                }
                for n in sorted(lat)
            },
        }
    return {
        "per_config": per_config,
        "clean_sweep_configs": [c for c in configs if c != "all22"],
        "caveats": [
            "all22 is confounded (E-cores are slower AND add pressure) -- excluded from the "
            "scaling claim.",
            "ORT != TensorRT: part of any rank delta is runtime, not device. This bench cannot "
            "separate them.",
            "The penalty metric assumes latency is ~linear in params; params ignores activation "
            "volume, which is what a memory-bound argument is about. Read the scatter.",
            "Single machine, single run: a rank inversion near the noise floor needs repeat "
            "rounds before being quoted.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = root_dir()
    ap = argparse.ArgumentParser(description="CPU vs Jetson rank-correlation report")
    ap.add_argument("--cpu-dir", type=Path, default=root / "data" / "cpu")
    ap.add_argument("--e2e-dir", type=Path, default=root / "data" / "e2e")
    ap.add_argument("--out", type=Path, default=root / "data" / "cpu" / "rank_report.json")
    args = ap.parse_args(argv)

    cpu = load_cpu(args.cpu_dir)
    if not cpu:
        print(f"error: no CPU rows in {args.cpu_dir} -- run cpu_bench first")
        return 2
    jetson = load_jetson(args.e2e_dir)
    families = {p.jetson_name: p.family for p in PAIRS}
    params = {
        p.jetson_name: count_params(root / p.onnx) / 1e6
        for p in PAIRS
        if (root / p.onnx).is_file()
    }
    configs = sorted({c for _, c in cpu}, key=lambda c: (c == "all22", c))

    report = build_report(cpu, jetson, params, families, configs)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    per_config: dict[str, dict[str, object]] = report["per_config"]  # type: ignore[assignment]
    print(f"{'config':>8} {'rho':>7} {'tau':>7} {'n':>4} {'graft penalty (ms)':>20}")
    print("-" * 52)
    for cfg, block in per_config.items():
        rk: dict[str, float] = block["rank_vs_jetson"]  # type: ignore[assignment]
        pen: float = block["graft_residual_mean_ms"]  # type: ignore[assignment]
        tag = "  (confounded)" if cfg == "all22" else ""
        print(
            f"{cfg:>8} {rk['spearman']:>7.3f} {rk['kendall']:>7.3f} {int(rk['n']):>4} "
            f"{pen:>20.2f}{tag}"
        )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
