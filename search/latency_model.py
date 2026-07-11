"""Track 2 (pruning-as-search program) — e2e latency surrogate fitted on measured Nano points.

Stage 0 killed the @224 LUT-sum additivity at 640 (measured backbone = 1.236x the sum, and the
calibration *inverts*), so off-grid widths (every pruned net) have had no latency oracle at all
— "measured-only". This module fits the cheap replacement: a per-precision ridge regression
``ms ≈ f(activation bytes, param bytes, n_convs, conv GFLOPs)`` over every measured e2e bench in
``data/e2e/`` (mode 0, clocks locked — ~30 points across baseline / dense / pruned / graft /
backbone families), with features read straight from each point's deploy ONNX (static @640
shapes → onnx shape inference). The two leading features are exactly the roofline axes the
cross-family campaign measured: activation traffic (memory-bound, the graft's wall) and FLOPs
(compute-bound, R50's wall).

Doctrine (unchanged from the Stage-0 lesson): the surrogate is **ranking-only, whole-net-only**
— it prices complete candidates in the gated width-aware re-search (Track 5) and pre-screens
ceilings; any *claimed* latency is verified on-device. It must NEVER drive marginal/allocation
decisions: the features are collinear across families (param bytes ≈ conv FLOPs), so the ridge
splits them unphysically (conv_gflops lands NEGATIVE at every lambda; a sign-constrained NNLS
fit zeroes two features and degrades LOO 8.7→15.4 %). HALP-lite allocation (CP 6.4) uses the
per-block @640 LUT rows instead. fp32 is the reliable axis (LOO-MAPE 8.7 % on 33 points); fp16
TRT builds carry ±~20 % autotuner variance (models/README.md) → fp16 LOO 11.8 % on 21.

Run (laptop ``.venv``)::

    python -m search.latency_model --out data/latency_model.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ("act_mbytes", "param_mbytes", "n_convs", "conv_gflops")

# bench-json name → ONNX basename, where they differ (bench names are session-scoped).
ALIASES = {
    "baseline_recheck_640": "yolo11n_pose_640",
    "graft_prune_r40_e2e_640": "recover_graft_r40_640",   # params-matched same graph (CP 6.2-G)
    "graft_prune_r60_e2e_640": "recover_graft_r60_640",
}

# Where deploy/bench ONNX live (searched in order; first hit wins).
ONNX_ROOTS = (
    "data/e2e",
    "models",
    "data/kaggle_out_asilarnous/prune_baseline",
    "data/kaggle_out_asilarnous47/dense_scaling",
    "data/cp33_kaggle_out",
)


def _dtype_bytes(elem_type: int) -> int:
    import onnx

    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
        return int(np.dtype(np_dtype).itemsize)
    except Exception:
        return 4


def extract_onnx_features(path: Any) -> dict:
    """Static-graph features of one deploy ONNX (batch-1, fixed imgsz exports).

    ``act_mbytes`` sums every shape-resolved intermediate/output tensor (nominal element size
    — precision is handled by fitting per precision); ``coverage`` reports the fraction of
    value_infos that were fully static (guard against symbolic dims silently zeroing traffic).
    """
    import onnx

    model = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph

    param_bytes = 0
    init_names = set()
    for init in graph.initializer:
        init_names.add(init.name)
        n = 1
        for d in init.dims:
            n *= int(d)
        param_bytes += n * _dtype_bytes(init.data_type)

    act_bytes = 0
    resolved = 0
    total = 0
    for vi in list(graph.value_info) + list(graph.output):
        if vi.name in init_names:
            continue
        total += 1
        shape = vi.type.tensor_type.shape.dim
        dims = [int(d.dim_value) for d in shape if d.HasField("dim_value") and d.dim_value > 0]
        if not shape or len(dims) != len(shape):
            continue
        resolved += 1
        n = 1
        for d in dims:
            n *= d
        act_bytes += n * _dtype_bytes(vi.type.tensor_type.elem_type)

    # conv GFLOPs need output shapes + weight shapes
    vi_shape = {}
    for vi in list(graph.value_info) + list(graph.output):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            vi_shape[vi.name] = [int(d.dim_value) for d in dims]
    init_dims = {i.name: list(i.dims) for i in graph.initializer}

    n_convs = 0
    flops = 0.0
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        n_convs += 1
        w = init_dims.get(node.input[1]) if len(node.input) > 1 else None
        out = vi_shape.get(node.output[0])
        if not w or not out:
            continue
        out_numel = 1
        for d in out:
            out_numel *= d
        # weight [out_c, in_c/groups, kh, kw] → MACs = out_numel * in_c/groups * kh * kw
        macs = out_numel * int(w[1]) * int(w[2]) * int(w[3])
        flops += 2.0 * macs

    return {
        "act_mbytes": act_bytes / 2**20,
        "param_mbytes": param_bytes / 2**20,
        "n_convs": n_convs,
        "conv_gflops": flops / 1e9,
        "coverage": (resolved / total) if total else 0.0,
    }


def resolve_onnx(name: str, root: Any = None) -> Path | None:
    """The ONNX behind a bench-json ``name`` (strip the _fp16 suffix, apply ALIASES, search)."""
    root = ROOT if root is None else Path(root)
    stem = name[: -len("_fp16")] if name.endswith("_fp16") else name
    stem = ALIASES.get(stem, stem)
    for r in ONNX_ROOTS:
        hits = sorted((root / r).rglob(f"{stem}.onnx")) if (root / r).exists() else []
        if hits:
            return hits[0]
    return None


def collect_points(e2e_dir: Any, root: Any = None) -> list[dict]:
    """Every locked-clock mode-0 bench json paired with its ONNX features."""
    rows: list[dict] = []
    for f in sorted(Path(e2e_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "latency_ms" not in d or "name" not in d:
            continue
        if not d.get("clocks_locked") or str(d.get("power_mode")) != "0":
            continue
        onnx_path = resolve_onnx(d["name"], root=root)
        if onnx_path is None:
            rows.append({"name": d["name"], "skipped": "no ONNX resolved"})
            continue
        feats = extract_onnx_features(onnx_path)
        rows.append({"name": d["name"], "precision": d.get("precision", "fp32"),
                     "ms": float(d["latency_ms"]["mean"]), "onnx": str(onnx_path), **feats})
    return rows


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> dict:
    """Standardized ridge with intercept (closed form); coefficients in ORIGINAL units."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    xs = (x - mu) / sd
    a = xs.T @ xs + lam * np.eye(xs.shape[1])
    b = xs.T @ (y - y.mean())
    w_std = np.linalg.solve(a, b)
    w = w_std / sd
    intercept = float(y.mean() - (w * mu).sum())
    return {"coef": w.tolist(), "intercept": intercept, "lam": lam,
            "features": list(FEATURES)}


def predict_ms(model: dict, feats: dict) -> float:
    w = np.asarray(model["coef"], dtype=float)
    x = np.asarray([feats[k] for k in model["features"]], dtype=float)
    return float(model["intercept"] + (w * x).sum())


def loo_mape(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> float:
    """Leave-one-out MAPE by brute refit (n is ~a dozen per precision)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        m = fit_ridge(x[keep], y[keep], lam=lam)
        pred = m["intercept"] + (np.asarray(m["coef"]) * x[i]).sum()
        errs.append(abs(pred - y[i]) / y[i])
    return float(np.mean(errs))


def fit_report(points: list[dict], lam: float = 1e-2) -> dict:
    """Per-precision fits + LOO-MAPE + per-point residuals over the usable points."""
    usable = [p for p in points if "ms" in p]
    out: dict = {"n_points": len(usable),
                 "skipped": [p["name"] for p in points if "skipped" in p],
                 "doctrine": "ranking-only, WHOLE-NET-only; never marginal/allocation use "
                             "(collinear features -> unphysical per-feature signs); any "
                             "claimed latency is verified on-device (Stage-0 lesson: the "
                             "@224 additivity inverted at 640)",
                 "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "fits": {}}
    for prec in sorted({p["precision"] for p in usable}):
        pts = [p for p in usable if p["precision"] == prec]
        if len(pts) < len(FEATURES) + 2:
            out["fits"][prec] = {"skipped": f"only {len(pts)} points"}
            continue
        x = np.array([[p[k] for k in FEATURES] for p in pts])
        y = np.array([p["ms"] for p in pts])
        model = fit_ridge(x, y, lam=lam)
        preds = [predict_ms(model, p) for p in pts]
        out["fits"][prec] = {
            **model,
            "n": len(pts),
            "loo_mape": loo_mape(x, y, lam=lam),
            "points": [{"name": p["name"], "ms": p["ms"], "pred": round(pr, 3),
                        "resid_pct": round(100 * (pr - p["ms"]) / p["ms"], 1)}
                       for p, pr in zip(pts, preds, strict=True)],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--e2e-dir", type=Path, default=ROOT / "data" / "e2e")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "latency_model.json")
    a = p.parse_args(argv)

    points = collect_points(a.e2e_dir)
    report = fit_report(points, lam=a.lam)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for prec, fit in report["fits"].items():
        if "skipped" in fit:
            print(f"{prec}: {fit['skipped']}")
            continue
        print(f"{prec}: n={fit['n']}  LOO-MAPE={fit['loo_mape'] * 100:.1f}%")
        for row in fit["points"]:
            print(f"  {row['name']:42s} {row['ms']:7.2f} → {row['pred']:7.2f}  "
                  f"({row['resid_pct']:+.1f}%)")
    if report["skipped"]:
        print(f"skipped (no ONNX): {report['skipped']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
