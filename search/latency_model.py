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

# Physical memory-bound latency model (2026-07-13): for this family latency is DRAM-traffic-bound,
# so a single feature — total activation bytes — predicts fp32 e2e ms within ±3.6 % across the 6
# measured dense/search points (353–695 MB / 9.56–15.27 ms; R²≈0.99). Far more robust than the
# multi-feature ridge (which is collinear and over-predicts small nets), so THIS is the search
# fence. ms ≈ intercept + slope · act_mbytes. Refit via `fit_physical()`.
PHYSICAL_FP32_SLOPE = 0.017627      # ms per MB of activation traffic (6-point fit)
PHYSICAL_FP32_INTERCEPT = 2.994     # ms (fixed overhead: stem launch, IO, non-modeled)
PHYSICAL_FP32 = {"slope": PHYSICAL_FP32_SLOPE, "intercept": PHYSICAL_FP32_INTERCEPT,
                 "unit": "act_mbytes", "n": 6, "max_err_pct": 3.6}


def act_bytes_to_ms(act_mbytes: float,
                    slope: float = PHYSICAL_FP32_SLOPE,
                    intercept: float = PHYSICAL_FP32_INTERCEPT) -> float:
    """Physical fp32-latency estimate from activation traffic (the memory-bound fence)."""
    return intercept + slope * act_mbytes


# --- graft-family physical fit (winner-v2-OFA program, 2026-07-13) ---------------------------
# Same single-feature law, fit ONLY on the measured OFA-graft e2e points (winner-v1 variants +
# every pruned-graft descendant); the dense-family PHYSICAL_FP32 above does NOT transfer (the
# graft's depthwise op mix has a different bytes→ms slope). fp16 is fit DIRECTLY, not via the
# 0.700 ratio: measured family ratios span 0.69–0.73 and a pure ratio misses the intercept.
# Pinned from search/graft_latency_fit.json (regenerate: python -m search.latency_model
# --fit-graft); tests assert constants == tracked JSON.

# Pinned from search/graft_latency_fit.json (2026-07-13). fp32 includes the two Stage-0
# fallback topologies (different depth vectors, resid −2.6/−5.9 %) — the law generalizes
# across (e,d) graft topologies, which the min-act probe relies on. No fallback fp16 rows.
PHYSICAL_GRAFT_FP32 = {"slope": 0.031617914, "intercept": 0.787579279,
                       "unit": "act_mbytes", "n": 12, "loo_mape": 0.027325}
PHYSICAL_GRAFT_FP16 = {"slope": 0.020541899, "intercept": 1.199872874,
                       "unit": "act_mbytes", "n": 10, "loo_mape": 0.019797}

# Bench-row names of the graft e2e family. "fallback_" = the Stage-0 fallback graft topologies
# (different depth vectors) — they anchor the fit's topology generalization.
GRAFT_POINT_PREFIXES = ("graft_", "winner_v1", "fallback_")
GRAFT_EXCLUDE_SUBSTR = ("backbone",)             # partial nets (bare backbone) are not e2e


def is_graft_e2e_point(name: str) -> bool:
    """True for whole-net OFA-graft bench rows (partial nets like the bare backbone excluded)."""
    return name.startswith(GRAFT_POINT_PREFIXES) and not any(
        s in name for s in GRAFT_EXCLUDE_SUBSTR)


def fit_physical(points: list[dict], feature: str = "act_mbytes") -> dict:
    """Single-feature least-squares law ``ms = intercept + slope·feature`` + brute LOO.

    The physical counterpart of ``fit_ridge`` for memory-bound families (closes the
    ``fit_physical()`` promise in the PHYSICAL_FP32 comment). Points need ``ms`` + feature.
    """
    usable = [p for p in points if "ms" in p and feature in p]
    if len(usable) < 3:
        raise ValueError(f"need >=3 points with 'ms' and '{feature}' (got {len(usable)})")
    x = np.array([p[feature] for p in usable], dtype=float)
    y = np.array([p["ms"] for p in usable], dtype=float)

    def _line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        (s, b), *_ = np.linalg.lstsq(a, ys, rcond=None)
        return float(s), float(b)

    slope, intercept = _line(x, y)
    preds = intercept + slope * x
    resid_pct = 100.0 * (preds - y) / y
    loo_errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        s_i, b_i = _line(x[keep], y[keep])
        loo_errs.append(abs(b_i + s_i * x[i] - y[i]) / y[i])
    return {
        "feature": feature, "slope": slope, "intercept": intercept, "n": len(usable),
        "loo_mape": float(np.mean(loo_errs)),
        "max_err_pct": float(np.max(np.abs(resid_pct))),
        "points": [{"name": p["name"], "ms": float(ms), feature: float(xx),
                    "pred": round(float(pr), 3), "resid_pct": round(float(rp), 1)}
                   for p, xx, ms, pr, rp in zip(usable, x, y, preds, resid_pct, strict=True)],
    }


def act_limit_for_ms(target_ms: float, model: dict) -> float:
    """Invert the physical law: the max feature value whose prediction stays ≤ target_ms."""
    return (float(target_ms) - float(model["intercept"])) / float(model["slope"])


def _family_fit_report(e2e_dir: Any, root: Any, *, family: str, keep: Any) -> dict:
    """Per-precision physical fits over one measured e2e family (``keep`` filters by name)."""
    points = collect_points(e2e_dir, root=root)
    fam = [p for p in points if "ms" in p and keep(p["name"])]
    out: dict = {
        "family": family,
        "n_points": len(fam),
        "excluded": sorted({p["name"] for p in points
                            if "ms" in p and not keep(p["name"])}),
        "doctrine": "physical single-feature law (memory-bound: ms = b + a*act_mbytes); "
                    f"fit per precision on {family} points only — whole nets, never marginal; "
                    "claimed latencies are still verified on-device",
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fits": {},
    }
    for prec in sorted({p["precision"] for p in fam}):
        pts = [p for p in fam if p["precision"] == prec]
        out["fits"][prec] = (fit_physical(pts) if len(pts) >= 3
                             else {"skipped": f"only {len(pts)} points"})
    return out


def graft_fit_report(e2e_dir: Any, root: Any = None) -> dict:
    """Per-precision physical fits over the measured OFA-graft e2e family only."""
    return _family_fit_report(e2e_dir, root, family="ofa_graft_e2e", keep=is_graft_e2e_point)


# --- dense-native-family physical fits (NAS-born beat-n program, 2026-07-18) ------------------
# Same single-feature law over the dense op family (yolo11-shaped), but split into TWO
# subfamilies, because they are different act CURRENCIES: channel-pruned nets pass through
# prune/yolo_tp_prep.py's C2f chunk-split rewrite, which restructures the graph (97 → 105
# convs on the baseline) and deflates ONNX-counted activation bytes (the fat chunk/concat
# tensors vanish: baseline 549 MB → 197 MB at r20) without a proportional latency drop. A
# pooled fit mixes the currencies and degrades to LOO 5–7 % with ±12 % systematic residuals;
# split, both laws are tight:
#   pruned  (prune_base_*): fp32 LOO 0.6 % / fp16 1.2 %   ← the Arm-S fence (a pruned s39 IS
#           a prep-rewritten graph, priced by the same CPU probe pipeline)
#   scaled  (dense_*/densenas_*/baseline): fp32 LOO 3.4 % / fp16 3.9 %  (diagnostic; the
#           legacy PHYSICAL_FP32 above stays the pinned dense_nas search fence)
# fp16 is usable since the "±20 % build variance" caveat was retracted (contention,
# models/README.md 2026-07-17; audit_e2e runs 69/70 clean — the one suspect, yolo11s fp16,
# is outside this family). Extrapolation caveat: the pruned fit's support is 158–203 MB and
# Arm-S specs land ~230–265 MB — finalists gate on MEASURED ms, per standing doctrine.
# Pinned from search/dense_latency_fit.json (regenerate: python -m search.latency_model
# --fit-dense); tests assert constants == tracked JSON.
PHYSICAL_DENSE_PRUNED_FP32 = {"slope": 0.048426468, "intercept": -0.021571931,
                              "unit": "act_mbytes", "n": 7, "loo_mape": 0.006176}
PHYSICAL_DENSE_PRUNED_FP16 = {"slope": 0.023450210, "intercept": 1.317179853,
                              "unit": "act_mbytes", "n": 7, "loo_mape": 0.011771}

DENSE_POINT_PREFIXES = ("dense_", "densenas_", "prune_base_", "baseline_recheck")
DENSE_PRUNED_PREFIX = "prune_base_"


def is_dense_e2e_point(name: str) -> bool:
    """True for whole-net dense-family bench rows: width-scaled (dense_w*), searched
    (densenas_*), channel-pruned (prune_base_*) and the deployed baseline itself. Grafts,
    bare backbones and the yolo11s anchor (the audit's one suspect row) fall outside."""
    return name.startswith(DENSE_POINT_PREFIXES)


def is_dense_pruned_point(name: str) -> bool:
    """The prep-rewritten channel-pruned subfamily — its own act currency (see above)."""
    return name.startswith(DENSE_PRUNED_PREFIX)


def dense_fit_report(e2e_dir: Any, root: Any = None) -> dict:
    """Subfamily-split physical fits over the measured dense-family e2e points.

    ``subfamilies.pruned`` is the Arm-S fence; ``scaled`` and ``pooled`` are kept for the
    record (the pooled residual structure is the evidence for the split)."""
    subs = {
        "pruned": lambda n: is_dense_e2e_point(n) and is_dense_pruned_point(n),
        "scaled": lambda n: is_dense_e2e_point(n) and not is_dense_pruned_point(n),
        "pooled": is_dense_e2e_point,
    }
    out: dict = {
        "family": "dense_native_e2e",
        "split_rationale": "prep-rewritten pruned graphs deflate ONNX act (549→197 MB at "
                           "r20) — a different currency; pooled LOO 5–7 % vs split 0.6–3.9 %",
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subfamilies": {},
    }
    for name, keep in subs.items():
        rep = _family_fit_report(e2e_dir, root, family=f"dense_native_e2e/{name}", keep=keep)
        rep.pop("timestamp", None)
        out["subfamilies"][name] = rep
    return out

FEATURES = ("act_mbytes", "param_mbytes", "n_convs", "conv_gflops")

# bench-json name → ONNX basename, where they differ (bench names are session-scoped).
ALIASES = {
    "baseline_recheck_640": "yolo11n_pose_640",
    "graft_prune_r40_e2e_640": "recover_graft_r40_640",   # params-matched same graph (CP 6.2-G)
    "graft_prune_r60_e2e_640": "recover_graft_r60_640",
    # champion-session names (2026-07-12) → their deploy ONNX stems
    "graft_halp_10p4_640": "recover_graft_halp_fp32_10p4_640",
    "graft_halp_9p0_640": "recover_graft_halp_fp32_9p0_640",
    "graft_r50_gtay_640": "recover_graft_r50_gtay_640",
    "graft_r60_gtay_640": "recover_graft_r60_gtay_640",
    # Stage-3 dense-NAS finalists (2026-07-13 bench)
    "densenas_s39_640": "dense_s39-40-38-38-14_o100_640",
    "densenas_s31_640": "dense_s31-40-40-40-13_o100_640",
    "densenas_s40_640": "dense_s40-38-39-36-13_o100_640",
}

# Where deploy/bench ONNX live (searched in order; first hit wins).
ONNX_ROOTS = (
    "data/e2e",
    "models",
    "data/kaggle_out_asilarnous",
    "data/kaggle_out_asilarnous47",
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
                 "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    p.add_argument("--fit-graft", action="store_true",
                   help="physical single-feature fit on the OFA-graft e2e family only "
                        "(default out: the tracked search/graft_latency_fit.json)")
    p.add_argument("--fit-dense", action="store_true",
                   help="physical single-feature fit on the dense e2e family "
                        "(scaled/searched/pruned/baseline; default out: the tracked "
                        "search/dense_latency_fit.json)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    if a.fit_graft:
        report = graft_fit_report(a.e2e_dir)
        out = a.out or ROOT / "search" / "graft_latency_fit.json"
    elif a.fit_dense:
        report = dense_fit_report(a.e2e_dir)
        out = a.out or ROOT / "search" / "dense_latency_fit.json"
    else:
        report = fit_report(collect_points(a.e2e_dir), lam=a.lam)
        out = a.out or ROOT / "data" / "latency_model.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    sections = (report["subfamilies"].items() if "subfamilies" in report
                else [("", report)])
    for sub_name, sub in sections:
        for prec, fit in sub["fits"].items():
            label = f"{sub_name + ' ' if sub_name else ''}{prec}"
            if "skipped" in fit:
                print(f"{label}: {fit['skipped']}")
                continue
            head = f"{label}: n={fit['n']}  LOO-MAPE={fit['loo_mape'] * 100:.1f}%"
            if "slope" in fit:
                head += (f"  ms = {fit['intercept']:.3f} + {fit['slope']:.6f}·act_MB"
                         f"  (max |err| {fit['max_err_pct']:.1f}%)")
            print(head)
            for row in fit["points"]:
                print(f"  {row['name']:42s} {row['ms']:7.2f} → {row['pred']:7.2f}  "
                      f"({row['resid_pct']:+.1f}%)")
    if report.get("skipped"):
        print(f"skipped (no ONNX): {report['skipped']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
