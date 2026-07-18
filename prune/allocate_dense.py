"""Arm S (NAS-born beat-n program, 2026-07-18) — act-fenced allocation for pruning the
searched dense winner (s39) under the baseline's measured bars.

The dense mirror of ``prune/allocate_v2.py``, same doctrine: price every candidate by a
physical single-feature law applied to its ACTUAL activation bytes, in the right CURRENCY.
For this family the currency is the PRUNED one (``search/latency_model.py``
``PHYSICAL_DENSE_PRUNED_*``, LOO 0.6 % fp32 / 1.2 % fp16): every measured ``prune_base_*``
row went through ``prune/yolo_tp_prep.py``'s C2f chunk-split rewrite, which deflates
ONNX-counted act (baseline 549 → 197 MB at r20) — stock-graph act numbers do NOT transfer.
So each probe loads the trained donor, applies the prep rewrite, prunes to the candidate
spec with data-free l2 (channel COUNTS at fixed per-stage ratios are importance-invariant —
the shapes global_taylor/uniform would train are priced exactly), exports the deploy ONNX,
and reads ``act_mbytes``.

fp32 is the binding axis for this family (bar 12.74 → act ≤ 263 MB; the fp16 bar 7.75 →
274), so fences are ``--target-fp32-ms``; the payload records both predictions and the
fp16-bar cap is applied as a second ceiling. Extrapolation fence (recorded, not hidden):
the pruned law's support is 158–203 MB and specs land above it — so the emitted spec's
probe ONNX is kept under the spec's own name for a weight-independent Nano bench BEFORE
any recovery train buys in (the HALP lesson, applied proactively), and finalists still
gate on MEASURED ms.

Run (laptop ``.venv-nas``, CPU)::

    python -m prune.allocate_dense --target-fp32-ms 12.5 --target-fp32-ms 12.2 \
        --target-fp32-ms 11.6
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

from prune.allocate_v2 import (
    N_STAGES,
    REST_BASE,
    _fit_args,
    enumerate_specs,
    predict_act,
    rank_candidates,
)
from search.latency_model import (
    PHYSICAL_DENSE_PRUNED_FP16,
    PHYSICAL_DENSE_PRUNED_FP32,
    act_bytes_to_ms,
    act_limit_for_ms,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "prune" / "specs"
WORKDIR = ROOT / "data" / "allocate_dense"       # gitignored scratch (probe ONNX exports)

DEFAULT_DONOR = ROOT / "data" / "cp33_kaggle_out" / "dense_nas" / \
    "dense_s39-40-38-38-14_o100_best.pt"
FP16_BAR_MS = 7.75                                # deployed-baseline fp16 (models/README.md)
PROBE_RATIO = 0.4                                 # same Phase-A amplitude as allocate_v2


def honest_dense_spec_features(donor: Any, spec: dict | None, *, imgsz: int = 640,
                               workdir: Any = None, name: str = "probe") -> dict:
    """Load donor → prep rewrite → (spec-prune) → deploy ONNX → exact act bytes.

    ``spec=None`` prices the prep-rewritten UNPRUNED donor (the currency's base anchor).
    """
    import torch

    from prune.prune_baseline import (
        TRACE_IMGSZ,
        _export_deploy_onnx,
        dense_spec_ratio_dict,
        load_baseline_model,
    )
    from prune.prune_graft import prune_graft
    from prune.yolo_tp_prep import prepare_yolo_for_pruning_
    from search.latency_model import extract_onnx_features

    model = load_baseline_model(Path(donor))
    prep_ignored = prepare_yolo_for_pruning_(model)
    report = None
    if spec is not None:
        prd, spec_ignored = dense_spec_ratio_dict(model, spec)
        report = prune_graft(model, torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                             ratio=spec["rest_ratio"], pruning_ratio_dict=prd,
                             extra_ignored=prep_ignored + spec_ignored)
    wd = Path(workdir) if workdir is not None else WORKDIR
    wd.mkdir(parents=True, exist_ok=True)
    onnx_path = wd / f"{name}.onnx"
    _export_deploy_onnx(model, onnx_path, imgsz=imgsz)
    feats = extract_onnx_features(onnx_path)
    params = (report["params_after"] if report is not None
              else sum(p.numel() for p in model.parameters()))
    return {"act_mbytes": float(feats["act_mbytes"]), "coverage": float(feats["coverage"]),
            "params_after": int(params), "onnx": str(onnx_path)}


def fit_dense_sensitivities(donor: Any, *, imgsz: int = 640, workdir: Any = None) -> dict:
    """Phase A: 8 honest builds → the linear per-knob act model for grid screening."""
    def zspec(**over: Any) -> dict:
        s = {"stage_ratios": [0.0] * N_STAGES, "rest_ratio": REST_BASE}
        s.update(over)
        return s

    unpruned = honest_dense_spec_features(donor, None, imgsz=imgsz, workdir=workdir,
                                          name="sens_unpruned")
    base = honest_dense_spec_features(donor, zspec(), imgsz=imgsz, workdir=workdir,
                                      name="sens_base")
    stage_sens = []
    for s in range(N_STAGES):
        ratios = [0.0] * N_STAGES
        ratios[s] = PROBE_RATIO
        probe = honest_dense_spec_features(donor, zspec(stage_ratios=ratios), imgsz=imgsz,
                                           workdir=workdir, name=f"sens_s{s}")
        stage_sens.append((base["act_mbytes"] - probe["act_mbytes"]) / PROBE_RATIO)
    rest_hi = 0.5
    probe_r = honest_dense_spec_features(donor, zspec(rest_ratio=rest_hi), imgsz=imgsz,
                                         workdir=workdir, name="sens_rest")
    rest_sens = (base["act_mbytes"] - probe_r["act_mbytes"]) / (rest_hi - REST_BASE)
    return {"base_act": base["act_mbytes"], "stage": stage_sens, "rest": rest_sens,
            "unpruned_act": unpruned["act_mbytes"],
            "params_unpruned": unpruned["params_after"]}


def dense_spec_payload(stage_ratios: Any, rest_ratio: float, *, act_honest: float,
                       act_predicted: float, params_after: int, target_fp32_ms: float,
                       act_max: float, donor: Any, extra: dict | None = None) -> dict:
    """The emitted spec JSON — same 5 legacy keys the ladder row copies + provenance."""
    pred_fp32 = act_bytes_to_ms(act_honest, **_fit_args(PHYSICAL_DENSE_PRUNED_FP32))
    pred_fp16 = act_bytes_to_ms(act_honest, **_fit_args(PHYSICAL_DENSE_PRUNED_FP16))
    return {
        "stage_ratios": [float(r) for r in stage_ratios],
        "rest_ratio": float(rest_ratio),
        "predicted_fp32_ms": round(pred_fp32, 3),
        "fp16_estimate_ms": round(pred_fp16, 3),
        "target_fp32_ms": round(float(target_fp32_ms), 3),
        "act_mbytes_honest": round(act_honest, 2),
        "act_mbytes_predicted": round(act_predicted, 2),
        "params_after": int(params_after),
        "donor": str(donor),
        "family": "dense_pruned",
        "fence": {"fp32_target_ms": float(target_fp32_ms), "fp16_bar_ms": FP16_BAR_MS,
                  "act_max_mb": round(act_max, 2),
                  "fit": "search/dense_latency_fit.json subfamilies.pruned"},
        "objective": "max act ≤ fence; ties: min max-stage, min rest, min mean "
                     "(allocate_v2's measured rule)",
        "provenance": "allocate_dense honest fence: donor → yolo_tp_prep rewrite → l2 "
                      "spec-prune → deploy ONNX → act_mbytes → PRUNED-currency law (LOO "
                      "0.6 % fp32); law support 158–203 MB, spec extrapolates above it — "
                      "the probe ONNX is Nano-benched weight-free before any recovery run",
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(extra or {}),
    }


def search_dense_specs(donor: Any, target_fp32_ms: list[float], *, imgsz: int = 640,
                       workdir: Any = None, out_dir: Any = None, spec_prefix: str = "s39d",
                       verify_top: int = 12, max_verify: int = 36) -> list[Path]:
    """Phase A + grid screen + honest verification; one emitted spec per fp32 target."""
    out_dir = Path(out_dir) if out_dir is not None else SPEC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    wd = Path(workdir) if workdir is not None else WORKDIR

    sens = fit_dense_sensitivities(donor, imgsz=imgsz, workdir=workdir)
    print(f"[phase A] prep'd-unpruned act={sens['unpruned_act']:.1f} MB  "
          f"base_act={sens['base_act']:.1f}  stage_sens="
          f"{[round(c, 1) for c in sens['stage']]}  rest_sens={sens['rest']:.1f}",
          flush=True)

    grid = enumerate_specs()
    cands = [{"stage_ratios": sr, "rest_ratio": rr, "act": predict_act(sr, rr, sens)}
             for sr, rr in grid]

    written: list[Path] = []
    for target in target_fp32_ms:
        act_max = min(act_limit_for_ms(target, PHYSICAL_DENSE_PRUNED_FP32),
                      act_limit_for_ms(FP16_BAR_MS, PHYSICAL_DENSE_PRUNED_FP16))
        ranked = rank_candidates(cands, act_max)
        if not ranked:
            raise ValueError(f"target {target} ms (act ≤ {act_max:.0f} MB) infeasible "
                             f"even at max rungs")
        chosen = None
        verified: list[dict] = []
        for i, c in enumerate(ranked[:max_verify]):
            if i >= verify_top and chosen is not None:
                break
            spec = {"stage_ratios": list(c["stage_ratios"]), "rest_ratio": c["rest_ratio"]}
            h = honest_dense_spec_features(donor, spec, imgsz=imgsz, workdir=workdir,
                                           name=f"verify_t{str(target).replace('.', 'p')}_{i}")
            row = {**spec, "act_predicted": round(c["act"], 2),
                   "act_honest": round(h["act_mbytes"], 2), "params": h["params_after"],
                   "coverage": round(h["coverage"], 3),
                   "fits": h["act_mbytes"] <= act_max}
            verified.append(row)
            print(f"[verify {target}ms #{i}] stages={spec['stage_ratios']} "
                  f"rest={spec['rest_ratio']} act={h['act_mbytes']:.1f} "
                  f"(pred {c['act']:.1f}) {'OK' if row['fits'] else 'OVER'}", flush=True)
            if row["fits"] and chosen is None:
                chosen = (spec, c, h)
        if chosen is None:
            raise ValueError(f"target {target} ms: none of the top {max_verify} verified "
                             f"under {act_max:.0f} MB — widen rungs or raise the target")
        spec, c, h = chosen
        stem = f"{spec_prefix}_act{round(act_max)}"
        # Keep the chosen graph under the spec's own name — the weight-free Nano bench
        # candidate (deploy contract, same exporter as every measured prune_base_* row).
        probe_onnx = wd / f"{stem}_{imgsz}.onnx"
        shutil.copy2(h["onnx"], probe_onnx)
        payload = dense_spec_payload(spec["stage_ratios"], spec["rest_ratio"],
                                     act_honest=h["act_mbytes"], act_predicted=c["act"],
                                     params_after=h["params_after"],
                                     target_fp32_ms=target, act_max=act_max, donor=donor,
                                     extra={"probe_onnx": str(probe_onnx),
                                            "sensitivities": {k: sens[k] for k in
                                                              ("base_act", "stage", "rest")},
                                            "verified": verified})
        out = out_dir / f"{stem}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[emit] {out.name}: stages={payload['stage_ratios']} "
              f"rest={payload['rest_ratio']} act={payload['act_mbytes_honest']} MB "
              f"pred fp32={payload['predicted_fp32_ms']} fp16={payload['fp16_estimate_ms']} "
              f"params={payload['params_after']:,}", flush=True)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donor", type=Path, default=DEFAULT_DONOR,
                   help="trained dense donor .pt (default: the de-noised searched winner "
                        "s39-40-38-38-14 oracle checkpoint)")
    p.add_argument("--target-fp32-ms", type=float, action="append",
                   help="repeatable; fp32 binds for this family (bar 12.74). Defaults "
                        "12.5 / 12.2 / 11.6 — capacity-max, main, insurance")
    p.add_argument("--spec-prefix", type=str, default="s39d")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--verify-top", type=int, default=12)
    a = p.parse_args(argv)

    if not Path(a.donor).exists():
        raise SystemExit(f"donor not found: {a.donor}")
    targets = a.target_fp32_ms or [12.5, 12.2, 11.6]
    search_dense_specs(a.donor, targets, imgsz=a.imgsz, workdir=a.workdir,
                       out_dir=a.out_dir, spec_prefix=a.spec_prefix,
                       verify_top=a.verify_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
