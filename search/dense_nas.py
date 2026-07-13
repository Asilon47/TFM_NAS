"""Stage-3 — hardware-aware NAS over the DEVICE-NATIVE (yolo11-pose) family (user, 2026-07-12).

Every measured verdict of the cross-family program points here: the pretrained-supernet
families lose on this board for primitive-level reasons (MBv3 memory-bound, R50 compute-bound),
the device-native dense family owns the frontier, and the gate task un-binds the original
no-scratch-training constraint (~1.5 GPU-h trains any candidate from scratch). So stage 3
searches the winning family DIRECTLY: **per-stage width multipliers** of the yolo11-pose graph
(AutoSlim's lesson — layer-wise widths beat any global multiplier; the CP 3c waves and the
DepGraph prune ladder are both coarse 1-knob slices of this space, and prune_base r20 = the
point to beat at 0.8381 / measured 9.52 fp32).

Signals: accuracy = short from-scratch proxy (stock recipe, PROXY_EPOCHS) **gated by G3**
(Spearman ρ ≥ 0.70 + top-1-regret vs the ten 100-ep dense oracles, CP 2.4 discipline; the
30-ep re-trains ride kernels v28/v23); latency = the measured-anchored whole-net surrogate
(``search/latency_model.py``, ranking-only) as a HARD pre-train ceiling, with on-device
verification of finalists only (the HALP misprediction is the standing warning). Proposer =
Optuna TPE (CP 3.4: TPE ≈ BO here), one independent seed per GPU, per-candidate row files →
resumable + coordination-free striping, the repo's standard pattern.

Run (GPU)::

    python -m search.dense_nas --budget 20 --seed 0 --device 0 --out-dir data/dense_nas
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Stage of every channel-bearing layer of the stock yolo11-pose yaml (backbone B0..B10,
# head H0..H12 — indices into the yaml lists). Stages: 1=stride-2 stem, 2=stride-4,
# 3=P3(stride-8), 4=P4(stride-16), 5=P5(stride-32). Head C3k2/Conv entries take the stage of
# the pyramid level they emit. Non-channel layers (Upsample/Concat/Pose) are absent.
BACKBONE_STAGE = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5, 9: 5, 10: 5}
HEAD_STAGE = {2: 4, 5: 3, 6: 3, 8: 4, 9: 4, 11: 5}

# Search box: per-stage scale on the yaml's NOMINAL channels (yolo11n == 0.25 everywhere; the
# measured practical floor is ~0.13 global — C2PSA head-count starvation — so stage 5 gets a
# raised floor and every candidate is build-checked before training).
SCALE_LO, SCALE_HI = 0.10, 0.40
STAGE5_LO = 0.13
N_STAGES = 5

# Wave-2 constrained box (the recalibrated re-search, 2026-07-13). Physical cost ∝ Σ channels ×
# spatial²: the STEM/early stages dominate (stage1 64×320², stage2 256×160² ≫ P3 512×80² > P4 >
# P5) — NOT the feature stages. Wave-1's finalists were expensive because they ran wide EARLY
# stages (s1≈s2≈0.40). So cap the high-spatial early stages hard (they're cheap on accuracy —
# low-level features), keep the accuracy-bearing feature stages P3/P4 open, gut the P5 tail. The
# physical act_mbytes fence (--ceiling-fp32-ms 12.0 → act ≤ ~512 MB) does the real gating.
STAGE_HI_FEASIBLE = [0.20, 0.22, 0.40, 0.40, 0.25]
DEPTH_MULT = 0.50          # depth is a dead knob below n (CP 3c.1) — pinned at yolo11n's own
MAX_CHANNELS = 4096        # never cap: per-stage scales own the widths
DIVISOR = 16               # tensor-core alignment, same knob as the prune ladder's round_to


def _div16(x: float) -> int:
    return max(DIVISOR, int(round(x / DIVISOR)) * DIVISOR)


def stagewise_yaml(base: dict, scales: list[float]) -> dict:
    """The stock yolo11-pose yaml with per-stage ABSOLUTE channels (scales × nominal, /16).

    ``scales`` has N_STAGES entries. The yaml's ``scales:`` block pins width_mult=1.0 so
    Ultralytics uses our absolute numbers; depth stays DEPTH_MULT.
    """
    if len(scales) != N_STAGES:
        raise ValueError(f"need {N_STAGES} stage scales, got {len(scales)}")
    for i, s in enumerate(scales):
        lo = STAGE5_LO if i == N_STAGES - 1 else SCALE_LO
        if not lo <= s <= SCALE_HI:
            raise ValueError(f"stage {i + 1} scale {s} outside [{lo}, {SCALE_HI}]")
    out = {k: v for k, v in base.items() if k not in ("backbone", "head", "scales")}
    out["scales"] = {"n": [DEPTH_MULT, 1.0, MAX_CHANNELS]}

    def _scale_section(section: list, stage_of: dict) -> list:
        new = []
        for i, entry in enumerate(section):
            entry = [entry[0], entry[1], entry[2], list(entry[3])]
            if i in stage_of:
                nominal = entry[3][0]
                if not isinstance(nominal, int):
                    raise ValueError(f"layer {i}: expected int channels, got {nominal!r}")
                entry[3][0] = _div16(nominal * scales[stage_of[i] - 1])
            new.append(entry)
        return new

    out["backbone"] = _scale_section(base["backbone"], BACKBONE_STAGE)
    out["head"] = _scale_section(base["head"], HEAD_STAGE)
    return out


def candidate_tag(scales: list[float]) -> str:
    """Stable tag, e.g. s25-25-25-25-25 == the yolo11n point."""
    return "s" + "-".join(f"{int(round(s * 100)):02d}" for s in scales)


def build_check(model_yaml_dict: dict, out_dir: Path, tag: str) -> dict | None:
    """CPU construct + ONNX export + surrogate fp32 prediction; None if the build fails
    (e.g. C2PSA starvation) — the candidate is rejected before any GPU time."""
    import torch
    from ultralytics import YOLO

    from search.latency_model import extract_onnx_features

    path = Path(out_dir) / f"probe_{tag}.yaml"
    path.write_text(yaml.safe_dump(model_yaml_dict, sort_keys=False))
    try:
        m = YOLO(str(path))
        params = int(sum(p.numel() for p in m.model.parameters()))
        onnx = Path(m.export(format="onnx", imgsz=640, opset=17, device="cpu"))
    except Exception as e:  # starved attention / bad graph → reject, don't crash the study
        print(f"[reject] {tag}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # Fence on the PHYSICAL memory-bound model (act_mbytes → ms, ±3.6 %), not the collinear
    # ridge (which over-predicts small nets — the wave-1 miscalibration). See latency_model.
    from search.latency_model import act_bytes_to_ms

    feats = extract_onnx_features(onnx)
    pred = act_bytes_to_ms(feats["act_mbytes"])
    onnx.unlink(missing_ok=True)  # probe only; the trainer exports the real deploy ONNX
    return {"params": params, "pred_fp32_ms": round(pred, 3), **{k: round(float(v), 4)
            for k, v in feats.items()}}


def run_tpe(
    *,
    budget: int,
    seed: int,
    out_dir: Path,
    data_yaml: Path,
    proxy_epochs: int = 30,
    ceiling_fp32_ms: float = 14.0,
    imgsz: int = 640,
    batch: int = 16,
    device: Any = 0,
    stage_hi: list[float] | None = None,
) -> list[dict]:
    """One independent TPE study: propose scales → build-check + ceiling → proxy-train → row.

    ``stage_hi`` (len N_STAGES) overrides the per-stage upper bounds — pass STAGE_HI_FEASIBLE
    for the wave-2 constrained re-search; None keeps the uniform SCALE_HI box.
    """
    import optuna

    from search.dense_family import _base_pose_yaml, train_from_yaml

    hi = [SCALE_HI] * N_STAGES if stage_hi is None else list(stage_hi)
    if len(hi) != N_STAGES:
        raise ValueError(f"stage_hi needs {N_STAGES} entries, got {len(hi)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _base_pose_yaml()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: Any) -> float:
        scales = [trial.suggest_float(f"s{i + 1}",
                                      STAGE5_LO if i == N_STAGES - 1 else SCALE_LO, hi[i])
                  for i in range(N_STAGES)]
        tag = candidate_tag(scales) + f"_p{proxy_epochs}"
        row_file = out_dir / f"dense_{tag}.row.json"
        if row_file.exists():  # resumability + duplicate proposals
            row = json.loads(row_file.read_text())
            return float(row["map"])
        spec = stagewise_yaml(base, scales)
        probe = build_check(spec, out_dir, tag)
        if probe is None:
            return -1.0
        if probe["pred_fp32_ms"] > ceiling_fp32_ms:
            print(f"[ceiling] {tag}: pred {probe['pred_fp32_ms']} > {ceiling_fp32_ms}",
                  flush=True)
            return -0.5
        row = train_from_yaml(tag, spec, data_yaml=data_yaml, out_dir=out_dir,
                              epochs=proxy_epochs, imgsz=imgsz, batch=batch,
                              device=device, seed=seed)
        row.update({"scales": scales, "study_seed": seed, **probe})
        row_file.write_text(json.dumps(row, indent=2) + "\n")
        print(f"[dense-nas] {tag} map={row['map']:.4f} pred={probe['pred_fp32_ms']}ms "
              f"params={row['params']:,}", flush=True)
        return float(row["map"])

    study.optimize(objective, n_trials=budget)
    rows = [json.loads(f.read_text()) for f in sorted(out_dir.glob("dense_s*.row.json"))]
    (out_dir / f"dense_nas_seed{seed}.json").write_text(json.dumps(
        {"seed": seed, "budget": budget, "ceiling_fp32_ms": ceiling_fp32_ms,
         "proxy_epochs": proxy_epochs, "n_rows": len(rows),
         "best": (max(rows, key=lambda r: r["map"]) if rows else None)},
        indent=2) + "\n")
    return rows


def scales_from_tag(tag: str) -> list[float]:
    """Invert candidate_tag: 's31-40-40-40-13' → [0.31, 0.40, 0.40, 0.40, 0.13]."""
    body = tag[1:].split("_")[0]
    parts = body.split("-")
    if len(parts) != N_STAGES:
        raise ValueError(f"tag {tag!r} does not carry {N_STAGES} stage scales")
    return [int(x) / 100 for x in parts]


def train_oracles(tags: list[str], *, epochs: int, out_dir: Path, data_yaml: Path,
                  imgsz: int = 640, batch: int = 16, device: Any = 0, seed: int = 0) -> None:
    """Finalist round: re-train tag-encoded candidates to capacity (the 100-ep oracle)."""
    from search.dense_family import _base_pose_yaml, train_from_yaml

    base = _base_pose_yaml()
    for tag in tags:
        spec = stagewise_yaml(base, scales_from_tag(tag))
        otag = f"{tag.split('_')[0]}_o{epochs}" + (f"_s{seed}" if seed else "")
        if (Path(out_dir) / f"dense_{otag}.row.json").exists():
            print(f"[skip] {otag}", flush=True)
            continue
        row = train_from_yaml(otag, spec, data_yaml=data_yaml, out_dir=out_dir,
                              epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                              seed=seed)
        print(f"[oracle] {otag} map={row['map']:.4f} params={row['params']:,}", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--oracle-tags", type=str, default=None,
                   help="comma list of sNN-NN-NN-NN-NN tags → finalist re-train instead of TPE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--proxy-epochs", type=int, default=30)
    p.add_argument("--ceiling-fp32-ms", type=float, default=14.0,
                   help="surrogate fence (finalists gate on MEASURED ms); wave-2 uses 12.0")
    p.add_argument("--stage-hi", type=str, default=None,
                   help="comma list of N_STAGES per-stage upper bounds (wave-2 constrained box; "
                        "'feasible' → STAGE_HI_FEASIBLE)")
    p.add_argument("--data", type=Path, default=ROOT / "dataset" / "dataset.yaml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "dense_nas")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    a = p.parse_args(argv)

    if a.oracle_tags:
        train_oracles([t for t in a.oracle_tags.split(",") if t], epochs=a.proxy_epochs,
                      out_dir=a.out_dir, data_yaml=a.data, imgsz=a.imgsz, batch=a.batch,
                      device=a.device, seed=a.seed)
        return 0

    stage_hi = None
    if a.stage_hi == "feasible":
        stage_hi = STAGE_HI_FEASIBLE
    elif a.stage_hi:
        stage_hi = [float(x) for x in a.stage_hi.split(",")]
    rows = run_tpe(budget=a.budget, seed=a.seed, out_dir=a.out_dir, data_yaml=a.data,
                   proxy_epochs=a.proxy_epochs, ceiling_fp32_ms=a.ceiling_fp32_ms,
                   imgsz=a.imgsz, batch=a.batch, device=a.device, stage_hi=stage_hi)
    for r in sorted(rows, key=lambda r: -r["map"])[:5]:
        print(f"  {r['tag']:28s} map={r['map']:.4f} pred={r.get('pred_fp32_ms')}ms "
              f"params={r['params']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
