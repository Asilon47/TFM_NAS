"""Phase 3c wave 1 — dense-family scaling search: yolo11-pose (depth, width, max_ch) variants.

Plan amendment 2026-07-07 ("dense-family arm", user decision A+B1+B2). Stage 0 proved the
OFA-MBv3 family is device-dominated on the Nano (depthwise ≈ 0.30 TFLOP/s effective vs the
baseline's dense 0.60); the literature's answer is to search the device-native family
(YOLO-NAS/QA-RepVGG, DAMO-YOLO/MAE-NAS — dense primitives under TensorRT constraints). The
gate task un-binds the no-scratch-training constraint (2,842 imgs ⇒ ~1–2 GPU-h per 100-epoch
train), so the yolo11-pose scaling knobs themselves become searchable: each candidate is the
stock yolo11-pose graph at a custom ``scales:`` triple, trained from scratch with the stock
Ultralytics recipe (recipe-uniform ⇒ internally fair), exported to the SAME deploy-ONNX
contract as the baseline (`yolo11n_pose_640.onnx`) for measured-only Nano latencies.

Wave-1 grid brackets yolo11n from below (the saturation says the interesting direction is
DOWN); ``ctrl_n`` re-trains yolo11n's own scale from scratch — the recipe control that
separates "smaller arch" from "no COCO pretrain" in every comparison against the 0.877 anchor.
Single-seed wave: winner's-curse discipline (CP 3.5) applies before any pick.

Run (GPU)::

    python -m search.dense_family --data dataset/dataset.yaml --epochs 100 \
        --device 0 --out-dir data/dense_scaling            # all wave-1 candidates
    python -m search.dense_family --report-only --out-dir data/dense_scaling
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# (tag, depth_mult, width_mult, max_channels) — yolo11n-pose is (0.50, 0.25, 1024).
WAVE1: list[tuple[str, float, float, int]] = [
    ("ctrl_n", 0.50, 0.25, 1024),   # yolo11n scale, from scratch — the recipe control
    ("d33_w25", 0.33, 0.25, 1024),
    ("d50_w20", 0.50, 0.20, 1024),
    ("d33_w20", 0.33, 0.20, 1024),
    ("d25_w25", 0.25, 0.25, 1024),
    ("d50_w15", 0.50, 0.15, 1024),
]

# WAVE2 (2026-07-08): a finer WIDTH sweep of the dense frontier — CP 3c.1 proved depth is a
# dead knob below n (the C3k2 repeat floor), so every candidate here fixes depth at yolo11n's
# own 0.50 and varies only width. Fills below (w0.13), between (w0.18/0.22) and above (w0.30)
# the wave-1 points {0.15, 0.20, 0.25} → a 7-point width/accuracy/latency curve. w0.10 is
# INFEASIBLE (verified): the stock C2PSA attention needs dim//64 >= 1 heads, and below ~w0.13
# the deepest block starves it → ZeroDivisionError. 0.13 is the practical width floor.
WAVE2: list[tuple[str, float, float, int]] = [
    ("w13", 0.50, 0.13, 1024),
    ("w18", 0.50, 0.18, 1024),
    ("w22", 0.50, 0.22, 1024),
    ("w30", 0.50, 0.30, 1024),
]

WAVES: dict[str, list[tuple[str, float, float, int]]] = {"1": WAVE1, "2": WAVE2}

# Cross-family reference points for the wave report (the CP 3c.3 figure's y-axis anchors).
# baseline = deployed yolo11n-pose (COCO-pretrained + Ultralytics recipe); yolo11s is anchor B
# (CP 3.4). ctrl_n (in-wave) is the from-scratch/stock-recipe control for the SAME arch as the
# baseline — its gap to `baseline` isolates the COCO-pretrain + recipe advantage.
DENSE_ANCHORS = {"yolo11n_pretrained": 0.877, "yolo11s_pretrained": 0.882}

WAVE_CAVEAT = (
    "single-seed, from-scratch (no COCO pretrain), stock Ultralytics recipe for every "
    "candidate incl. the ctrl_n control; latencies are measured-only (Nano e2e bench of the "
    "exported deploy ONNX) and pending until that session; de-noise the top band at fresh "
    "seeds before any pick (CP 3.5 winner's-curse discipline)."
)


def wave_tags(wave: list[tuple[str, float, float, int]] | None = None) -> list[str]:
    """The candidate tags (validated unique) — the split unit for multi-GPU striping."""
    wave = WAVE1 if wave is None else wave
    tags = [t for t, *_ in wave]
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate wave tags")
    return tags


def scaled_yaml(base: dict, depth_mult: float, width_mult: float, max_channels: int) -> dict:
    """The stock yolo11-pose model dict pinned to ONE custom scale triple (key 'n').

    The file is later named ``yolo11n-pose_<tag>.yaml`` so Ultralytics' scale-from-filename
    guess resolves to 'n' — the only key we provide — making the triple unambiguous.
    """
    if not 0.0 < depth_mult <= 1.25 or not 0.0 < width_mult <= 1.25:
        raise ValueError(f"suspicious scale ({depth_mult}, {width_mult}) — wave-1 is sub-n")
    if not 64 <= max_channels <= 2048:
        raise ValueError(f"max_channels {max_channels} out of range")
    out = dict(base)
    out["scales"] = {"n": [depth_mult, width_mult, max_channels]}
    return out


def fixed_data_yaml_dict(data: dict, root: Path) -> dict:
    """The gate data dict with ``path:`` re-rooted (the shipped yaml carries a stale absolute
    path from the machine that created it — see dataset/dataset.yaml)."""
    out = dict(data)
    out["path"] = str(root)
    return out


def assemble_wave_report(rows: list[dict], *, anchors: dict | None = None) -> dict:
    """Wave report: rows sorted by params, control flagged, standing caveats attached."""
    out_rows = sorted(rows, key=lambda r: r["params"])
    return {
        "rows": out_rows,
        "n": len(out_rows),
        "control_tag": "ctrl_n",
        "anchors": anchors or {},
        "note": WAVE_CAVEAT,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _base_pose_yaml() -> dict:
    """The stock yolo11-pose model yaml shipped inside the installed ultralytics package."""
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    return yaml.safe_load((pkg / "cfg" / "models" / "11" / "yolo11-pose.yaml").read_text())


def _prepare_data_yaml(data_yaml: Path, workdir: Path) -> Path:
    src = Path(data_yaml).resolve()
    fixed = fixed_data_yaml_dict(yaml.safe_load(src.read_text()), src.parent)
    out = workdir / "gate_data.yaml"
    out.write_text(yaml.safe_dump(fixed, sort_keys=False))
    return out


def train_candidate(
    tag: str,
    depth_mult: float,
    width_mult: float,
    max_channels: int,
    *,
    data_yaml: Path,
    out_dir: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Any = 0,
    seed: int = 0,
) -> dict:
    """Train + val + export one scaling candidate (stock model, stock recipe); return its row."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = out_dir / f"yolo11n-pose_{tag}.yaml"
    model_yaml.write_text(yaml.safe_dump(
        scaled_yaml(_base_pose_yaml(), depth_mult, width_mult, max_channels), sort_keys=False))
    data = _prepare_data_yaml(data_yaml, out_dir)

    model = YOLO(str(model_yaml))
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                seed=seed, pretrained=False, project=str(out_dir / "runs"), name=tag,
                exist_ok=True, plots=False, workers=2)
    metrics = model.val(data=str(data), imgsz=imgsz, device=device)

    weights = out_dir / f"dense_{tag}_best.pt"
    best = out_dir / "runs" / tag / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, weights)
    onnx_src = model.export(format="onnx", imgsz=imgsz, opset=17, device="cpu")
    onnx = out_dir / f"dense_{tag}_640.onnx"
    shutil.copy(onnx_src, onnx)

    row = {
        "tag": tag, "depth_mult": depth_mult, "width_mult": width_mult,
        "max_channels": max_channels,
        "params": int(sum(p.numel() for p in model.model.parameters())),
        "map": float(metrics.pose.map), "map50": float(metrics.pose.map50),
        "box_map": float(metrics.box.map),
        "epochs": epochs, "seed": seed, "pretrained": False,
        "weights": str(weights), "onnx": str(onnx),
    }
    (out_dir / f"dense_{tag}.row.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def run_wave(
    *,
    data_yaml: Path,
    out_dir: Path,
    only: list[str] | None = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Any = 0,
    wave: list[tuple[str, float, float, int]] | None = None,
) -> list[dict]:
    """Train the wave (or the ``only`` subset); per-tag row files make the loop resumable and
    let two GPU workers stripe the wave without coordination."""
    rows: list[dict] = []
    for tag, d, w, mc in (WAVE1 if wave is None else wave):
        if only is not None and tag not in only:
            continue
        row_file = Path(out_dir) / f"dense_{tag}.row.json"
        if row_file.exists():
            print(f"[skip] {tag} (row exists)", flush=True)
            rows.append(json.loads(row_file.read_text()))
            continue
        print(f"[train] {tag}: d={d} w={w} mc={mc}", flush=True)
        rows.append(train_candidate(tag, d, w, mc, data_yaml=data_yaml, out_dir=out_dir,
                                    epochs=epochs, imgsz=imgsz, batch=batch, device=device))
    return rows


def write_report(out_dir: Path, *, anchors: dict | None = None) -> dict:
    rows = [json.loads(f.read_text()) for f in sorted(Path(out_dir).glob("dense_*.row.json"))]
    payload = assemble_wave_report(rows, anchors=DENSE_ANCHORS if anchors is None else anchors)
    (Path(out_dir) / "dense_scaling.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=ROOT / "dataset" / "dataset.yaml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "dense_scaling")
    p.add_argument("--only", type=str, default=None, help="comma list of wave tags")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--wave", choices=sorted(WAVES), default="1",
                   help="which wave to run/tag (1=original 6, 2=finer width sweep)")
    p.add_argument("--report-only", action="store_true",
                   help="assemble dense_scaling.json from existing row files")
    a = p.parse_args(argv)

    if a.report_only:
        payload = write_report(a.out_dir)
        for row in payload["rows"]:
            print(f"  {row['tag']:8s} d={row['depth_mult']:.2f} w={row['width_mult']:.2f} "
                  f"params={row['params']:,}  map={row['map']:.4f}")
        return 0

    wave = WAVES[a.wave]
    only = a.only.split(",") if a.only else None
    unknown = set(only or []) - set(wave_tags(wave))
    if unknown:
        raise SystemExit(f"unknown wave tags: {sorted(unknown)}")
    run_wave(data_yaml=a.data, out_dir=a.out_dir, only=only, epochs=a.epochs,
             imgsz=a.imgsz, batch=a.batch, device=a.device, wave=wave)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
