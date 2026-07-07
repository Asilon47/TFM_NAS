"""CP 6.2-B — the pruned-BASELINE control arm: DepGraph ladder on gate-trained yolo11n-pose.

Plan amendment 2026-07-07 ("dense-family arm", user decision A+B1+B2): Phase 6 prunes
winner-v1.5; this module runs the IDENTICAL ladder on the baseline the thesis compares
against — the gate-trained yolo11n-pose donor (`runs/pose/experiments/gate_baseline/…/best.pt`,
0.877 mAP). Products per ratio: a recovered checkpoint, a deploy ONNX (same export contract as
the baseline's own `yolo11n_pose_640.onnx` → directly Nano-benchable), and a report row. The
two ladders together form the controlled cross-family compression comparison (see
docs/research/stageR_prune_kd_edge.md — prune → recover → measure-through-TRT is the
evidenced pipeline; latencies for pruned nets are measured-only, off the LUT grid).

Protocol notes recorded in every payload: recovery is this repo's bare-AdamW loop (the
`eval.shortft` precedent), NOT the Ultralytics recipe that trained the donor — the ladder is
internally consistent (same recovery for every point and for Phase 6's winner ladder), and the
unpruned donor anchor is evaluated under the same validator.

Run (GPU)::

    python -m prune.prune_baseline --donor runs/pose/experiments/gate_baseline/weights/best.pt \
        --ratios 0.15,0.30,0.45 --epochs 50 --device cuda --imgsz 640 --batch 16 \
        --out-dir data/prune_baseline
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# DepGraph tracing resolution. Dependency tracing runs a real forward and holds every
# intermediate activation via grad_fn, so its HOST memory scales with the traced
# resolution — tracing at the 640 deploy size OOM-killed Kaggle's ~13 GB box
# (rc=137, 2026-07-07) while the result (channel-coupling groups + the data-free
# group-L2 importance) is resolution-independent. 128 is the CP 6.1 DoD-tested size;
# anything stride-32-safe and small works. Never trace at the deploy resolution.
TRACE_IMGSZ = 128

RECOVERY_CAVEAT = (
    "recovery = bare-AdamW loop (eval.shortft precedent), not the Ultralytics recipe that "
    "trained the donor; ladder-internal comparisons (and vs Phase 6's winner ladder) are "
    "protocol-consistent, the donor anchor is re-evaluated under the same validator. "
    "Latency per point is measured-only (Nano e2e bench; pruned widths are off the LUT grid)."
)


def ladder_plan(ratios: Sequence[float]) -> list[float]:
    """Validate + canonicalize the sparsity ladder (each in (0,1), deduped, ascending)."""
    if not ratios:
        raise ValueError("empty ratio ladder")
    for r in ratios:
        if not 0.0 < r < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {r}")
    return sorted(set(float(r) for r in ratios))


def assemble_ladder_report(donor: dict, rows: Sequence[dict]) -> dict:
    """The ladder report: per-row deltas vs the unpruned donor anchor + standing caveats."""
    if "map" not in donor:
        raise ValueError("donor anchor needs a 'map'")
    out_rows = []
    for row in sorted(rows, key=lambda r: r["ratio"]):
        out_rows.append({**row, "delta_map_vs_donor": row["map"] - donor["map"]})
    return {
        "donor": donor,
        "rows": out_rows,
        "best_row_ratio": (max(out_rows, key=lambda r: r["map"])["ratio"] if out_rows else None),
        "note": RECOVERY_CAVEAT,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def load_baseline_model(donor: Path) -> Any:
    """The gate-trained yolo11n-pose as a trainable fp32 ``PoseModel`` (DFL stays frozen).

    ``model.model[-1]`` is the same Ultralytics Pose head class the graft carries, so
    :func:`prune.prune_graft.head_ignored_layers` / :func:`prune_graft` apply unchanged.
    """
    from ultralytics import YOLO

    model = YOLO(str(donor)).model.float()
    for p in model.parameters():
        p.requires_grad_(True)
    head = model.model[-1]
    dfl = getattr(head, "dfl", None)
    if dfl is not None:  # fixed integration weight — frozen by design AND DepGraph-protected
        for p in dfl.parameters():
            p.requires_grad_(False)
    # Reset the loss plumbing to a clean, device-agnostic state, exactly as the graft path
    # does (detect.pose_model.default_pose_args). Two donor-checkpoint hazards (both surfaced
    # on Kaggle, 2026-07-08):
    #  * `model.criterion` is built eagerly at load; its DFL `proj` is a plain CPU tensor (not
    #    a registered buffer), so a later `model.to("cuda")` leaves proj on CPU → a cuda/cpu
    #    matmul mismatch in the loss's bbox_decode.
    #  * `model.args` is restored as a plain dict, so a freshly-built criterion trips on
    #    `self.hyp.box` (attribute access on a dict).
    # A fresh DEFAULT_CFG namespace + criterion=None → `.loss()` re-inits the criterion on the
    # model's CURRENT device with the standard pose loss gains on first training forward.
    from detect.pose_model import default_pose_args

    model.args = default_pose_args()
    model.criterion = None
    return model


def recovery_finetune(
    model: Any,
    *,
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    data_yaml: Any = None,
    seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Bare-AdamW recovery fine-tune (the shortft loop, on an EXISTING model) → pose mAP."""
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML, pose_map_model
    from eval.shortft import _build_pose_loader, _preprocess_batch, _seed_everything, _to_device

    _seed_everything(seed)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    model = model.to(device).train()
    loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    step = 0
    for epoch in range(epochs):
        for raw in loader:
            batch_dict = _to_device(_preprocess_batch(raw), device)
            loss, _items = model(batch_dict)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        print(f"[recover] epoch {epoch + 1}/{epochs} done ({step} steps)", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    # Val a throwaway copy: the validator's AutoBackend fuses Conv+BN IN PLACE, which would
    # leave the model we return (and prune_ladder then saves/exports) without its BNs.
    import copy
    return pose_map_model(copy.deepcopy(model).eval(), data_yaml=data_yaml, imgsz=imgsz,
                          device=device)


def _export_deploy_onnx(model: Any, out: Path, *, imgsz: int, opset: int = 17) -> None:
    """Deploy-tensor ONNX (head.export contract — same graph shape as the baseline export)."""
    import torch

    from detect.export_grafted_onnx import _legacy_export

    head = model.model[-1]
    head.export, head.format = True, "onnx"
    try:
        _legacy_export(model.eval().cpu(), torch.randn(1, 3, imgsz, imgsz), out,
                       opset=opset, output_names=["output0"])
    finally:
        head.export = False


def prune_ladder(
    donor: Path,
    *,
    ratios: Sequence[float],
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    out_dir: Path,
    data_yaml: Any = None,
    bn_batches: int = 16,
    max_steps: int | None = None,
) -> dict:
    """The full control-arm ladder: prune → BN re-estimate → recover → eval → export, per ratio."""
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML, pose_map_model
    from eval.shortft import _build_pose_loader, _preprocess_batch
    from net2net.bn import reestimate_bn
    from prune.prune_graft import prune_graft
    from prune.yolo_tp_prep import prepare_yolo_for_pruning_

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml

    donor_model = load_baseline_model(donor)
    # Params counted pre-val: the validator fuses Conv+BN in place, dropping the BN params.
    donor_params = sum(p.numel() for p in donor_model.parameters())
    donor_metrics = pose_map_model(donor_model.to(device), data_yaml=data_yaml,
                                   imgsz=imgsz, device=device)
    donor_row = {"path": str(donor), "params": donor_params, **donor_metrics}
    del donor_model

    rows: list[dict] = []
    for ratio in ladder_plan(ratios):
        model = load_baseline_model(donor)  # fresh weights per point — points are independent
        # yolo11-specific prep: split the C2f-family chunks + keep attention dense —
        # stock C3k2/C2PSA graphs break tp's DepGraph (see prune/yolo_tp_prep.py).
        extra_ignored = prepare_yolo_for_pruning_(model)
        report = prune_graft(model, torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ), ratio=ratio,
                             extra_ignored=extra_ignored)

        bn_feed = []
        loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
        for i, raw in enumerate(loader):
            if i >= bn_batches:
                break
            bn_feed.append(_preprocess_batch(raw)["img"].to(device))
        reestimate_bn(model.to(device), bn_feed)

        metrics = recovery_finetune(model, epochs=epochs, lr=lr, device=device, imgsz=imgsz,
                                    batch=batch, data_yaml=data_yaml, max_steps=max_steps)

        tag = f"r{int(round(ratio * 100)):02d}"
        weights = out_dir / f"prune_base_{tag}.pt"
        torch.save(model.state_dict(), str(weights))
        onnx = out_dir / f"prune_base_{tag}_640.onnx"
        _export_deploy_onnx(model, onnx, imgsz=imgsz)
        row = {"ratio": ratio, "params": report["params_after"],
               "params_sparsity": report["params_sparsity"], "all_rounded": report["all_rounded"],
               "n_convs_changed": report["n_convs_changed"], **metrics,
               "weights": str(weights), "onnx": str(onnx)}
        (out_dir / f"prune_base_{tag}.meta.json").write_text(json.dumps(
            {"pruned_baseline": True, **row}, indent=2) + "\n")
        rows.append(row)
        print(f"[ladder] ratio={ratio:.2f} params={row['params']:,} map={row['map']:.4f}",
              flush=True)

    payload = assemble_ladder_report(donor_row, rows)
    (out_dir / "prune_baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donor", type=Path,
                   default=ROOT / "runs/pose/experiments/gate_baseline/weights/best.pt")
    p.add_argument("--ratios", type=str, default="0.15,0.30,0.45")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "prune_baseline")
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--bn-batches", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    a = p.parse_args(argv)

    payload = prune_ladder(
        a.donor, ratios=[float(s) for s in a.ratios.split(",")], epochs=a.epochs, lr=a.lr,
        device=a.device, imgsz=a.imgsz, batch=a.batch, out_dir=a.out_dir,
        data_yaml=a.data_yaml, bn_batches=a.bn_batches, max_steps=a.max_steps)
    print(f"donor map={payload['donor']['map']:.4f}")
    for row in payload["rows"]:
        print(f"  ratio={row['ratio']:.2f}  params={row['params']:,}  "
              f"map={row['map']:.4f}  ({row['delta_map_vs_donor']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
