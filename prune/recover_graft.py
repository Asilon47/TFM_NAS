"""CP 6.2 (graft arm) — does pruning the winner graft to a baseline-beating latency keep any mAP?

The prune-graft latency screen (``models/screen_prune_graft/``) found that only the r=0.60 rung
(84 % param sparsity) beats the fp16 baseline; r=0.40 (64 %) beats fp32 only. This trains those
pruned architectures to answer the accuracy half: build the winner graft (OFA-pretrained backbone
+ warm-started gate head), structurally prune it (``prune.prune_graft`` DepGraph — the SAME harness
the latency screen used), BN-re-estimate, then train it to capacity with this repo's bare-AdamW
loop (the ``prune_baseline`` recovery loop / ``dense_scaling`` from-scratch precedent) and measure
pose mAP.

Self-contained on Kaggle: needs only the gate-head donor + dataset (both in the attached Dataset),
no separately-shipped trained-graft weights. Protocol note vs the CP 6.2-B pruned-BASELINE control:
that arm prunes a fully gate-trained 0.877 donor then recovers; here the graft backbone is only
OFA-ImageNet-pretrained at prune time, so this is prune-then-TRAIN (like ``dense_scaling``'s
from-scratch training of a compressed net) rather than prune-then-recover. Both report the pruned
architecture's achievable mAP under this repo's validator; the unpruned reference is the winner-v1
full-FT 0.841 (``full_finetune.json``), not re-trained here.

Run (GPU)::

    python -m prune.recover_graft --winner-dir state/winner_v1 --head-weights <gate best.pt> \
        --ratios 0.40,0.60 --epochs 100 --device cuda --imgsz 640 --batch 16 \
        --out-dir data/recover_graft
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The unpruned reference: winner-v1's full-FT mAP under this repo's protocol (full_finetune.json,
# 2026-07-05). The ladder's deltas are vs this, not a re-trained unpruned graft.
UNPRUNED_GRAFT_ANCHOR_MAP = 0.841


def graft_prune_train_ladder(
    winner_dir: Any,
    *,
    head_weights: Any,
    ratios: Sequence[float],
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    out_dir: Any,
    data_yaml: Any = None,
    bn_batches: int = 16,
    max_steps: int | None = None,
    supernet: Any = None,
) -> dict:
    """Prune the winner graft at each ratio, train to capacity, eval mAP; per-point + report."""
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML
    from detect.pose_model import build_grafted_pose_model
    from eval.shortft import _build_pose_loader, _preprocess_batch
    from eval.verify_winner import load_winner
    from net2net.bn import reestimate_bn
    from prune.prune_baseline import (
        TRACE_IMGSZ,
        _export_deploy_onnx,
        assemble_ladder_report,
        ladder_plan,
        recovery_finetune,
    )
    from prune.prune_graft import prune_graft
    from supernet.sampler import load_supernet

    sn = supernet if supernet is not None else load_supernet()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    arch = load_winner(Path(winner_dir))["arch"]

    # Unpruned anchor: winner-v1's own full-FT mAP (not re-trained). params from a plain build.
    anchor_model = build_grafted_pose_model(arch, supernet=sn)
    donor_row = {"path": "winner-v1 full-FT (full_finetune.json)",
                 "params": sum(p.numel() for p in anchor_model.parameters()),
                 "map": UNPRUNED_GRAFT_ANCHOR_MAP}
    del anchor_model

    rows: list[dict] = []
    for ratio in ladder_plan(ratios):
        # Fresh warm-head graft per point (OFA-pretrained backbone + gate-donor head, trainable):
        # points are independent, and prune_graft mutates in place.
        model = build_grafted_pose_model(arch, supernet=sn, head_weights=head_weights,
                                         freeze_head=False)
        report = prune_graft(model, torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ), ratio=ratio)

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
        weights = out_dir / f"recover_graft_{tag}.pt"
        torch.save(model.state_dict(), str(weights))
        onnx = out_dir / f"recover_graft_{tag}_640.onnx"
        _export_deploy_onnx(model, onnx, imgsz=imgsz)
        row = {"ratio": ratio, "params": report["params_after"],
               "params_sparsity": report["params_sparsity"], "all_rounded": report["all_rounded"],
               "n_convs_changed": report["n_convs_changed"], **metrics,
               "weights": str(weights), "onnx": str(onnx)}
        (out_dir / f"recover_graft_{tag}.meta.json").write_text(
            json.dumps({"pruned_graft": True, **row}, indent=2) + "\n")
        rows.append(row)
        print(f"[graft-ladder] ratio={ratio:.2f} params={row['params']:,} "
              f"map={row['map']:.4f}", flush=True)

    payload = assemble_ladder_report(donor_row, rows)
    payload["protocol"] = ("prune-then-TRAIN from OFA-pretrained backbone + warm gate head "
                           "(dense_scaling-comparable); unpruned anchor = winner-v1 full-FT 0.841, "
                           "not re-trained here. Latency per point is the screen's measured-only "
                           "number (weight-independent).")
    (out_dir / "recover_graft.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--head-weights", type=Path, required=True,
                   help="gate-trained donor .pt to warm-start the Pose head (dataset gate_best.pt)")
    p.add_argument("--ratios", type=str, default="0.40,0.60")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "recover_graft")
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--bn-batches", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    a = p.parse_args(argv)

    payload = graft_prune_train_ladder(
        a.winner_dir, head_weights=a.head_weights,
        ratios=[float(s) for s in a.ratios.split(",")], epochs=a.epochs, lr=a.lr,
        device=a.device, imgsz=a.imgsz, batch=a.batch, out_dir=a.out_dir,
        data_yaml=a.data_yaml, bn_batches=a.bn_batches, max_steps=a.max_steps)
    print(f"unpruned anchor map={payload['donor']['map']:.4f}")
    for row in payload["rows"]:
        print(f"  ratio={row['ratio']:.2f}  params={row['params']:,}  "
              f"map={row['map']:.4f}  ({row['delta_map_vs_donor']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
