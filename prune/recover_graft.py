"""CP 6.2-G — prune the winner graft to a target sparsity, train to capacity, measure pose mAP.

The prune-graft latency screen (``models/screen_prune_graft/``) found that only the r=0.60 rung
(84 % param sparsity) beats the fp16 baseline; the first ladder (uniform/magnitude/one-shot,
2026-07-11) measured r40=0.8163 / r60=0.7589 — retention far above the crater-prior yet strictly
dominated on the measured frontier (procedure.md "CP 6.2-G CLOSED"). This module now drives the
**pruning-as-search technique ladder**: build the graft (OFA-pretrained backbone + warm-started
gate head), structurally prune it (``prune.prune_graft`` DepGraph — the SAME harness the latency
screen used) under a chosen ``--technique`` (uniform / global_l2 / global_taylor, optionally
iterative with interleaved recovery), BN-re-estimate, then train to capacity with this repo's
bare-AdamW loop and measure pose mAP.

Self-contained on Kaggle: needs only the gate-head donor + dataset (both in the attached Dataset),
no separately-shipped trained-graft weights. Protocol note vs the CP 6.2-B pruned-BASELINE control:
that arm prunes a fully gate-trained 0.877 donor then recovers; here the graft backbone is only
OFA-ImageNet-pretrained at prune time, so this is prune-then-TRAIN (like ``dense_scaling``'s
from-scratch training of a compressed net) rather than prune-then-recover. Both report the pruned
architecture's achievable mAP under this repo's validator; the unpruned reference is the winner-v1
full-FT 0.841 (``full_finetune.json``), not re-trained here.

``--index N`` swaps the winner topology for ``denoise_candidates.json[candidates][N]`` — the G1
topology-re-ranking probe (fallbacks idx3 / idx11 are the Stage-0-benched pair; the 0.841 anchor
applies to the winner topology only, so fallback deltas are indicative).

Run (GPU)::

    python -m prune.recover_graft --winner-dir state/winner_v1 --head-weights <gate best.pt> \
        --ratios 0.50 --technique global_taylor --iterative-steps 3 --epochs 100 \
        --device cuda --imgsz 640 --batch 16 --out-dir data/recover_graft
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prune.prune_baseline import TECHNIQUES, run_tag  # noqa: F401  (re-exported ladder vocab)

ROOT = Path(__file__).resolve().parents[1]

# The unpruned reference: winner-v1's full-FT mAP under this repo's protocol (full_finetune.json,
# 2026-07-05). The ladder's deltas are vs this, not a re-trained unpruned graft.
UNPRUNED_GRAFT_ANCHOR_MAP = 0.841


def spec_ratio_dict(model: Any, depths: list[int], spec: dict) -> tuple[dict, list]:
    """HALP spec → (pruning_ratio_dict, extra_ignored) with allocate.py's stage grouping.

    Stage s's blocks get ``stage_ratios[s]``; ratio-0 stages are protected outright (DepGraph
    still slices their in-channels when an upstream stage shrinks — that coupling is the point);
    adapter + head fall through to MetaPruner's default ratio = ``rest_ratio``.
    """
    if spec["rest_ratio"] <= 0.0:
        raise ValueError("spec rest_ratio must be > 0 (MetaPruner default ratio); "
                         "re-run prune.allocate with a lower target")
    blocks = list(model.model[0].blocks)
    ratio_dict: dict = {}
    ignored: list = []
    i = 1
    for s, ds in enumerate(depths):
        group = blocks[i:i + ds]
        if i == 1:
            group = [blocks[0], *group]
        r = float(spec["stage_ratios"][s])
        for m in group:
            if r > 0.0:
                ratio_dict[m] = r
            else:
                ignored.append(m)
        i += ds
    return ratio_dict, ignored


def load_arch_json(path: Any) -> tuple[dict, str]:
    """``--arch-json`` loader: a bare arch dict or an {'arch': ...} wrapper (the Track-1b
    ``prune/specs/minact_arch.json`` shape); arch_tag = the file stem."""
    data = json.loads(Path(path).read_text())
    arch = data.get("arch", data)
    if not isinstance(arch, dict) or not {"ks", "e", "d"} <= set(arch):
        raise ValueError(f"{path}: not an arch dict (need ks/e/d keys)")
    return arch, Path(path).stem


def load_candidate_arch(candidates_json: Any, index: int) -> dict:
    """The arch dict of ``denoise_candidates.json[candidates][index]`` (select by INDEX —
    ``d=[2,2,4,3,2]`` appears twice in the top-12, so depth lists are not unique keys)."""
    data = json.loads(Path(candidates_json).read_text())
    candidates = data["candidates"]
    if not 0 <= index < len(candidates):
        raise ValueError(f"index {index} out of range (have {len(candidates)} candidates)")
    return candidates[index]["arch"]


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
    technique: str = "uniform",
    iterative_steps: int = 1,
    iter_recover_epochs: int = 5,
    seed: int = 0,
    taylor_batches: int = 8,
    arch: dict | None = None,
    arch_tag: str = "winner",
    ratio_spec: dict | None = None,
    spec_tag: str = "halp",
    teacher_path: Any = None,
    kd_alpha: float = 1.0,
    ckpt_every: int = 10,
) -> dict:
    """Prune the graft at each ratio, train to capacity, eval mAP; per-point + report."""
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML
    from detect.pose_model import build_grafted_pose_model
    from eval.shortft import _build_pose_loader, _preprocess_batch, _seed_everything
    from eval.verify_winner import load_winner
    from net2net.bn import reestimate_bn
    from prune.prune_baseline import (
        TRACE_IMGSZ,
        _export_deploy_onnx,
        accumulate_pose_grads,
        assemble_ladder_report,
        ladder_plan,
        recovery_finetune,
    )
    from prune.prune_graft import prune_graft
    from supernet.sampler import load_supernet

    tech = TECHNIQUES[technique]  # KeyError early on a bad name
    sn = supernet if supernet is not None else load_supernet()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    teacher = None
    if teacher_path is not None:
        from distill.kd_loss import load_frozen_teacher
        teacher = load_frozen_teacher(teacher_path, device=device)
    if arch is None:
        arch = load_winner(Path(winner_dir))["arch"]

    # Unpruned anchor: winner-v1's own full-FT mAP (not re-trained). params from a plain build.
    anchor_model = build_grafted_pose_model(arch, supernet=sn)
    donor_row = {"path": f"winner-v1 full-FT (full_finetune.json); topology={arch_tag}",
                 "params": sum(p.numel() for p in anchor_model.parameters()),
                 "map": UNPRUNED_GRAFT_ANCHOR_MAP}
    del anchor_model

    rows: list[dict] = []
    points = ([ratio_spec["rest_ratio"]] if ratio_spec is not None else ladder_plan(ratios))
    for ratio in points:
        # Tag first (it names the resume ckpt): spec stem / run_tag + _kd + arch prefix.
        if ratio_spec is not None:
            tag = spec_tag if seed == 0 else f"{spec_tag}_s{seed}"
        else:
            tag = run_tag(ratio, technique=technique, iterative_steps=iterative_steps,
                          seed=seed)
        if teacher is not None:
            tag += "_kd"
        if arch_tag != "winner":
            tag = f"{arch_tag}_{tag}"

        # Fresh warm-head graft per point (OFA-pretrained backbone + gate-donor head, trainable):
        # points are independent, and prune_graft mutates in place. Seed BEFORE the build — the
        # 1x1 adapters are randomly initialized, so the seed owns them too.
        _seed_everything(seed)
        model = build_grafted_pose_model(arch, supernet=sn, head_weights=head_weights,
                                         freeze_head=False)

        bn_feed = []
        loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
        for i, raw in enumerate(loader):
            if i >= bn_batches:
                break
            bn_feed.append(_preprocess_batch(raw)["img"].to(device))

        if tech["importance"] == "taylor":
            accumulate_pose_grads(model, data_yaml=data_yaml, device=device, imgsz=imgsz,
                                  batch=batch, n_batches=taylor_batches)

        def _between(step_i: int, _model: Any = model, _bn_feed: list = bn_feed) -> None:
            # interleaved short recovery: fresh BN stats first, grads refreshed for taylor
            reestimate_bn(_model.to(device), _bn_feed)
            recovery_finetune(_model, epochs=iter_recover_epochs, lr=lr, device=device,
                              imgsz=imgsz, batch=batch, data_yaml=data_yaml, seed=seed,
                              max_steps=max_steps)
            if tech["importance"] == "taylor":
                accumulate_pose_grads(_model, data_yaml=data_yaml, device=device, imgsz=imgsz,
                                      batch=batch, n_batches=taylor_batches)

        if ratio_spec is not None:
            prd, spec_ignored = spec_ratio_dict(model, arch["d"], ratio_spec)
            # importance IS passed (pre-2026-07-13 spec runs silently pruned with l2);
            # global_pruning is NOT — per-stage counts stay pinned to the spec, so shapes
            # (and latency) are importance-invariant while taylor picks better channels.
            report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                                 ratio=ratio, pruning_ratio_dict=prd,
                                 extra_ignored=spec_ignored,
                                 importance=tech["importance"])
        else:
            report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                                 ratio=ratio, iterative_steps=iterative_steps,
                                 between_steps=(_between if iterative_steps > 1 else None),
                                 **tech)

        reestimate_bn(model.to(device), bn_feed)
        metrics = recovery_finetune(model, epochs=epochs, lr=lr, device=device, imgsz=imgsz,
                                    batch=batch, data_yaml=data_yaml, seed=seed,
                                    max_steps=max_steps, teacher=teacher, kd_alpha=kd_alpha,
                                    ckpt_path=out_dir / f"ckpt_{tag}.pt",
                                    ckpt_every=ckpt_every)

        weights = out_dir / f"recover_graft_{tag}.pt"
        torch.save(model.state_dict(), str(weights))
        onnx = out_dir / f"recover_graft_{tag}_640.onnx"
        _export_deploy_onnx(model, onnx, imgsz=imgsz)
        row = {"ratio": ratio, "params": report["params_after"],
               "params_sparsity": report["params_sparsity"], "all_rounded": report["all_rounded"],
               "n_convs_changed": report["n_convs_changed"],
               "technique": ("halp_spec" if ratio_spec is not None else technique),
               "iterative_steps": iterative_steps, "seed": seed, "arch_tag": arch_tag,
               "kd": (None if teacher is None else {"teacher": str(teacher_path),
                                                    "alpha": kd_alpha}), **metrics,
               "weights": str(weights), "onnx": str(onnx)}
        if ratio_spec is not None:
            row["spec"] = {k: ratio_spec.get(k) for k in
                           ("stage_ratios", "rest_ratio", "predicted_fp32_ms",
                            "fp16_estimate_ms", "target_fp32_ms")}
        (out_dir / f"recover_graft_{tag}.meta.json").write_text(
            json.dumps({"pruned_graft": True, **row}, indent=2) + "\n")
        rows.append(row)
        print(f"[graft-ladder] ratio={ratio:.2f} tech={row['technique']} arch={arch_tag} "
              f"params={row['params']:,} map={row['map']:.4f}", flush=True)

    payload = assemble_ladder_report(donor_row, rows)
    payload["technique"] = technique
    payload["iterative_steps"] = iterative_steps
    payload["seed"] = seed
    payload["arch_tag"] = arch_tag
    payload["protocol"] = ("prune-then-TRAIN from OFA-pretrained backbone + warm gate head "
                           "(dense_scaling-comparable); unpruned anchor = winner-v1 full-FT 0.841, "
                           "not re-trained here (fallback-topology deltas are indicative). Latency "
                           "per point is measured-only (weight-independent).")
    (out_dir / "recover_graft.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--head-weights", type=Path, required=True,
                   help="gate-trained donor .pt to warm-start the Pose head (dataset gate_best.pt)")
    p.add_argument("--ratios", type=str, default="0.40,0.60")
    p.add_argument("--technique", choices=sorted(TECHNIQUES), default="uniform")
    p.add_argument("--iterative-steps", type=int, default=1)
    p.add_argument("--iter-recover-epochs", type=int, default=5,
                   help="recovery epochs between iterative prune steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taylor-batches", type=int, default=8,
                   help="gradient-accumulation batches for taylor importance")
    p.add_argument("--candidates-json", type=Path,
                   default=ROOT / "state" / "winner_v1" / "denoise_candidates.json")
    p.add_argument("--index", type=int, default=None,
                   help="prune candidates[index] from --candidates-json instead of the winner "
                        "(G1 probe; select by index — d lists repeat)")
    p.add_argument("--arch-json", type=Path, default=None,
                   help="explicit arch dict (or {'arch': ...} wrapper, e.g. "
                        "prune/specs/minact_arch.json) instead of the winner — the Track-1b "
                        "min-act probe; arch_tag = the file stem")
    p.add_argument("--ratio-spec", type=Path, default=None,
                   help="HALP-lite allocation spec (prune/specs/halp_*.json from "
                        "prune.allocate) — overrides --ratios/--technique with per-stage ratios")
    p.add_argument("--teacher", type=Path, default=None,
                   help="frozen raw-map KD teacher .pt (CP 8.2-early; e.g. the gate donor)")
    p.add_argument("--kd-alpha", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "recover_graft")
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--bn-batches", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    p.add_argument("--ckpt-every", type=int, default=10,
                   help="resume-ckpt cadence in epochs (0 = off); the ckpt lives in --out-dir "
                        "so a durable out-dir (Drive/studio) survives free-tier disconnects")
    a = p.parse_args(argv)

    arch = None
    arch_tag = "winner"
    if a.index is not None and a.arch_json is not None:
        raise SystemExit("--index and --arch-json are mutually exclusive")
    if a.index is not None:
        arch = load_candidate_arch(a.candidates_json, a.index)
        arch_tag = f"idx{a.index}"
    elif a.arch_json is not None:
        arch, arch_tag = load_arch_json(a.arch_json)
    ratio_spec = None
    spec_tag = "halp"
    if a.ratio_spec is not None:
        ratio_spec = json.loads(Path(a.ratio_spec).read_text())
        spec_tag = Path(a.ratio_spec).stem

    payload = graft_prune_train_ladder(
        a.winner_dir, head_weights=a.head_weights,
        ratios=[float(s) for s in a.ratios.split(",")], epochs=a.epochs, lr=a.lr,
        device=a.device, imgsz=a.imgsz, batch=a.batch, out_dir=a.out_dir,
        data_yaml=a.data_yaml, bn_batches=a.bn_batches, max_steps=a.max_steps,
        technique=a.technique, iterative_steps=a.iterative_steps,
        iter_recover_epochs=a.iter_recover_epochs, seed=a.seed,
        taylor_batches=a.taylor_batches, arch=arch, arch_tag=arch_tag,
        ratio_spec=ratio_spec, spec_tag=spec_tag,
        teacher_path=a.teacher, kd_alpha=a.kd_alpha, ckpt_every=a.ckpt_every)
    print(f"unpruned anchor map={payload['donor']['map']:.4f}")
    for row in payload["rows"]:
        print(f"  ratio={row['ratio']:.2f}  params={row['params']:,}  "
              f"map={row['map']:.4f}  ({row['delta_map_vs_donor']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
