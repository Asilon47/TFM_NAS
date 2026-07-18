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

# Pruning-as-search technique ladder (CP 6.2-G program): name → prune_graft() knobs.
# "uniform" is the floor configuration every earlier rung used (per-layer ratio, magnitude).
TECHNIQUES: dict[str, dict] = {
    "uniform": {"global_pruning": False, "importance": "l2"},
    "global_l2": {"global_pruning": True, "importance": "l2"},
    "global_taylor": {"global_pruning": True, "importance": "taylor"},
}
_TECH_ABBREV = {"uniform": "", "global_l2": "gl2", "global_taylor": "gtay"}


def run_tag(ratio: float, *, technique: str = "uniform", iterative_steps: int = 1,
            seed: int = 0) -> str:
    """Canonical artifact tag for a ladder point — default point keeps the legacy ``rNN``
    name so pre-program artifacts (prune_base_r15.pt, recover_graft_r60.pt, …) stay valid."""
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r} (have {sorted(TECHNIQUES)})")
    if iterative_steps < 1:
        raise ValueError(f"iterative_steps must be >= 1, got {iterative_steps}")
    tag = f"r{int(round(ratio * 100)):02d}"
    if _TECH_ABBREV[technique]:
        tag += f"_{_TECH_ABBREV[technique]}"
    if iterative_steps > 1:
        tag += f"_it{iterative_steps}"
    if seed != 0:
        tag += f"_s{seed}"
    return tag


def accumulate_pose_grads(
    model: Any,
    *,
    data_yaml: Any = None,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    n_batches: int = 8,
) -> int:
    """Populate ``.grad`` on every trainable param with summed pose-loss gradients.

    Taylor importance prep: first-order saliency reads ``w.grad * w``, so the gradients must
    exist BEFORE ``prune_graft(importance="taylor")`` — and must be re-accumulated between
    iterative steps (the ``between_steps`` hook). No optimizer step is taken.
    """
    from detect.evaluate import DEFAULT_DATA_YAML
    from eval.shortft import _build_pose_loader, _preprocess_batch, _to_device

    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    model = model.to(device).train()
    model.zero_grad()
    loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
    done = 0
    for raw in loader:
        if done >= n_batches:
            break
        loss, _items = model(_to_device(_preprocess_batch(raw), device))
        loss.sum().backward()
        done += 1
    if done == 0:
        raise RuntimeError("accumulate_pose_grads saw no batches — empty loader?")
    return done


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
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def dense_spec_ratio_dict(model: Any, spec: dict) -> tuple[dict, list]:
    """Per-stage spec → (pruning_ratio_dict, extra_ignored) for a DENSE (yolo11-pose) model.

    The dense mirror of :func:`prune.recover_graft.spec_ratio_dict` (Arm S of the beat-n
    program — "pruning-as-search stage 2" applied to the searched s39 donor): stage s's
    modules get ``stage_ratios[s]`` via the SAME yaml-index → stage maps the width search
    used (``search.dense_nas.BACKBONE_STAGE`` / ``HEAD_STAGE``), so the cut allocation
    speaks the search's own coordinates. Ratio-0 stages are protected outright (DepGraph
    still slices their in-channels when a producer shrinks); the Pose head's internals fall
    through to MetaPruner's default ratio = ``rest_ratio``.
    """
    from search.dense_nas import BACKBONE_STAGE, HEAD_STAGE

    if spec["rest_ratio"] <= 0.0:
        raise ValueError("spec rest_ratio must be > 0 (MetaPruner default ratio)")
    layers = list(model.model)
    n_backbone = 1 + max(BACKBONE_STAGE)          # yolo11 backbone section length (11)
    if len(layers) != n_backbone + 1 + max(HEAD_STAGE) + 1:  # + head section + Pose module
        raise ValueError(f"expected the stock yolo11-pose layout "
                         f"({n_backbone + max(HEAD_STAGE) + 2} modules), got {len(layers)} — "
                         "the stage maps would misalign")
    ratio_dict: dict = {}
    ignored: list = []
    for yaml_i, stage in {**BACKBONE_STAGE,
                          **{n_backbone + j: s for j, s in HEAD_STAGE.items()}}.items():
        r = float(spec["stage_ratios"][stage - 1])
        if r > 0.0:
            ratio_dict[layers[yaml_i]] = r
        else:
            ignored.append(layers[yaml_i])
    return ratio_dict, ignored


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


def lr_schedule_factor(gstep: int, steps_per_epoch: int, *, epochs: int,
                       cos_lr: bool, warmup_epochs: float) -> float:
    """Recipe-lite LR multiplier at global step ``gstep`` (linear warmup → cosine to 0).

    Pure so the schedule shape is testable without a training loop; with both knobs off
    it is the constant 1.0 the bare loop always used.
    """
    import math

    e = gstep / max(1, steps_per_epoch)
    if warmup_epochs > 0 and e < warmup_epochs:
        return max(e / warmup_epochs, 1e-3)
    if not cos_lr:
        return 1.0
    t = (e - warmup_epochs) / max(epochs - warmup_epochs, 1e-9)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def _save_train_ckpt(path: Path, model: Any, optimizer: Any, *, epoch: int,
                     step: int) -> None:
    """Atomic resume-ckpt write (tmp + rename) — a disconnect mid-save leaves the old file."""
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save({"epoch": epoch, "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state()}, tmp)
    tmp.replace(path)


def _load_train_ckpt(path: Path, model: Any, optimizer: Any) -> tuple[int, int] | None:
    """Resume from a ckpt if usable → (start_epoch, step); None (fresh start) otherwise.

    Model weights load FIRST — a shape-drifted ckpt raises before the optimizer is touched
    (the caller re-prunes deterministically, but GPU nondeterminism can flip near-tie
    channel picks between sessions)."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        return int(state["epoch"]) + 1, int(state.get("step", 0))
    except (RuntimeError, KeyError, ValueError, EOFError) as e:
        print(f"[resume] IGNORING unusable ckpt {path.name} ({e}) — training fresh",
              flush=True)
        return None


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
    teacher: Any = None,
    kd_alpha: float = 1.0,
    feat_kd: Any = None,
    kd_feat_alpha: float = 1.0,
    cos_lr: bool = False,
    warmup_epochs: float = 0.0,
    ema: bool = False,
    close_mosaic: int = 0,
    ckpt_path: Any = None,
    ckpt_every: int = 10,
) -> dict[str, float]:
    """Bare-AdamW recovery fine-tune (the shortft loop, on an EXISTING model) → pose mAP.

    ``teacher`` (a frozen raw-map teacher from ``distill.kd_loss.load_frozen_teacher``) adds
    output-level KD: ``pose_loss + kd_alpha · kd_map_loss`` (CP 8.2-early). ``None`` = the
    plain loop, bit-identical to the pre-KD behavior.

    ``feat_kd`` (a ``distill.kd_feat.FeatureKD`` already attached to this model + the
    teacher; Arm K) adds ``kd_feat_alpha · feat_kd.loss()`` — the FitNets regressors join
    the optimizer, never the model's state_dict. Requires ``teacher`` (its forward feeds
    the teacher-side taps).

    Recipe-lite (Arm R; all default-off = bit-identical to the bare loop): ``cos_lr``
    (cosine decay to 0 over the run) + ``warmup_epochs`` (linear ramp), ``ema``
    (ultralytics ModelEMA; final weights = the EMA weights, the stock-trainer convention),
    ``close_mosaic`` (disable mosaic/mixup for the last N epochs, the stock trainer's
    close_mosaic). The loader already applies stock train augmentation (get_cfg(DEFAULT_CFG)
    mosaic/HSV/flip), so these three are the actual recipe gap vs the donor's training —
    minus COCO pretrain, which no recovery can retrofit. EMA restarts on a resumed run
    (initialized from the resumed weights), same caveat class as the KD regressors.

    ``ckpt_path`` (free-tier hardening, 2026-07-13): save {epoch, model, optimizer, rng}
    every ``ckpt_every`` epochs (atomic tmp+rename) and auto-resume when the file exists —
    a Colab/Lightning disconnect loses ≤ ckpt_every epochs when the out-dir is durable
    (Drive / studio storage). A stale/shape-mismatched ckpt is ignored with a warning (the
    caller re-prunes deterministically, but GPU nondeterminism can flip near-tie channel
    picks). The ckpt is deleted only AFTER the final eval, so an eval-time crash resumes
    straight to eval. Resume is not bit-identical to an uninterrupted run (fresh loader RNG).
    """
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML, pose_map_model
    from eval.shortft import _build_pose_loader, _preprocess_batch, _seed_everything, _to_device

    _seed_everything(seed)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    model = model.to(device).train()
    if teacher is not None:
        from distill.kd_loss import kd_map_loss
        teacher = teacher.to(device)
    if feat_kd is not None:
        if teacher is None:
            raise ValueError("feat_kd needs a teacher — its taps come from the teacher "
                             "forward")
        feat_kd = feat_kd.to(device).train()
    loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
    train_params = [p for p in model.parameters() if p.requires_grad]
    if feat_kd is not None:
        train_params += list(feat_kd.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=lr)

    steps_per_epoch = max(1, len(loader))
    ema_obj = None
    if ema:
        from ultralytics.utils.torch_utils import ModelEMA
        ema_obj = ModelEMA(model)

    step = 0
    start_epoch = 0
    ckpt = Path(ckpt_path) if ckpt_path is not None else None
    if ckpt is not None and ckpt.exists():
        resumed = _load_train_ckpt(ckpt, model, optimizer)
        if resumed is not None:
            start_epoch, step = resumed
            print(f"[resume] {ckpt.name}: continuing at epoch {start_epoch + 1}/{epochs} "
                  f"({step} steps done)", flush=True)

    for epoch in range(start_epoch, epochs):
        if close_mosaic and epoch == max(epochs - close_mosaic, 0) and \
                hasattr(loader.dataset, "close_mosaic"):
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG

            hyp = get_cfg(DEFAULT_CFG)
            hyp.task, hyp.imgsz = "pose", imgsz     # same cfg _build_pose_loader used
            loader.dataset.close_mosaic(hyp=hyp)
            print(f"[recipe] mosaic closed at epoch {epoch + 1}/{epochs}", flush=True)
        for raw in loader:
            if cos_lr or warmup_epochs > 0:
                factor = lr_schedule_factor(step, steps_per_epoch, epochs=epochs,
                                            cos_lr=cos_lr, warmup_epochs=warmup_epochs)
                for g in optimizer.param_groups:
                    g["lr"] = lr * factor
            batch_dict = _to_device(_preprocess_batch(raw), device)
            if teacher is None:
                loss, _items = model(batch_dict)
                loss = loss.sum()
            else:
                preds = model(batch_dict["img"])
                task_loss, _items = model.loss(batch_dict, preds)
                with torch.no_grad():
                    tpreds = teacher(batch_dict["img"])
                kd = kd_map_loss(preds, tpreds)
                loss = task_loss.sum() + kd_alpha * kd
                feat = None
                if feat_kd is not None:
                    feat = feat_kd.loss()
                    loss = loss + kd_feat_alpha * feat
                if step == 0:
                    print(f"[kd] step0 task={float(task_loss.sum()):.3f} "
                          f"kd={float(kd):.3f} alpha={kd_alpha}"
                          + (f" feat={float(feat):.3f} feat_alpha={kd_feat_alpha}"
                             if feat is not None else ""), flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_obj is not None:
                ema_obj.update(model)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        print(f"[recover] epoch {epoch + 1}/{epochs} done ({step} steps)", flush=True)
        if ckpt is not None and ckpt_every > 0 and (
                (epoch + 1) % ckpt_every == 0 or epoch + 1 == epochs):
            _save_train_ckpt(ckpt, model, optimizer, epoch=epoch, step=step)
            print(f"[ckpt] {ckpt.name} @ epoch {epoch + 1}/{epochs}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    if ema_obj is not None:
        # Final weights = EMA weights (the stock-trainer convention) — what gets evaluated
        # is what the ladder saves and exports.
        model.load_state_dict(ema_obj.ema.state_dict())
    # Val a throwaway copy: the validator's AutoBackend fuses Conv+BN IN PLACE, which would
    # leave the model we return (and prune_ladder then saves/exports) without its BNs.
    import copy
    metrics = pose_map_model(copy.deepcopy(model).eval(), data_yaml=data_yaml, imgsz=imgsz,
                             device=device)
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()          # run complete — a leftover ckpt must not seed the next tag
    return metrics


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
    technique: str = "uniform",
    iterative_steps: int = 1,
    iter_recover_epochs: int = 5,
    seed: int = 0,
    taylor_batches: int = 8,
    teacher_path: Any = None,
    kd_alpha: float = 1.0,
    ratio_spec: dict | None = None,
    spec_tag: str = "spec",
    ckpt_every: int = 10,
    cos_lr: bool = False,
    warmup_epochs: float = 0.0,
    ema: bool = False,
    close_mosaic: int = 0,
) -> dict:
    """The full control-arm ladder: prune → BN re-estimate → recover → eval → export, per ratio.

    ``technique``/``iterative_steps``/``seed`` are the pruning-as-search knobs (CP 6.2-G
    program): TECHNIQUES maps the name to prune_graft's allocation/importance flags; iterative
    runs ``iter_recover_epochs`` of recovery between prune steps (BN re-estimated first);
    ``seed`` drives the recovery loop (de-noise waves) and lands in the artifact tags.

    ``ratio_spec`` (Arm S, beat-n program) replaces the uniform ladder with ONE per-stage
    allocation from ``prune.allocate_dense`` (``dense_spec_ratio_dict``): stage counts stay
    spec-pinned (global_pruning never passed → shapes importance-invariant, the allocate_v2
    contract), and the donor may be any yolo11-shaped .pt — the searched s39, not just the
    gate baseline. Spec runs get a tag-named resume ckpt (``ckpt_every`` epochs), the
    recover_graft pattern.
    """
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML, pose_map_model
    from eval.shortft import _build_pose_loader, _preprocess_batch
    from net2net.bn import reestimate_bn
    from prune.prune_graft import prune_graft
    from prune.yolo_tp_prep import prepare_yolo_for_pruning_

    tech = TECHNIQUES[technique]  # KeyError early on a bad name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml
    teacher = None
    if teacher_path is not None:
        from distill.kd_loss import load_frozen_teacher
        teacher = load_frozen_teacher(teacher_path, device=device)

    donor_model = load_baseline_model(donor)
    # Params counted pre-val: the validator fuses Conv+BN in place, dropping the BN params.
    donor_params = sum(p.numel() for p in donor_model.parameters())
    donor_metrics = pose_map_model(donor_model.to(device), data_yaml=data_yaml,
                                   imgsz=imgsz, device=device)
    donor_row = {"path": str(donor), "params": donor_params, **donor_metrics}
    del donor_model

    rows: list[dict] = []
    points = ([ratio_spec["rest_ratio"]] if ratio_spec is not None else ladder_plan(ratios))
    for ratio in points:
        # Tag first — it names the resume ckpt (spec runs only; the uniform ladder keeps its
        # legacy no-resume behavior and rNN names).
        if ratio_spec is not None:
            tag = spec_tag if seed == 0 else f"{spec_tag}_s{seed}"
        else:
            tag = run_tag(ratio, technique=technique, iterative_steps=iterative_steps,
                          seed=seed)
        if teacher is not None:
            tag += "_kd"
        if cos_lr or warmup_epochs > 0 or ema or close_mosaic:
            tag += "_rl"        # recipe-lite rides the tag — twins must not clobber

        model = load_baseline_model(donor)  # fresh weights per point — points are independent
        # yolo11-specific prep: split the C2f-family chunks + keep attention dense —
        # stock C3k2/C2PSA graphs break tp's DepGraph (see prune/yolo_tp_prep.py).
        extra_ignored = prepare_yolo_for_pruning_(model)

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
            prd, spec_ignored = dense_spec_ratio_dict(model, ratio_spec)
            # importance rides --technique; global_pruning is NOT passed — per-stage counts
            # stay pinned to the spec, so shapes (and act/latency) are importance-invariant
            # (the allocate contract, same as recover_graft's spec branch).
            report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                                 ratio=ratio, pruning_ratio_dict=prd,
                                 extra_ignored=extra_ignored + spec_ignored,
                                 importance=tech["importance"])
        else:
            report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                                 ratio=ratio, extra_ignored=extra_ignored,
                                 iterative_steps=iterative_steps,
                                 between_steps=(_between if iterative_steps > 1 else None),
                                 **tech)

        reestimate_bn(model.to(device), bn_feed)
        metrics = recovery_finetune(model, epochs=epochs, lr=lr, device=device, imgsz=imgsz,
                                    batch=batch, data_yaml=data_yaml, seed=seed,
                                    max_steps=max_steps, teacher=teacher, kd_alpha=kd_alpha,
                                    cos_lr=cos_lr, warmup_epochs=warmup_epochs, ema=ema,
                                    close_mosaic=close_mosaic,
                                    ckpt_path=(out_dir / f"ckpt_{tag}.pt"
                                               if ratio_spec is not None else None),
                                    ckpt_every=ckpt_every)

        weights = out_dir / f"prune_base_{tag}.pt"
        torch.save(model.state_dict(), str(weights))
        onnx = out_dir / f"prune_base_{tag}_640.onnx"
        _export_deploy_onnx(model, onnx, imgsz=imgsz)
        row = {"ratio": ratio, "params": report["params_after"],
               "params_sparsity": report["params_sparsity"], "all_rounded": report["all_rounded"],
               "n_convs_changed": report["n_convs_changed"],
               "technique": ("dense_spec" if ratio_spec is not None else technique),
               "iterative_steps": iterative_steps, "seed": seed,
               "kd": (None if teacher is None else {"teacher": str(teacher_path),
                                                    "alpha": kd_alpha}), **metrics,
               "weights": str(weights), "onnx": str(onnx)}
        if ratio_spec is not None:
            row["spec"] = {k: ratio_spec.get(k) for k in
                           ("stage_ratios", "rest_ratio", "predicted_fp32_ms",
                            "fp16_estimate_ms", "act_mbytes_honest")}
        if cos_lr or warmup_epochs > 0 or ema or close_mosaic:
            row["recipe"] = {"cos_lr": cos_lr, "warmup_epochs": warmup_epochs,
                             "ema": ema, "close_mosaic": close_mosaic}
        (out_dir / f"prune_base_{tag}.meta.json").write_text(json.dumps(
            {"pruned_baseline": True, **row}, indent=2) + "\n")
        rows.append(row)
        print(f"[ladder] ratio={ratio:.2f} tech={technique} params={row['params']:,} "
              f"map={row['map']:.4f}", flush=True)

    payload = assemble_ladder_report(donor_row, rows)
    payload["technique"] = ("dense_spec" if ratio_spec is not None else technique)
    payload["iterative_steps"] = iterative_steps
    payload["seed"] = seed
    if ratio_spec is not None:
        payload["spec_tag"] = spec_tag
    (out_dir / "prune_baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donor", type=Path,
                   default=ROOT / "runs/pose/experiments/gate_baseline/weights/best.pt")
    p.add_argument("--ratios", type=str, default="0.15,0.30,0.45")
    p.add_argument("--technique", choices=sorted(TECHNIQUES), default="uniform")
    p.add_argument("--iterative-steps", type=int, default=1)
    p.add_argument("--iter-recover-epochs", type=int, default=5,
                   help="recovery epochs between iterative prune steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taylor-batches", type=int, default=8,
                   help="gradient-accumulation batches for taylor importance")
    p.add_argument("--teacher", type=Path, default=None,
                   help="frozen raw-map KD teacher .pt (CP 8.2-early; e.g. the gate donor)")
    p.add_argument("--kd-alpha", type=float, default=1.0)
    p.add_argument("--ratio-spec", type=Path, default=None,
                   help="per-stage allocation spec (prune/specs/s39d_*.json from "
                        "prune.allocate_dense) — overrides --ratios with spec-pinned stage "
                        "counts (Arm S: prune the searched s39 donor); --technique then "
                        "selects the importance only")
    p.add_argument("--ckpt-every", type=int, default=10,
                   help="resume-ckpt cadence in epochs for spec runs (0 = off)")
    p.add_argument("--cos-lr", action="store_true",
                   help="recipe-lite (Arm R): cosine LR decay over the recovery")
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--ema", action="store_true",
                   help="recipe-lite: ModelEMA; final weights = the EMA weights")
    p.add_argument("--close-mosaic", type=int, default=0,
                   help="recipe-lite: disable mosaic/mixup for the last N epochs")
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

    ratio_spec = None
    spec_tag = "spec"
    if a.ratio_spec is not None:
        ratio_spec = json.loads(Path(a.ratio_spec).read_text())
        spec_tag = Path(a.ratio_spec).stem

    payload = prune_ladder(
        a.donor, ratios=[float(s) for s in a.ratios.split(",")], epochs=a.epochs, lr=a.lr,
        device=a.device, imgsz=a.imgsz, batch=a.batch, out_dir=a.out_dir,
        data_yaml=a.data_yaml, bn_batches=a.bn_batches, max_steps=a.max_steps,
        technique=a.technique, iterative_steps=a.iterative_steps,
        iter_recover_epochs=a.iter_recover_epochs, seed=a.seed,
        taylor_batches=a.taylor_batches, teacher_path=a.teacher, kd_alpha=a.kd_alpha,
        ratio_spec=ratio_spec, spec_tag=spec_tag, ckpt_every=a.ckpt_every,
        cos_lr=a.cos_lr, warmup_epochs=a.warmup_epochs, ema=a.ema,
        close_mosaic=a.close_mosaic)
    print(f"donor map={payload['donor']['map']:.4f}")
    for row in payload["rows"]:
        print(f"  ratio={row['ratio']:.2f}  params={row['params']:,}  "
              f"map={row['map']:.4f}  ({row['delta_map_vs_donor']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
