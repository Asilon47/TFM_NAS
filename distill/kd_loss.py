"""CP 8.2-early — output-level knowledge distillation for the pose recovery loops.

Pulled forward by the pruning-as-search program (procedure.md "CP 6.2-G CLOSED", Track 4): the
compressed champions of BOTH families recover with a teacher, symmetrically. Teacher =
``gate_best.pt`` (the gate-trained yolo11n-pose donor, 0.887 under this repo's validator — it
already ships in the Kaggle Dataset; the full Phase-8 teacher choice, yolo11s vs a trained
yolo11m, stays a later user decision).

Mechanism: every student in this repo carries the SAME Ultralytics Pose head class with its
semantic output convs protected from pruning (``prune.prune_graft.head_ignored_layers``), so in
train mode student and teacher emit structurally IDENTICAL raw multi-scale maps
(4·reg_max+nc box/cls per scale + nkpt·3 keypoint maps). KD is then a recursive mean-MSE over
that shared structure — no format introspection, no feature adapters. The teacher runs frozen
in eval (BN uses running stats); only its HEAD module's ``training`` flag is flipped so the
head returns raw maps instead of decoded predictions (the flag is per-module — child conv/BN
behavior is untouched).

Total loss in the recovery loop: ``pose_loss + kd_alpha · kd_map_loss(student, teacher)``.
The CP 8.3 DoD precedent applies per point: the KD twin must beat its no-KD twin by ≥ +0.3 mAP.

Teacher ladder (winner-v2-OFA Track 2t, user request 2026-07-13): the KD contract needs
student and teacher to emit the SAME raw head structure, so it is agnostic to teacher SIZE
within the yolo11 family — any **gate-trained yolo11n/s/m/l/x-pose** works as a drop-in teacher
(``--teacher``), no code change. It is NOT agnostic to head ARCHITECTURE: **yolo26**'s
``Pose26`` head emits a nested ``{one2many, one2one}`` dict with DFL-free 4-channel boxes (vs
yolo11's 4·reg_max=64) and an extra ``kpts_sigma`` (RLE) stream — structurally incompatible
with a yolo11-head student, so ``kd_map_loss`` detects it and raises with guidance rather than
distilling mismatched tensors. Verdict: use a yolo11-family teacher; yolo26→yolo11 KD would
need a cross-parameterization adapter whose payoff is doubtful on this saturated task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# yolo26 Pose26 train-mode head keys — their presence means an incompatible teacher head.
_YOLO26_HEAD_KEYS = ("one2many", "one2one")

# The distilled streams of the train-mode head dict ({boxes, scores, feats, kpts} in this
# ultralytics version): boxes/scores/kpts come from the PROTECTED output convs, so student and
# teacher shapes always match even after pruning. 'feats' (pre-output head features) is
# deliberately skipped — pruned widths differ there, and feature-mimic KD (adapters at
# P3/P4/P5) is an explicitly deferred user decision (old plan CP 8.2).
KD_KEYS = ("boxes", "scores", "kpts")


def kd_map_loss(student: Any, teacher: Any) -> Any:
    """Recursive mean-MSE over matched (nested) train-mode head outputs.

    Structures must match exactly — a mismatch means the models do not share the head
    contract, and silently truncating would distill against the wrong scale.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(student, dict):
        if not isinstance(teacher, dict):
            raise ValueError(f"KD structure mismatch: student dict vs {type(teacher).__name__}")
        # A yolo26 teacher (one2many/one2one head) cannot distill into a yolo11-head student:
        # DFL-free 4-ch boxes vs 64, extra kpts_sigma. Fail loudly with the fix, not silently.
        if any(k in teacher for k in _YOLO26_HEAD_KEYS) and not any(
                k in student for k in _YOLO26_HEAD_KEYS):
            raise ValueError(
                "KD teacher looks like a yolo26 Pose26 head (one2many/one2one) but the student "
                "is a yolo11-family head — incompatible output parameterization (DFL-free boxes, "
                "kpts_sigma). Use a gate-trained yolo11n/s/m/l/x-pose teacher instead "
                "(Track 2t: yolo11x is the zero-adaptation upgrade).")
        missing = [k for k in KD_KEYS if k not in student or k not in teacher]
        if missing:
            raise ValueError(f"KD head dict missing stream(s) {missing} — head contract "
                             f"changed? (student has {sorted(student)})")
        return sum(kd_map_loss(student[k], teacher[k]) for k in KD_KEYS)
    if isinstance(student, torch.Tensor):
        if not isinstance(teacher, torch.Tensor) or student.shape != teacher.shape:
            raise ValueError(
                f"KD structure mismatch: student {getattr(student, 'shape', type(student))} "
                f"vs teacher {getattr(teacher, 'shape', type(teacher))}")
        return F.mse_loss(student, teacher.detach())
    if isinstance(student, (list, tuple)):
        if not isinstance(teacher, (list, tuple)) or len(student) != len(teacher):
            raise ValueError(f"KD structure mismatch: student {type(student).__name__}"
                             f"[{len(student) if isinstance(student, (list, tuple)) else '?'}]"
                             f" vs teacher {type(teacher).__name__}")
        return sum(kd_map_loss(s, t) for s, t in zip(student, teacher, strict=True))
    raise TypeError(f"unsupported KD output node: {type(student).__name__}")


def load_frozen_teacher(donor: Path, device: str = "cpu") -> Any:
    """The gate-trained donor as a frozen raw-map teacher.

    eval() everywhere (BN running stats), every param frozen, then ONLY the head module's
    ``training`` flag flipped so its forward takes the raw-maps branch. Reuses
    ``prune.prune_baseline.load_baseline_model`` (criterion/args reset — harmless here).
    """
    from prune.prune_baseline import load_baseline_model

    teacher = load_baseline_model(Path(donor)).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.model[-1].training = True  # per-module flag: raw-map branch only, BN stays eval
    return teacher
