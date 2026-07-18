"""Arm K (beat-n program, 2026-07-18) — feature-mimic KD at the head-input taps.

Output-level KD (``kd_loss.kd_map_loss``) measured NULL on small taylor cuts and the
winner-v2 record names the remaining levers for a sub-1M student: temperature/KL on the DFL
bins, lower alpha, or **feature-level KD — not a bigger teacher**. This module is that
lever, in its literature form (FitNets regressors): a 1×1 projection per scale maps the
student's P3/P4/P5 head-INPUT taps to the teacher's widths, and the loss is mean-MSE after
the projection. Taps are captured with forward pre-hooks on each model's head module —
every student in this repo (graft, necked graft, pruned dense) carries the Ultralytics Pose
head as ``model.model[-1]`` taking ``[P3, P4, P5]``, so the hook point is family-invariant.

The regressors are TRAINING-ONLY: they live in this side module, join the optimizer, and
are never part of the student's ``state_dict`` or deploy export. When a student tap already
matches the teacher's width the projection is the identity (day-0 aligned, nothing to
learn). Resume note: the recovery loop's ckpt stores the student+optimizer only, so a
resumed run restarts the regressors fresh — acceptable for the pilot (they re-anneal within
an epoch); revisit if feature-KD graduates.

Total loss in the recovery loop:
``pose_loss + kd_alpha · kd_map_loss + kd_feat_alpha · FeatureKD.loss()``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def head_input_channels(model: Any) -> list[int]:
    """Per-scale head-input widths, read from the head's own first convs.

    ``model.model[-1]`` is the Ultralytics Pose head; ``cv2[i][0]`` is the first Conv of
    scale i's box branch, whose in_channels IS the P(3+i) tap width — robust to pruning
    (DepGraph rewrites these in_channels when the taps shrink).
    """
    head = model.model[-1]
    out: list[int] = []
    for scale_seq in head.cv2:
        first = scale_seq[0]
        conv = getattr(first, "conv", first)   # ultralytics Conv wraps .conv; tolerate bare
        out.append(int(conv.in_channels))
    return out


class FeatureKD(nn.Module):
    """FitNets-style feature mimic over the three head-input taps.

    Build AFTER pruning (student widths are only known then), then ``attach`` both models.
    The two training forwards (student with grad, teacher under no_grad) populate the tap
    stores through pre-hooks; ``loss()`` consumes and clears them.
    """

    def __init__(self, student_ch: list[int], teacher_ch: list[int]) -> None:
        super().__init__()
        if len(student_ch) != len(teacher_ch):
            raise ValueError(f"scale count mismatch: {student_ch} vs {teacher_ch}")
        self.proj = nn.ModuleList([
            nn.Identity() if s == t else nn.Conv2d(s, t, kernel_size=1)
            for s, t in zip(student_ch, teacher_ch, strict=True)])
        self._taps: dict[str, list[Any] | None] = {"student": None, "teacher": None}
        self._handles: list[Any] = []

    def attach(self, student: Any, teacher: Any) -> FeatureKD:
        """Register head pre-hooks on both models (idempotent per FeatureKD instance)."""
        self.detach_hooks()

        def _mk(which: str) -> Any:
            def hook(_mod: Any, args: tuple) -> None:
                feats = args[0]
                self._taps[which] = list(feats)
            return hook

        self._handles = [
            student.model[-1].register_forward_pre_hook(_mk("student")),
            teacher.model[-1].register_forward_pre_hook(_mk("teacher")),
        ]
        return self

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def loss(self) -> Any:
        """Sum of per-scale MSE(proj(student_tap), teacher_tap.detach()); clears the taps."""
        s_taps, t_taps = self._taps["student"], self._taps["teacher"]
        if s_taps is None or t_taps is None:
            raise RuntimeError("FeatureKD.loss() before both forwards ran — attach() the "
                               "models and run student + teacher on the batch first")
        if len(s_taps) != len(self.proj) or len(t_taps) != len(self.proj):
            raise ValueError(f"tap count mismatch: student {len(s_taps)} / teacher "
                             f"{len(t_taps)} vs {len(self.proj)} projections")
        total: Any = torch.zeros((), device=s_taps[0].device)
        for p, s, t in zip(self.proj, s_taps, t_taps, strict=True):
            total = total + F.mse_loss(p(s), t.detach())
        self._taps = {"student": None, "teacher": None}
        return total


def build_feature_kd(student: Any, teacher: Any) -> FeatureKD:
    """FeatureKD sized from the two models' actual head-input widths, hooks attached."""
    kd = FeatureKD(head_input_channels(student), head_input_channels(teacher))
    return kd.attach(student, teacher)
