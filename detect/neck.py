"""CP 5.1 — the zero-gated "nano-neck": function-preserving cross-scale fusion for the graft.

The deployed graft is neck-less: the adapter's P3/P4/P5 feed the Pose head with no cross-scale
mixing, while the yolo11n-pose baseline carries a full PAN-FPN. This module adds the *minimal
viable fusion*: top-down adds (P5→P4, then P4′→P3), each 1×1-projected, 2× nearest-upsampled,
and scaled by a **zero-initialized scalar gate — one per fusion edge** — so at init the neck is
exactly the identity and the frozen donor head sees unchanged inputs (day-0 function
preservation; the ReZero mechanism, and the same trick the old expansion plan's CP 5.5 reserved
for attention blocks). ``bottom_up=True`` adds the PAN-style return path (3×3 stride-2 conv +
gated add over the *updated* maps) for the V3 ablation arm.

Design evidence (docs/research/stageR_graft_interface.md): scalar per-edge gates are the
literature-backed form (ReZero: a single zero-init scalar per residual; BiFPN: per-edge scalar
fusion weights — no evidence favors per-channel gates, and they add TRT pointwise cost);
top-down-only is the weakest-but-cheapest topology. Whether *any* of it helps under a frozen
head is exactly what CP 5.2's V1-vs-V2-vs-V3 ablation measures — ``gate_values()`` exists so
the ablation can report whether the data ever turned the neck on.

Torch-only (no ultralytics) → unit-tests under ``.venv`` (tests/test_neck.py); the graft wiring
is ``detect.pose_model.build_grafted_pose_model(neck=...)``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

NECK_KINDS: tuple[str, ...] = ("topdown", "pan")


class ZeroGatedTopDownNeck(nn.Module):
    """Gated top-down (optionally + bottom-up) adds over ``(P3, P4, P5)``; identity at init.

    Every fusion edge is ``target + gate * proj(source)`` with ``gate`` a scalar Parameter
    initialized to 0 — the projections exist from step 0 but contribute nothing until the
    optimizer opens the gate (ReZero dynamics: at init the gate receives gradient, the
    projection does not; once the gate moves off 0, the projection trains too).
    """

    def __init__(self, channels: Sequence[int] = (64, 128, 256), *,
                 bottom_up: bool = False) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"expected 3 scales (P3, P4, P5), got {len(channels)}")
        c3, c4, c5 = channels
        self.channels: tuple[int, ...] = tuple(channels)
        self.bottom_up = bottom_up
        # top-down: coarse → fine (1×1 project to the finer width, ×2 nearest upsample)
        self.lat54 = nn.Conv2d(c5, c4, kernel_size=1)
        self.lat43 = nn.Conv2d(c4, c3, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.g54 = nn.Parameter(torch.zeros(()))
        self.g43 = nn.Parameter(torch.zeros(()))
        if bottom_up:
            # PAN-style return path over the updated maps (3×3 stride-2, the PANet form)
            self.down34 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
            self.down45 = nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1)
            self.g34 = nn.Parameter(torch.zeros(()))
            self.g45 = nn.Parameter(torch.zeros(()))

    def forward(self, feats: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = feats
        if getattr(self, "_gates_folded", False):
            p4 = p4 + self.up(self.lat54(p5))
            p3 = p3 + self.up(self.lat43(p4))
            if self.bottom_up:
                p4 = p4 + self.down34(p3)
                p5 = p5 + self.down45(p4)
            return p3, p4, p5
        p4 = p4 + self.g54 * self.up(self.lat54(p5))
        p3 = p3 + self.g43 * self.up(self.lat43(p4))
        if self.bottom_up:
            p4 = p4 + self.g34 * self.down34(p3)
            p5 = p5 + self.g45 * self.down45(p4)
        return p3, p4, p5

    @torch.no_grad()
    def fold_gates_(self) -> dict[str, float]:
        """Fold each scalar gate into its edge conv (weight and bias scaled by the gate) and
        drop the multiply from ``forward`` — exact because every edge is
        ``target + g * up_or_down(conv(source))`` with a bare Conv2d and a linear resample.

        Deploy/export transform (the GAP8 path needs it: NNTool's expression_matcher fuses
        ``mul+add`` into a 3-arg expression AutoTiler's AddNode template rejects). Idempotent;
        NOT for training — ReZero dynamics need the explicit gate.
        """
        if getattr(self, "_gates_folded", False):
            return {}
        folded = self.gate_values()
        edges = [(self.g54, self.lat54), (self.g43, self.lat43)]
        if self.bottom_up:
            edges += [(self.g34, self.down34), (self.g45, self.down45)]
        for gate, conv in edges:
            conv.weight.mul_(gate)
            if conv.bias is not None:
                conv.bias.mul_(gate)
            gate.fill_(1.0)
        self._gates_folded = True
        return folded

    def gate_values(self) -> dict[str, float]:
        """The learned gate magnitudes — CP 5.2 logs these ("did the data turn the neck on?")."""
        gates = {"g54": float(self.g54), "g43": float(self.g43)}
        if self.bottom_up:
            gates |= {"g34": float(self.g34), "g45": float(self.g45)}
        return gates


def build_neck(kind: str | None,
               channels: Sequence[int] = (64, 128, 256)) -> nn.Module | None:
    """The CP 5.2 variant switch: ``None`` (V0/V1) | ``"topdown"`` (V2) | ``"pan"`` (V3)."""
    if kind is None:
        return None
    if kind == "topdown":
        return ZeroGatedTopDownNeck(channels)
    if kind == "pan":
        return ZeroGatedTopDownNeck(channels, bottom_up=True)
    raise ValueError(f"unknown neck kind {kind!r}; known: {NECK_KINDS} (or None)")
