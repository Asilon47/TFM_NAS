"""Expose a sampled OFA-MBv3 subnet as a P3/P4/P5 detection/pose backbone.

The NAS pivot (D1 = gate-pose) repurposes each OFA subnet as a *feature extractor* under a
YOLO-pose head instead of an ImageNet classifier. A detection/pose head consumes three
multi-scale feature maps; this wraps a subnet so its stride-8/16/32 activations come out
directly, dropping the classification head (``final_expand``/``feature_mix``/``classifier``).

Where the taps sit (see ``catalog.ofa_mbv3``): ``blocks[0]`` is OFA's fixed first block, then
the five elastic stages. Stages 1/3/4 end at strides 8/16/32, so the taps are those stages'
last-block indices — pure cumulative sums of the active depths ``d`` (``stage_tap_indices``).

Because OFA-w1.0 fixes the per-stage output widths (only ks/e/d are elastic), the tap channels
are **invariant across the whole search space**: ``out_channels == (40, 112, 160)`` for every
arch. One fixed neck/head/adapter therefore serves every sampled backbone — what makes
"search the backbone, freeze the head" tractable (see ``detect/pose_model.py``).

The tap math is pure (``.venv``-testable); the forward only needs torch (also in ``.venv``),
so ``tests/test_pose_backbone.py`` exercises it with a stub backbone. The real OFA wiring is
the ``__main__`` smoke test below, run under ``.venv-nas`` (mirrors ``supernet/sampler.py``)::

    python -m supernet.pose_backbone
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from catalog.ofa_mbv3 import STAGES

# OFA-MBv3 stage indices whose outputs land at strides 8 / 16 / 32. Stages 2 and 3 are both
# stride-16; P4 taps the deeper (stage 3), the standard FPN choice for the last map per stride.
P3_STAGE, P4_STAGE, P5_STAGE = 1, 3, 4

# Tap channel counts = those stages' output widths. Fixed across the search space (w1.0).
FEATURE_TAP_CHANNELS: tuple[int, int, int] = (
    STAGES[P3_STAGE]["out_c"],
    STAGES[P4_STAGE]["out_c"],
    STAGES[P5_STAGE]["out_c"],
)


def stage_tap_indices(depths: Sequence[int]) -> tuple[int, int, int]:
    """Block indices of the P3/P4/P5 taps for a subnet with per-stage active ``depths``.

    The active block list is ``[first_block] + stage0(d0) + ... + stage4(d4)``, so the last
    block of stage ``s`` sits at index ``sum(depths[: s + 1])`` (the leading first block makes
    the cumulative sum land on the stage's final block). P5 is always ``blocks[-1]``.
    """
    if len(depths) != len(STAGES):
        raise ValueError(
            f"depths must have length {len(STAGES)} (one per OFA stage), got {len(depths)}"
        )
    p3 = sum(depths[: P3_STAGE + 1])
    p4 = sum(depths[: P4_STAGE + 1])
    p5 = sum(depths[: P5_STAGE + 1])
    return p3, p4, p5


class PoseBackbone(nn.Module):
    """Wrap an OFA subnet so ``forward`` returns its ``(P3, P4, P5)`` feature maps.

    ``subnet`` is any module exposing ``.first_conv`` and a ``.blocks`` sequence (OFA's
    ``MobileNetV3``); ``depths`` is the arch's per-stage active depth list (``arch["d"]``),
    which fixes the tap indices. Only the stem + MBConv blocks are retained — the classifier
    head is dropped.
    """

    out_channels: tuple[int, int, int] = FEATURE_TAP_CHANNELS

    def __init__(self, subnet: nn.Module, depths: Sequence[int]) -> None:
        super().__init__()
        self.first_conv: nn.Module = subnet.first_conv  # type: ignore[assignment]
        self.blocks: nn.ModuleList = subnet.blocks  # type: ignore[assignment]
        self._taps = stage_tap_indices(depths)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3_i, p4_i, p5_i = self._taps
        x = self.first_conv(x)
        outs: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in (p3_i, p4_i, p5_i):
                outs[i] = x
        return outs[p3_i], outs[p4_i], outs[p5_i]


if __name__ == "__main__":  # real OFA integration smoke test — run under .venv-nas
    from supernet.sampler import load_supernet, random_arch, sample

    supernet = load_supernet()
    arch = random_arch(supernet)
    backbone = PoseBackbone(sample(arch, supernet), arch["d"]).eval()
    with torch.no_grad():
        feats = backbone(torch.randn(1, 3, 640, 640))
    print(f"arch d={arch['d']}  out_channels={backbone.out_channels}")
    for name, t in zip(("P3", "P4", "P5"), feats, strict=True):
        print(f"  {name}: {tuple(t.shape)}")
    expected = ((1, 40, 80, 80), (1, 112, 40, 40), (1, 160, 20, 20))
    assert tuple(tuple(t.shape) for t in feats) == expected, "unexpected tap shapes"
    print("PoseBackbone forward OK: (1,3,640,640) -> P3/P4/P5")
