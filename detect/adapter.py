"""1x1 channel adapters bridging the OFA backbone's P3/P4/P5 to a YOLO-pose head.

``supernet.pose_backbone.PoseBackbone`` emits feature maps at the fixed channels
``(40, 112, 160)``; a YOLO11n-pose head expects its own neck-output channels. ``ChannelAdapter``
is a per-scale 1x1 conv that remaps the former to the latter, leaving spatial dims untouched.
Torch-only (no ultralytics) so it unit-tests under ``.venv`` independently of the head graft.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ChannelAdapter(nn.Module):
    """Per-scale 1x1 convs mapping ``in_channels`` -> ``out_channels`` for the 3 feature maps."""

    def __init__(self, in_channels: Sequence[int], out_channels: Sequence[int]) -> None:
        super().__init__()
        self.out_channels: tuple[int, ...] = tuple(out_channels)
        self.adapters = nn.ModuleList(
            nn.Conv2d(ci, co, kernel_size=1)
            for ci, co in zip(in_channels, out_channels, strict=True)  # strict: lengths must match
        )

    def forward(self, feats: Sequence[Tensor]) -> tuple[Tensor, ...]:
        return tuple(adapt(f) for adapt, f in zip(self.adapters, feats, strict=True))
