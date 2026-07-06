"""CP 4.4 — graft-seam applicability: identity-embedding init for the ChannelAdapter.

The pivot's re-scope of CP 4.4 (procedure.md "Plan pivot"): the original OFA-space graph diff
served BO warm-starts that no longer exist; what the refinement track needs is Net2Net *at the
graft seam*. The ChannelAdapter's 1×1 convs (40→64, 112→128, 160→256) are random-initialized
today, so the frozen donor head spends its first proxy epochs looking at random channel
mixtures — the LP-FT distortion CP 2.4 diagnosed, one level deeper.
:func:`identity_embed_conv1x1_` re-initializes an expanding 1×1 conv as the identity on its
first ``in_channels`` outputs plus Net2Wider-replicated copies (:func:`net2net.wider.
widen_mapping`) for the extras, so the head sees the backbone's real features from step 0.

**An initialization prior, NOT function preservation.** Net2Wider's consumer-side division is
impossible here — the consumer is the frozen, donor-trained Pose head whose weights must not be
touched. Whether the prior helps is exactly what CP 5.2's V0-vs-V1 ablation measures.

Pure torch (``.venv``/CI-testable); wired as
``detect.pose_model.build_grafted_pose_model(adapter_init="net2wider")``.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from net2net.wider import widen_mapping

ADAPTER_INITS: tuple[str, ...] = ("net2wider",)


def identity_embed_conv1x1_(conv: nn.Conv2d, *, seed: int = 0) -> list[int]:
    """In place: re-init an expanding 1×1 conv as identity + replicated extras; zero bias.

    Output channel ``j`` becomes an exact copy of input channel ``mapping[j]`` — the identity
    for ``j < in_channels``, a Net2Wider-replicated source for the extras. Returns the mapping.
    """
    if tuple(conv.kernel_size) != (1, 1):
        raise ValueError(f"identity embedding is defined for 1x1 convs, got {conv.kernel_size}")
    if conv.groups != 1:
        raise ValueError("identity embedding needs groups=1")
    if conv.out_channels < conv.in_channels:
        raise ValueError(
            f"adapter must be expanding (out {conv.out_channels} < in {conv.in_channels})")
    mapping = widen_mapping(conv.in_channels, conv.out_channels, random.Random(seed))
    with torch.no_grad():
        conv.weight.zero_()
        for j, src in enumerate(mapping):
            conv.weight[j, src, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()
    return mapping


def apply_adapter_init(adapter: nn.Module, kind: str, *, seed: int = 0) -> list[list[int]]:
    """Apply a named init to every per-scale conv of a ChannelAdapter(-shaped) module.

    ``adapter`` must expose an ``adapters`` ModuleList of 1×1 convs
    (``detect.adapter.ChannelAdapter``). Each scale gets a distinct deterministic seed
    (``seed + scale_index``). Returns the per-scale mappings.
    """
    if kind not in ADAPTER_INITS:
        raise ValueError(f"unknown adapter_init {kind!r}; known: {ADAPTER_INITS}")
    convs = getattr(adapter, "adapters", None)
    if convs is None:
        raise TypeError("expected a ChannelAdapter-like module with an .adapters ModuleList")
    return [identity_embed_conv1x1_(conv, seed=seed + i) for i, conv in enumerate(convs)]
