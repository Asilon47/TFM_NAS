"""CP 4.1 — Net2WiderNet: function-preserving widening of Conv2d / Linear (+ BatchNorm).

Chen et al., *Net2Net: Accelerating Learning via Knowledge Transfer* (ICLR 2016,
https://arxiv.org/abs/1511.05641), §3.2. Widening a layer from ``old_out`` to ``new_out``
channels picks a mapping ``g: [0, new_out) → [0, old_out)`` that is the identity on the first
``old_out`` indices and uniform random replication on the extras; the producer's weights are
copied row-wise through ``g`` and the consumer's column-wise **divided by the replication
count**, so the widened pair computes exactly the original function. The identity survives any
elementwise activation between the pair (duplicated channels carry identical activations; the
consumer's division sums them back) and BatchNorm (per-channel statistics duplicate with their
channels).

Post-pivot role (procedure.md "Plan pivot"): the substrate for the graft-seam adapter init
(CP 4.4, ``net2net/graft_init.py``) and any later width edits — not BO warm-starts (Phase 3
closed without them).

Pure torch (``.venv``/CI-testable). All functions return **new** modules; inputs are untouched.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import cast

import torch
from torch import nn


def widen_mapping(old_out: int, new_out: int, rng: random.Random) -> list[int]:
    """The Net2Wider replication map: identity on the first ``old_out``, uniform on the rest."""
    if new_out < old_out:
        raise ValueError(
            f"cannot widen {old_out} -> {new_out} (shrinking is pruning — Phase 6, not Net2Net)")
    return list(range(old_out)) + [rng.randrange(old_out) for _ in range(new_out - old_out)]


def _consumer_scale(mapping: list[int]) -> torch.Tensor:
    """Per-new-channel divisor: how many times each new channel's source was replicated."""
    counts = Counter(mapping)
    return torch.tensor([float(counts[src]) for src in mapping])


def _like_conv2d(conv: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    """A fresh Conv2d sharing ``conv``'s geometry, with new channel counts.

    The casts narrow torch's ``tuple[int, ...]`` attribute types to the 2-tuples they always
    are at runtime for Conv2d (the stubs only accept ``int | tuple[int, int]``).
    """
    padding = conv.padding if isinstance(conv.padding, str) \
        else cast("tuple[int, int]", tuple(conv.padding))
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=cast("tuple[int, int]", tuple(conv.kernel_size)),
        stride=cast("tuple[int, int]", tuple(conv.stride)),
        padding=padding,
        dilation=cast("tuple[int, int]", tuple(conv.dilation)),
        bias=conv.bias is not None,
    )


def widen_conv2d(
    conv: nn.Conv2d,
    next_conv: nn.Conv2d,
    new_out: int,
    *,
    bn: nn.BatchNorm2d | None = None,
    seed: int = 0,
) -> tuple[nn.Conv2d, nn.Conv2d, nn.BatchNorm2d | None, list[int]]:
    """Widen ``conv``'s out-channels to ``new_out``, consumer-adjusting ``next_conv`` (and ``bn``).

    ``conv -> [bn] -> elementwise activation -> next_conv`` must be the actual dataflow, both
    convs ``groups=1`` (the graft seam and OFA pointwise convs qualify; a depthwise producer
    needs a different rule and is out of scope for v1). Returns
    ``(new_conv, new_next, new_bn, mapping)`` — fresh modules, originals untouched; the widened
    pair's output matches the original to float precision (DoD: 1e-5).
    """
    if conv.groups != 1 or next_conv.groups != 1:
        raise ValueError("Net2Wider v1 covers groups=1 convs only")
    if next_conv.in_channels != conv.out_channels:
        raise ValueError(
            f"consumer expects {next_conv.in_channels} in-channels but the producer emits "
            f"{conv.out_channels} — not a producer/consumer pair")
    if bn is not None and bn.num_features != conv.out_channels:
        raise ValueError(
            f"bn tracks {bn.num_features} features, producer emits {conv.out_channels}")

    mapping = widen_mapping(conv.out_channels, new_out, random.Random(seed))
    scale = _consumer_scale(mapping)

    new_conv = _like_conv2d(conv, conv.in_channels, new_out)
    new_next = _like_conv2d(next_conv, new_out, next_conv.out_channels)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[mapping])
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias[mapping])
        new_next.weight.copy_(next_conv.weight[:, mapping] / scale.view(1, -1, 1, 1))
        if next_conv.bias is not None and new_next.bias is not None:
            new_next.bias.copy_(next_conv.bias)

    new_bn = None if bn is None else _widen_bn(bn, mapping)
    return new_conv, new_next, new_bn, mapping


def _widen_bn(bn: nn.BatchNorm2d, mapping: list[int]) -> nn.BatchNorm2d:
    """A BN over the widened channels: parameters + running stats duplicated through ``mapping``."""
    new_bn = nn.BatchNorm2d(len(mapping), eps=bn.eps, momentum=bn.momentum,
                            affine=bn.affine, track_running_stats=bn.track_running_stats)
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight[mapping])
            new_bn.bias.copy_(bn.bias[mapping])
        if bn.track_running_stats and bn.running_mean is not None:
            assert new_bn.running_mean is not None and new_bn.running_var is not None
            assert bn.running_var is not None and bn.num_batches_tracked is not None
            assert new_bn.num_batches_tracked is not None
            new_bn.running_mean.copy_(bn.running_mean[mapping])
            new_bn.running_var.copy_(bn.running_var[mapping])
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def widen_linear(
    linear: nn.Linear,
    next_linear: nn.Linear,
    new_out: int,
    *,
    seed: int = 0,
) -> tuple[nn.Linear, nn.Linear, list[int]]:
    """The Linear twin of :func:`widen_conv2d` (same mapping + consumer-division rule)."""
    if next_linear.in_features != linear.out_features:
        raise ValueError(
            f"consumer expects {next_linear.in_features} features but the producer emits "
            f"{linear.out_features} — not a producer/consumer pair")
    mapping = widen_mapping(linear.out_features, new_out, random.Random(seed))
    scale = _consumer_scale(mapping)

    new_linear = nn.Linear(linear.in_features, new_out, bias=linear.bias is not None)
    new_next = nn.Linear(new_out, next_linear.out_features, bias=next_linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[mapping])
        if linear.bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(linear.bias[mapping])
        new_next.weight.copy_(next_linear.weight[:, mapping] / scale.view(1, -1))
        if next_linear.bias is not None and new_next.bias is not None:
            new_next.bias.copy_(next_linear.bias)
    return new_linear, new_next, mapping
