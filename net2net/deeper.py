"""CP 4.2 — Net2DeeperNet: identity-initialized layers whose insertion changes nothing.

Chen et al., ICLR 2016 (§3.3). An inserted :func:`identity_conv2d` (Dirac kernel, zero bias,
odd kernel, ``padding=k//2``, stride 1) or :func:`identity_linear` (eye, zero bias) computes the
exact identity, so inserting one anywhere leaves the network function untouched. When the
insertion also adds an activation (the usual "conv + ReLU block" deepen), Net2Net requires an
**idempotent** activation — ``relu(relu(x)) == relu(x)`` — i.e. insert the pair *after* an
existing ReLU. A BatchNorm-carrying insert is CP 4.3's problem (``net2net/bn.py``): a fresh BN
is only the identity after its affine is set to invert its (re-estimated) statistics.

Pure torch (``.venv``/CI-testable).
"""
from __future__ import annotations

import torch
from torch import nn


def identity_conv2d(channels: int, kernel_size: int = 3) -> nn.Conv2d:
    """A ``channels -> channels`` conv that is exactly the identity (Dirac kernel, zero bias)."""
    if kernel_size % 2 == 0:
        raise ValueError(
            f"identity conv needs an odd kernel for symmetric padding, got {kernel_size}")
    conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)
    with torch.no_grad():
        nn.init.dirac_(conv.weight)
        assert conv.bias is not None
        conv.bias.zero_()
    return conv


def identity_linear(features: int) -> nn.Linear:
    """A ``features -> features`` linear that is exactly the identity (eye weight, zero bias)."""
    lin = nn.Linear(features, features)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(features))
        assert lin.bias is not None
        lin.bias.zero_()
    return lin


def inserted(seq: nn.Sequential, index: int, *modules: nn.Module) -> nn.Sequential:
    """A new ``Sequential`` with ``modules`` inserted at ``index`` (originals untouched)."""
    layers = list(seq)
    layers[index:index] = list(modules)
    return nn.Sequential(*layers)
