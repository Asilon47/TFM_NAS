"""Hook-based FLOPs counting for catalog blocks.

Counts Conv2d / ConvTranspose2d / Linear multiply-adds via forward hooks on
an eager module. Shared by the real sweep (``lut.orchestrate.run_sweep``) and
the dummy generator (``lut.orchestrate.gen_dummy_lut``) so the two can never
disagree; ``search/cost.py`` (CP 2.2+) can reuse it for predictive-model
features.

Kept simple on purpose: a static estimate that's useful as a model feature,
not a deployment guarantee (same caveat as lut/docs/schema.md).
"""
import torch
import torch.nn as nn


def count_flops_forward(module: nn.Module, input_shape) -> tuple[int, torch.Tensor]:
    """Run one zeros-forward through ``module``; return ``(flops, output)``.

    The output tensor is returned so callers can derive IO sizes (e.g.
    gen_dummy_lut's ``io_bytes``) without a second forward pass. Pass an
    ``eval()``-mode module: the zeros-forward would otherwise update
    BatchNorm running stats.
    """
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            oh, ow = out.shape[-2:]
            cin_per_group = m.in_channels // m.groups
            kh, kw = m.kernel_size
            total += 2 * m.out_channels * cin_per_group * kh * kw * oh * ow
        elif isinstance(m, nn.ConvTranspose2d):
            oh, ow = out.shape[-2:]
            kh, kw = m.kernel_size
            total += (2 * m.out_channels * m.in_channels * kh * kw * oh * ow
                      // max(1, m.groups))
        elif isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in module.modules()
               if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    try:
        with torch.no_grad():
            out = module(torch.zeros(*input_shape))
    finally:
        for h in handles:
            h.remove()
    return int(total), out


def count_flops(module: nn.Module, input_shape) -> int:
    """FLOPs only — see :func:`count_flops_forward`."""
    return count_flops_forward(module, input_shape)[0]
