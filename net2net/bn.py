"""CP 4.3 — BatchNorm handling for function-preserving edits (the recorded decision).

**Decision (CP 4.3): re-estimate, then invert** — chosen over the alternative "freeze BN for the
first warm-start epoch". Rationale: (1) deterministic and optimizer-free — one forward-only
pass, no coupling to a training schedule; (2) reusable verbatim by Phase 6's post-prune recovery
and Phase 5's neck insertion; (3) freeze-first-epoch leaves stale statistics live inside the
frozen window and only reaches the same place if that epoch is long enough — strictly more
moving parts for the same guarantee.

Two pieces:

* :func:`reestimate_bn` — reset every BN's running stats and rebuild them by cumulative
  averaging (``momentum=None``) over forward-only passes with **only the BNs** in train mode
  (the "re-estimate BN" trick from the plan). No grads; momenta and train/eval modes restored.
* :func:`bn_to_identity_` — set ``(weight, bias) = (sqrt(running_var + eps), running_mean)`` so
  an **eval-mode** BN computes exactly the identity given its current running stats. This is
  what makes a *fresh* BN inserted by a CP 4.2 deepen function-preserving: eval-mode exactness
  holds for any stats (the affine inverts whatever eval normalizes with), while re-estimating
  first also makes **train-mode** behaviour near-identity (batch stats ≈ the inverted running
  stats), so post-insert training starts from a sane operating point instead of a distortion.

Pure torch (``.venv``/CI-testable).
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

_BatchNorm = nn.modules.batchnorm._BatchNorm


def reestimate_bn(model: nn.Module, batches: Iterable[Tensor]) -> int:
    """Rebuild every BN's running stats from ``batches`` (forward-only). Returns #batches used.

    Everything except the BNs stays in eval mode during the pass; BN momenta are set to ``None``
    (cumulative average over all batches — order-independent) and restored afterwards, as is the
    model's original train/eval mode.
    """
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bns:
        raise ValueError("model has no BatchNorm layers to re-estimate")
    momenta = [(bn, bn.momentum) for bn in bns]
    was_training = model.training

    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        # None = cumulative moving average across the whole pass (runtime-legal; the stub
        # types momentum as plain float).
        bn.momentum = None  # type: ignore[assignment]
        bn.train()
    n = 0
    with torch.no_grad():
        for x in batches:
            model(x)
            n += 1
    for bn, momentum in momenta:
        bn.momentum = momentum
    model.train(was_training)
    if n == 0:
        raise ValueError("no batches provided — running stats are now reset but unestimated")
    return n


def bn_to_identity_(bn: _BatchNorm) -> _BatchNorm:
    """In place: set the affine to invert the running stats, making eval-mode BN(x) == x."""
    if not bn.affine or not bn.track_running_stats:
        raise ValueError("bn_to_identity_ needs affine=True and track_running_stats=True")
    assert bn.running_var is not None and bn.running_mean is not None
    with torch.no_grad():
        bn.weight.copy_(torch.sqrt(bn.running_var + bn.eps))
        bn.bias.copy_(bn.running_mean)
    return bn
