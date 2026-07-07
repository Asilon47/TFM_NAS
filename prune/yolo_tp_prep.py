"""Make a stock YOLO11 ``PoseModel`` Torch-Pruning-tractable (CP 6.2-B prep).

Two structural facts about yolo11 break tp's DepGraph (both found via the 2026-07-07/08
Kaggle rc=137 incident — see procedure.md):

* **C2PSA / attention**: the ``qkv → view(B, heads, dims, H·W) → matmul`` reshapes register
  ``_FlattenIndexMapping`` chains whose index lists grow multiplicatively during group
  building — RAM grows unbounded (12 GB+ observed before SIGKILL). No cheap fix exists;
  the attention blocks are **kept dense** via ``ignored_layers`` (~2 % of yolo11n params).
* **C2f-family ``chunk``**: ``C2f.forward`` (inherited by ``C3k2``) runs
  ``cv1(x).chunk(2, 1)`` — one conv feeding two index spaces. tp mis-attributes the
  coupling and builds groups with out-of-range indices (``IndexError: index 384 …
  size 256``). The canonical fix (tp's own YOLOv8 example) is rewriting each block into
  an explicit two-conv "split" form; :class:`C2fSplit` does that **function-preservingly**
  (conv/BN rows are independent → slicing them into two convs is exact, same param count).

``prepare_yolo_for_pruning_(model)`` applies both and returns the ignored-module list to
pass as ``prune_graft(..., extra_ignored=...)``. Run it on the **unfused** model (before
any Ultralytics val — AutoBackend fuses Conv+BN in place).
"""
from __future__ import annotations

from torch import nn

# Class NAMES (not imports — ultralytics stays a lazy dep) of blocks whose internal
# reshapes explode tp's index mappings. Kept dense wholesale.
ATTENTION_BLOCK_NAMES = ("C2PSA", "PSABlock", "Attention", "A2C2f", "AAttn", "ABlock")

# Ultralytics per-layer routing attributes BaseModel._predict_once reads; a replacement
# module must carry them or the parse-graph forward breaks.
_ROUTING_ATTRS = ("i", "f", "type", "np")


class C2fSplit(nn.Module):
    """A ``C2f``/``C3k2`` with the chunked ``cv1`` split into two exact-slice convs.

    Built FROM a live instance: ``cv2`` and the bottleneck list ``m`` are adopted by
    reference (no copy); ``cv0``/``cv1`` get the first/second half of the original
    ``cv1``'s conv+BN rows, so eval outputs match the source block exactly.
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        import torch
        from ultralytics.nn.modules.conv import Conv

        old = src.cv1
        if getattr(old, "bn", None) is None:
            raise ValueError("source block is fused (no BN) — split before any val/fuse")
        c = src.c
        in_c = old.conv.in_channels
        self.c = c
        self.cv0 = Conv(in_c, c, 1, 1)
        self.cv1 = Conv(in_c, c, 1, 1)
        with torch.no_grad():
            for dst, rows in ((self.cv0, slice(0, c)), (self.cv1, slice(c, 2 * c))):
                # eps/momentum live on the INSTANCE (ultralytics stamps 1e-3/0.03 in a
                # post-construction pass) — a fresh Conv has torch defaults; copy them or
                # the split drifts (max|Δ|≈5e-2 per block, found the hard way).
                dst.bn.eps = old.bn.eps
                dst.bn.momentum = old.bn.momentum
                dst.conv.weight.copy_(old.conv.weight[rows])
                dst.bn.weight.copy_(old.bn.weight[rows])
                dst.bn.bias.copy_(old.bn.bias[rows])
                dst.bn.running_mean.copy_(old.bn.running_mean[rows])
                dst.bn.running_var.copy_(old.bn.running_var[rows])
                dst.bn.num_batches_tracked.copy_(old.bn.num_batches_tracked)
                dst.act = old.act
        self.cv2 = src.cv2
        self.m = src.m
        self.to(old.conv.weight.device, old.conv.weight.dtype)
        self.train(src.training)  # fresh modules default to train mode — inherit src's

    def forward(self, x):  # C2f.forward with the chunk made explicit
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        import torch
        return self.cv2(torch.cat(y, 1))


def split_c2f_(model: nn.Module) -> int:
    """Replace every C2f-family block (incl. ``C3k2``) with :class:`C2fSplit`, in place.

    Returns the number of blocks rewritten. Routing attrs (``.i``/``.f``/…) carry over so
    Ultralytics' parse-graph forward keeps working.
    """
    from ultralytics.nn.modules.block import C2f

    n = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, C2f):
                new = C2fSplit(child)
                for attr in _ROUTING_ATTRS:
                    if hasattr(child, attr):
                        setattr(new, attr, getattr(child, attr))
                setattr(mod, name, new)
                n += 1
    return n


def attention_modules(model: nn.Module) -> list[nn.Module]:
    """Outermost attention-family blocks to keep dense (see module docstring).

    Only the top-level block of each nest is returned: tp's MetaPruner expands an ignored
    module to ``list(layer.modules())``, so the outer ``C2PSA`` already protects its inner
    ``PSABlock``/``Attention`` convs — returning those too would be redundant.
    """
    outer: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ATTENTION_BLOCK_NAMES:
            continue
        if any(name == o or name.startswith(o + ".") for o, _ in outer):
            continue  # nested inside an already-collected attention block
        outer.append((name, mod))
    return [m for _, m in outer]


def prepare_yolo_for_pruning_(model: nn.Module) -> list[nn.Module]:
    """Split the C2f-family blocks and return the attention blocks for ``extra_ignored``."""
    n = split_c2f_(model)
    ignored = attention_modules(model)
    print(f"[tp-prep] split {n} C2f-family block(s); keeping {len(ignored)} "
          f"attention block(s) dense", flush=True)
    return ignored
