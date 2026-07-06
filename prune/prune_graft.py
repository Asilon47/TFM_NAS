"""CP 6.1 — DepGraph structured pruning harness for the grafted pose model.

Phase 6 (the latency lever; procedure.md "Plan pivot"): channel-level compression reaches
off-grid widths the OFA space cannot express. Built on Torch-Pruning's DepGraph (Fang et al.,
CVPR 2023 — see docs/research/stageR_prune_kd_edge.md): dependency-grouped pruning with
**group-L2 importance** (the paper's own recommendation) and ``round_to=16`` so pruned channel
counts stay tensor-core aligned (NVIDIA TRT: fp16 conv kernels degrade / implicitly pad when
in-channels aren't a multiple of 8; 16 covers fp16+int8 with one knob).

Two hard rules encoded here:

* **No frozen parameters.** DepGraph slices a consumer's in-channels when its producer's
  out-channels shrink — pruning upstream of a frozen module silently corrupts trained weights.
  :func:`prune_graft` refuses to run if anything has ``requires_grad=False`` (unfreeze the
  head; CP 6.2's recovery trains it).
* **Semantic outputs are untouchable.** The Pose head's per-scale output convs (box
  ``cv2[i][-1]`` = 4·reg_max, cls ``cv3[i][-1]`` = nc, keypoints ``cv4[i][-1]`` = nkpt·3) and
  the fixed-weight ``dfl`` define the output format — :func:`head_ignored_layers` collects
  them into DepGraph's ``ignored_layers``.

Latency claims for pruned nets are **measured-only** (e2e Nano benches; off the LUT grid).
GPU-free: the prune itself is pure CPU graph surgery — the CPU smoke (`tests/test_prune_graft.py`,
``.venv-nas``) IS the CP 6.1 DoD; the sparsity ladder + recovery + benches are CP 6.2.
"""
from __future__ import annotations

from typing import Any

from torch import nn


def head_ignored_layers(model: nn.Module) -> list[nn.Module]:
    """The graft's semantic-output modules DepGraph must never prune.

    ``model.model[-1]`` is the Ultralytics Pose head (the graft contract): the last conv of
    every ``cv2``/``cv3``/``cv4`` scale branch fixes the output channel semantics, and
    ``dfl.conv`` carries a fixed (non-trained) integration weight.
    """
    head = model.model[-1]  # type: ignore[index]
    ignored: list[nn.Module] = []
    for branch_name in ("cv2", "cv3", "cv4"):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        for scale_seq in branch:
            ignored.append(scale_seq[-1])
    dfl = getattr(head, "dfl", None)
    if dfl is not None:
        ignored.append(getattr(dfl, "conv", dfl))
    if not ignored:
        raise ValueError("no head output layers found — is model.model[-1] an Ultralytics "
                         "Pose head? Refusing to prune with nothing protected.")
    return ignored


def prune_graft(
    model: nn.Module,
    example_input: Any,
    *,
    ratio: float,
    round_to: int = 16,
    importance_p: int = 2,
    extra_ignored: list[nn.Module] | None = None,
) -> dict:
    """Structurally prune ``model`` in place by ``ratio`` (group sparsity); return a report.

    ``example_input`` traces the dependency graph (use the deploy shape, e.g.
    ``torch.randn(1, 3, 640, 640)``). The report carries params before/after, the achieved
    sparsity, and an alignment audit: every conv whose channel count *changed* must land on a
    multiple of ``round_to`` (unchanged convs keep their original counts).
    """
    import torch_pruning as tp

    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    ignored = head_ignored_layers(model) + list(extra_ignored or [])
    # Frozen params are only tolerable inside the *protected* modules (e.g. the DFL conv is
    # permanently frozen by design AND ignored, so DepGraph never touches its channels).
    # Anything frozen outside that set would be silently corrupted by consumer slicing.
    protected_ids = {id(p) for m in ignored for p in m.parameters()}
    frozen = [name for name, p in model.named_parameters()
              if not p.requires_grad and id(p) not in protected_ids]
    if frozen:
        raise ValueError(
            f"{len(frozen)} frozen parameter(s) (e.g. {frozen[0]!r}) — DepGraph slices "
            "consumer in-channels, so pruning around frozen weights corrupts them. Unfreeze "
            "the head (CP 6.2's recovery fine-tune trains it) and retry.")

    before = {name: (m.in_channels, m.out_channels)
              for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    params_before = sum(p.numel() for p in model.parameters())

    pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=tp.importance.GroupMagnitudeImportance(p=importance_p),
        pruning_ratio=ratio,
        ignored_layers=ignored,
        round_to=round_to,
    )
    pruner.step()

    params_after = sum(p.numel() for p in model.parameters())
    changed: list[dict] = []
    misaligned: list[str] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or name not in before:
            continue
        old_in, old_out = before[name]
        if (m.in_channels, m.out_channels) == (old_in, old_out):
            continue
        changed.append({"layer": name, "in": [old_in, m.in_channels],
                        "out": [old_out, m.out_channels]})
        # depthwise/group convs may carry channel counts DepGraph couples elsewhere; the
        # alignment contract applies to the pruned OUT channels of plain convs.
        if m.groups == 1 and m.out_channels != old_out and m.out_channels % round_to != 0:
            misaligned.append(name)

    return {
        "ratio": ratio,
        "round_to": round_to,
        "importance": f"group_l{importance_p}",
        "params_before": params_before,
        "params_after": params_after,
        "params_sparsity": 1.0 - params_after / params_before,
        "n_convs_changed": len(changed),
        "changed": changed,
        "misaligned": misaligned,
        "all_rounded": not misaligned,
    }
