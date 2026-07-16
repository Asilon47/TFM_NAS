"""Export a spec-pruned graft to ONNX for the GAP8 cycle probe — no training, no data.

CP 10.1 measured the UNPRUNED w1.0 graft losing to yolo11n on GAP8, and left "the graft is
unpruned; pruning-as-search is the lever that closed the Orin gap" as the standing caveat. This
closes that caveat without waiting for a recovery run, because **cycles do not depend on weight
values, and the pruned SHAPE does not either**:

``prune.recover_graft`` prunes with ``pruning_ratio_dict`` from the spec and deliberately does
NOT pass ``global_pruning``, so the per-stage channel counts are pinned by the spec — importance
only chooses *which* channels to drop, never how many. Its own comment says it: "shapes (and
latency) are importance-invariant while taylor picks better channels", and the torch-pruning
parity check measured a global_taylor ONNX's activation bytes equal to the l2 screen's to 0.00 %.

So an ``l2`` prune here (data-free — no gradients, no dataset) yields the SAME architecture, and
therefore the same cycle count, as the ``global_taylor`` champion training on the AGX. Only the
accuracy would differ, and this probe does not measure accuracy.

Run under ``.venv-nas`` (needs ofa + ultralytics + torch_pruning), as a module::

    python -m mcu.export_pruned --spec prune/specs/v2_act292.json --imgsz 160 \\
        --out models/res160/graft_r292_160_mcu.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DONOR = REPO / "runs/pose/experiments/gate_baseline/weights/best.pt"
DEFAULT_ARCH = REPO / "models/res224/graft_noneck_224_mcu.meta.json"


def load_arch(path: Path) -> dict:
    """Winner-v1 arch, read from an export meta sidecar ({'arch': {...}}) or a bare arch dict."""
    data = json.loads(Path(path).read_text())
    arch = data.get("arch", data)
    if not {"ks", "e", "d"} <= set(arch):
        raise ValueError(f"{path}: not an arch dict (need ks/e/d), got keys {sorted(arch)}")
    return arch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True, help="prune/specs/*.json ratio spec")
    ap.add_argument("--arch-meta", type=Path, default=DEFAULT_ARCH,
                    help=f"export meta sidecar carrying the arch (default: {DEFAULT_ARCH})")
    ap.add_argument("--donor", type=Path, default=DEFAULT_DONOR,
                    help="gate-trained head donor (nc=1/8-kpt) for the graft's head")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    from detect.export_grafted_onnx import export_grafted_onnx
    from detect.pose_model import build_grafted_pose_model
    from prune.prune_baseline import TRACE_IMGSZ
    from prune.prune_graft import prune_graft
    from prune.recover_graft import spec_ratio_dict
    from supernet.sampler import load_supernet

    arch = load_arch(args.arch_meta)
    spec = json.loads(args.spec.read_text())

    sn = load_supernet()
    model = build_grafted_pose_model(arch, supernet=sn, head_weights=args.donor).eval()
    before = sum(p.numel() for p in model.parameters())

    # Same call shape as recover_graft's spec branch, with importance='l2' (its default):
    # data-free, and shape-identical to the AGX's global_taylor run (see the module docstring).
    prd, ignored = spec_ratio_dict(model, arch["d"], spec)
    report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                         ratio=float(spec["rest_ratio"]), pruning_ratio_dict=prd,
                         extra_ignored=ignored, importance="l2")

    after = sum(p.numel() for p in model.parameters())
    expect = spec.get("params_after")
    print(f"pruned {before:,} -> {after:,} params (spec params_after={expect:,})"
          if expect else f"pruned {before:,} -> {after:,} params")
    if expect and after != expect:
        # Not fatal — a mismatch means this shape is NOT the one the AGX is training, and the
        # cycle number would then describe a different architecture. Say so loudly.
        print(f"  WARNING: params != spec params_after ({after:,} vs {expect:,}) — this is a "
              f"DIFFERENT shape than the spec's; the cycle probe would not describe the "
              f"champion. Investigate before trusting the number.")

    path, meta = export_grafted_onnx(
        arch, args.out, prebuilt=model, imgsz=args.imgsz, opset=args.opset,
        mcu_act=True, raw_head=True,
        provenance={"spec": str(args.spec), "spec_params_after": expect,
                    "params_before_prune": before, "prune_report": report,
                    "importance": "l2 (shape-identical to global_taylor: per-stage counts are "
                                  "pinned by the spec; recover_graft passes no global_pruning)"},
    )
    print(f"exported pruned graft (params {meta['params']:,}, mcu_act={meta['mcu_act']}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
