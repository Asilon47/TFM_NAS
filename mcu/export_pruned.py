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
from typing import Any

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


def build_pruned_graft(arch: dict, spec: dict, donor: Path,
                       neck: str | None = None) -> tuple[Any, dict]:
    """The spec-pruned graft, data-free (``l2``) — shapes pinned by the spec, weights untrained.

    Shape-identical to the AGX's ``global_taylor`` run (see the module docstring), which is what
    lets a *trained* pruned ``state_dict`` load straight into it: importance chooses which
    channels survive, never how many, so the tensors line up exactly. Callers that load trained
    weights overwrite every parameter anyway — for them this is purely a shape scaffold.

    ``neck`` must match the run whose weights are being loaded (``recover_graft --neck``), or the
    state_dict will not fit. The neck's convs are not named by the spec, so they fall under
    ``rest_ratio`` like any other unlisted layer.
    """
    import torch

    from detect.pose_model import build_grafted_pose_model
    from prune.prune_baseline import TRACE_IMGSZ
    from prune.prune_graft import prune_graft
    from prune.recover_graft import spec_ratio_dict
    from supernet.sampler import load_supernet

    sn = load_supernet()
    model = build_grafted_pose_model(arch, supernet=sn, head_weights=donor, neck=neck).eval()
    before = sum(p.numel() for p in model.parameters())
    prd, ignored = spec_ratio_dict(model, arch["d"], spec)
    report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                         ratio=float(spec["rest_ratio"]), pruning_ratio_dict=prd,
                         extra_ignored=ignored, importance="l2")
    after = sum(p.numel() for p in model.parameters())
    return model, {"params_before": before, "params_after": after, "prune_report": report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True, help="prune/specs/*.json ratio spec")
    ap.add_argument("--arch-meta", type=Path, default=DEFAULT_ARCH,
                    help=f"export meta sidecar carrying the arch (default: {DEFAULT_ARCH})")
    ap.add_argument("--donor", type=Path, default=DEFAULT_DONOR,
                    help="gate-trained head donor (nc=1/8-kpt) for the graft's head")
    ap.add_argument("--neck", choices=["topdown", "pan"], default=None,
                    help="must MATCH the recover_graft --neck of the weights this shape is "
                         "for; changes the graph, so it changes the cycle count")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from detect.export_grafted_onnx import export_grafted_onnx

    arch = load_arch(args.arch_meta)
    spec = json.loads(args.spec.read_text())

    model, build = build_pruned_graft(arch, spec, args.donor, neck=args.neck)
    before, after, report = build["params_before"], build["params_after"], build["prune_report"]
    expect = spec.get("params_after")
    print(f"pruned {before:,} -> {after:,} params (spec params_after={expect:,})"
          if expect else f"pruned {before:,} -> {after:,} params")
    if expect and after != expect and args.neck is None:
        # Not fatal — a mismatch means this shape is NOT the one the AGX is training, and the
        # cycle number would then describe a different architecture. Say so loudly.
        print(f"  WARNING: params != spec params_after ({after:,} vs {expect:,}) — this is a "
              f"DIFFERENT shape than the spec's; the cycle probe would not describe the "
              f"champion. Investigate before trusting the number.")
    elif expect and args.neck:
        # Expected: the spec's params_after is a neck-less count. Report the neck's real cost —
        # it is exactly what decides whether a necked graft can still win the Pareto.
        print(f"  neck={args.neck} adds {after - expect:+,} params over the spec's neck-less "
              f"{expect:,} ({100 * (after - expect) / expect:+.1f}%) — price its cycles before "
              f"reading any accuracy win as a Pareto win.")

    if args.neck:
        # Fold the scalar edge gates into their convs (exact: bare Conv2d + linear resample).
        # Without this, NNTool's expression_matcher fuses the gated mul+add into a 3-arg
        # expression AutoTiler's AddNode template rejects (measured 2026-07-19, S260_Op_expr_30)
        # — and turning the matcher off breaks the h-swish Muls instead.
        import torch

        from detect.neck import ZeroGatedTopDownNeck

        for m in model.modules():
            if isinstance(m, ZeroGatedTopDownNeck):
                if all(v == 0.0 for v in m.gate_values().values()):
                    # Data-free probe build: gates sit at their ReZero init. Folding 0 would
                    # zero the lateral convs and let NNTool degenerate them — under-pricing
                    # the neck. Price the OPEN-gate deploy graph (what a trained net ships).
                    with torch.no_grad():
                        for g in (p for n, p in m.named_parameters() if n.startswith("g")):
                            g.fill_(1.0)
                    print("[mcu-neck] probe build (all gates 0) -> forced open (1.0) so the "
                          "cycle probe prices the trained deploy structure")
                print(f"[mcu-neck] folded gates {m.fold_gates_()} into the edge convs")

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
