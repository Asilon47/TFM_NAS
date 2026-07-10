"""Export a grafted (OFA backbone → adapter → Pose head) model — or its backbone alone — to ONNX.

Stage 0 of the plan pivot (procedure.md "Plan pivot"): winner-v1's only latency so far is the
**backbone-blocks-only LUT sum** (11.208 ms) while the baseline's 12.755 ms is a full-network
measurement. These ONNX files are what ``lut.orchestrate.bench_model`` turns into TRT engines on
the Orin Nano to measure the graft **end-to-end** — and the backbone alone, so
``search.pose_offset`` can derive ``data/pose_stem_head_offset.json`` (= e2e − backbone,
isolating adapter + Pose head; the OFA stem rides inside the backbone measurement).

Arch selection (exactly one source):

* ``--winner PATH`` — a winner record with an ``"arch"`` key (default
  ``state/winner_v1/winner.json``).
* ``--candidates PATH --index N`` — a pinned de-noise candidate. ⚠ Select by **index**, never
  by depth: ``d=[2,2,4,3,2]`` appears twice in the top-12 (index 7 @ 11.097 ms vs index 11 @
  10.727 ms). The export prints the candidate's cached latency — eyeball it against the table.

Export details: ``head.export = True; head.format = "onnx"`` before tracing, so the Pose head
emits the single decoded ``(1, 4+nc+3·nkpt, 8400)`` deploy tensor — the benched graph is the
deployed one. ``torch.onnx.export(..., dynamo=False)`` where the parameter exists: torch ≥ 2.9
defaults to the dynamo exporter, which needs onnxscript (absent by design — see the venv-drift
memory) and produces a different graph; the legacy TorchScript path matches how every LUT block
was exported (``lut/export/to_onnx.py``). Weights are irrelevant to latency (same graph), so no
donor / fine-tuned weights are loaded. BN stays unfused — TensorRT folds it at engine build,
matching how the per-block LUT graphs carry BN too.

Run under ``.venv-nas`` (needs ofa + ultralytics); smoke-load the results under ``.venv``
(onnxruntime lives there)::

    python -m detect.export_grafted_onnx --out data/e2e/winner_v1_e2e_640.onnx
    python -m detect.export_grafted_onnx --backbone-only \
        --out data/e2e/winner_v1_backbone_640.onnx
    python -m detect.export_grafted_onnx \
        --candidates state/winner_v1/denoise_candidates.json --index 11 \
        --out data/e2e/fallback_idx11_e2e_640.onnx
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINNER = ROOT / "state" / "winner_v1" / "winner.json"


def load_arch(
    winner: Path | None = None,
    candidates: Path | None = None,
    index: int | None = None,
) -> tuple[dict, dict]:
    """Resolve the arch to export + its provenance from exactly one source (pure, tested).

    Returns ``(arch_dict, provenance)`` where provenance records the source path and, for a
    candidate, its index plus the cached ``latency_ms``/``acc`` so the operator can verify the
    pick against the de-noise table (the duplicate-``d`` trap).
    """
    if (winner is None) == (candidates is None):
        raise ValueError("pick exactly one arch source: --winner OR --candidates")
    if winner is not None:
        record = json.loads(Path(winner).read_text())
        if "arch" not in record:
            raise ValueError(f"{winner} has no 'arch' key — not a winner record")
        return record["arch"], {"source": str(winner), "d": record["arch"]["d"]}
    if index is None:
        raise ValueError("--candidates needs --index (d values repeat; index is the identity)")
    payload = json.loads(Path(candidates).read_text())  # type: ignore[arg-type]
    cands = payload["candidates"]
    if not 0 <= index < len(cands):
        raise ValueError(f"--index {index} out of range [0, {len(cands)})")
    cand = cands[index]
    return cand["arch"], {
        "source": str(candidates), "index": index, "d": cand["arch"]["d"],
        "cached_latency_ms": cand.get("latency_ms"), "cached_acc": cand.get("acc"),
        "method": cand.get("method"), "seed": cand.get("seed"),
    }


def _legacy_export(module: Any, dummy: Any, out: Path, *, opset: int,
                   output_names: list[str]) -> None:
    """torch.onnx.export forced onto the legacy TorchScript path where the knob exists."""
    import torch

    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(
        module, dummy, str(out), opset_version=opset, do_constant_folding=True,
        input_names=["images"], output_names=output_names, dynamic_axes=None, **kwargs)


def export_grafted_onnx(
    arch: dict,
    out: Path,
    *,
    backbone_only: bool = False,
    neck: str | None = None,
    imgsz: int = 640,
    opset: int = 17,
    supernet: Any = None,
    provenance: dict | None = None,
    prebuilt: Any = None,
) -> tuple[Path, dict]:
    """Build the model for ``arch`` and export it (static ``1×3×imgsz×imgsz``, batch 1).

    Writes a ``<out>.meta.json`` sidecar with params / FLOPs / arch / provenance so the Stage-0
    offset derivation (``search.pose_offset``) can compute the adapter+head params/flops as the
    e2e−backbone difference. Returns ``(onnx_path, meta)``.

    ``neck`` (CP 5.3): export the V2 (``"topdown"``) / V3 (``"pan"``) graph. The zero-init
    gates are **set to 1.0 before export** — a gate that is exactly 0 is an initializer
    constant, and TRT/onnx constant folding would elide the whole fusion path
    (``x + 0·conv(...)`` → ``x``), silently benching a neck-less graph. Gate=1 measures the
    latency of the *live* neck, which is what a trained (gates-open) deployment pays.

    ``prebuilt`` (prune screen): export an already-constructed end-to-end graft as-is — e.g. a
    structurally-pruned model from :func:`prune.prune_graft.prune_graft` — skipping the build.
    ``arch`` is still recorded in the sidecar for provenance; params/FLOPs come from the given
    module, so they reflect the pruned counts. Mutually exclusive with ``backbone_only``/``neck``.
    """
    import torch

    from catalog.flops import count_flops_forward
    from supernet.pose_backbone import PoseBackbone
    from supernet.sampler import load_supernet, sample

    if backbone_only and neck is not None:
        raise ValueError("--backbone-only and --neck are mutually exclusive (a neck sits "
                         "between adapter and head — there is none in a backbone-only export)")

    if prebuilt is not None:
        if backbone_only or neck is not None:
            raise ValueError("prebuilt= exports an already-built end-to-end graft as-is; "
                             "backbone_only/neck would rebuild it")
        module: Any = prebuilt.eval()
    elif backbone_only:
        sn = supernet if supernet is not None else load_supernet()
        module = PoseBackbone(sample(arch, sn), arch["d"]).eval()
    else:
        sn = supernet if supernet is not None else load_supernet()
        from detect.pose_model import build_grafted_pose_model

        module = build_grafted_pose_model(arch, supernet=sn, neck=neck).eval()
        if neck is not None:
            neck_module = module.model[2]  # (backbone, adapter, neck, head)
            with torch.no_grad():
                for pname, param in neck_module.named_parameters():
                    if pname.startswith("g"):
                        param.fill_(1.0)  # live fusion paths — see the docstring
    if backbone_only:
        output_names = ["p3", "p4", "p5"]
    else:
        head = module.model[-1]
        head.export = True        # deploy graph: the single decoded (1, 4+nc+3*nkpt, 8400)
        head.format = "onnx"
        output_names = ["output0"]

    dummy = torch.randn(1, 3, imgsz, imgsz)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _legacy_export(module, dummy, out, opset=opset, output_names=output_names)

    try:
        flops, _ = count_flops_forward(module, (1, 3, imgsz, imgsz))
    except Exception as e:  # provenance only — never fail an export over the FLOPs count
        print(f"[meta] FLOPs count failed ({e!r}); recording null")
        flops = None
    meta = {
        "arch": arch,
        "provenance": provenance or {},
        "backbone_only": backbone_only,
        "neck": neck,
        "imgsz": imgsz,
        "opset": opset,
        "params": sum(p.numel() for p in module.parameters()),
        "flops": flops,
        "onnx": out.name,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--winner", type=Path, default=None,
                     help=f"winner record with an 'arch' key (default: {DEFAULT_WINNER})")
    src.add_argument("--candidates", type=Path, default=None,
                     help="pinned de-noise candidate set (needs --index)")
    ap.add_argument("--index", type=int, default=None,
                    help="candidate index — d values repeat, the index is the identity")
    ap.add_argument("--backbone-only", action="store_true",
                    help="export PoseBackbone alone (P3/P4/P5 outs) for the offset derivation")
    ap.add_argument("--neck", choices=["topdown", "pan"], default=None,
                    help="CP 5.3: export the V2/V3 necked graph (gates forced to 1.0 so "
                         "constant folding cannot elide the fusion paths)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    winner = args.winner if (args.winner or args.candidates) else DEFAULT_WINNER
    arch, prov = load_arch(winner=winner, candidates=args.candidates, index=args.index)
    print(f"arch: d={arch['d']}  provenance={prov}")
    path, meta = export_grafted_onnx(
        arch, args.out, backbone_only=args.backbone_only, neck=args.neck, imgsz=args.imgsz,
        opset=args.opset, provenance=prov)
    kind = ("backbone-only (P3/P4/P5)" if args.backbone_only
            else f"end-to-end graft (neck={args.neck})" if args.neck else "end-to-end graft")
    print(f"exported {kind} -> {path}  (params {meta['params']:,}, flops {meta['flops']}, "
          f"opset {meta['opset']}, static {args.imgsz})")
    print(f"meta sidecar -> {path.with_suffix('.meta.json')}")
    print(f"next (laptop .venv): python -m lut.orchestrate.bench_model --onnx {path} "
          f"--imgsz {args.imgsz} --precision fp32 --out data/e2e/{path.stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
