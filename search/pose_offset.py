"""Stage 0 — derive the measured pose stem/adapter/head offset from two Nano bench rows.

``offset = e2e − backbone``: the ChannelAdapter + Pose head (+ whatever cross-seam fusion TRT
finds at the graft boundary). The OFA stem rides **inside** the backbone measurement —
``arch_to_blocks`` excludes ``first_conv`` while ``PoseBackbone`` includes it — so this offset
is exactly "everything the LUT sum never covered, minus what the backbone measurement already
absorbs". The record's top-level fields are the ``CostOffset`` shape
``search.cost.load_stem_head_offset`` reads, so absolute end-to-end cost becomes::

    offset = load_stem_head_offset(Path("data/pose_stem_head_offset.json"))
    cost(arch, lut, res=640, stem_head=offset)

It also emits ``backbone_measured_vs_lut_sum`` — the first @640 additivity/calibration data
point (the @224 fit in ``data/latency_calibration.json``, slope 0.934, was never validated at
the deploy resolution).

Guards: both rows must share precision, ``power_mode`` and ``clocks_locked=True`` (a
cross-regime subtraction is meaningless — see the 612 MHz-vs-Super clock trap), and the delta
must be positive. ``peak_mem_mib`` is the **e2e whole-model working set** — safe for
``cost()``'s max-fold, but it is not "the head's own scratch".

Replaces nothing: ``data/stem_head_offset.json`` (the OFA *classifier* stem+head @224) stays
for the CP 2.2 record; this writes ``data/pose_stem_head_offset.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "pose_stem_head_offset.json"


def _check_same_regime(e2e_row: dict, backbone_row: dict) -> None:
    for key in ("precision", "power_mode"):
        a, b = e2e_row.get(key), backbone_row.get(key)
        if a != b:
            raise ValueError(f"rows disagree on {key!r}: e2e={a!r} vs backbone={b!r} — "
                             "offset across regimes is meaningless, re-bench in one session")
    for name, row in (("e2e", e2e_row), ("backbone", backbone_row)):
        if not row.get("clocks_locked"):
            raise ValueError(f"{name} row has clocks_locked={row.get('clocks_locked')!r} — "
                             "refuse to derive an offset from an unlocked-clock measurement")


def pose_offset_record(
    e2e_row: dict,
    backbone_row: dict,
    *,
    offset_params: int = 0,
    offset_flops: int = 0,
    lut_summed_ms: float | None = None,
) -> dict:
    """Build the pose stem/adapter/head CostOffset record from the two measured rows (pure).

    ``offset_params`` / ``offset_flops`` are the e2e−backbone differences from the export meta
    sidecars (the adapter + Pose head's own weights/compute).
    """
    _check_same_regime(e2e_row, backbone_row)
    e2e_ms = float(e2e_row["latency_ms"]["mean"])
    backbone_ms = float(backbone_row["latency_ms"]["mean"])
    delta = e2e_ms - backbone_ms
    if delta <= 0:
        raise ValueError(
            f"e2e ({e2e_ms:.4g} ms) <= backbone ({backbone_ms:.4g} ms) — measurement "
            "inconsistency (same session? same arch?); refusing to write a negative offset")

    record: dict = {
        # --- CostOffset fields (search.cost.load_stem_head_offset reads exactly these) ---
        "latency_ms": delta,
        "peak_mem_mib": float(e2e_row["peak_mem_mib"]),
        "params": int(offset_params),
        "flops": int(offset_flops),
        # --- provenance ---
        "precision": e2e_row.get("precision"),
        "power_mode": e2e_row.get("power_mode"),
        "trt_version": e2e_row.get("trt_version"),
        "note": ("pose adapter + Pose head offset = e2e - backbone (measured, one session). "
                 "The OFA stem is INSIDE the backbone measurement (arch_to_blocks excludes "
                 "first_conv; PoseBackbone includes it). peak_mem_mib is the e2e whole-model "
                 "working set (max-fold-safe), not the head's own scratch."),
        "components": {
            "e2e": {"name": e2e_row.get("name"), "latency_ms": e2e_ms,
                    "peak_mem_mib": e2e_row.get("peak_mem_mib")},
            "backbone": {"name": backbone_row.get("name"), "latency_ms": backbone_ms,
                         "peak_mem_mib": backbone_row.get("peak_mem_mib")},
        },
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if lut_summed_ms is not None:
        # The first @640 additivity point: measured whole-backbone vs the raw LUT block sum.
        record["backbone_measured_vs_lut_sum"] = {
            "lut_summed_ms": float(lut_summed_ms),
            "backbone_measured_ms": backbone_ms,
            "ratio": backbone_ms / float(lut_summed_ms),
            "delta_pct": 100.0 * (backbone_ms - float(lut_summed_ms)) / float(lut_summed_ms),
            "note": ("ratio < 1 ⇒ TRT cross-seam fusion discount at 640 (the @224 fit was "
                     "0.934); the backbone row also contains the stem, which the LUT sum "
                     "excludes — the two effects are entangled here"),
        }
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e2e", type=Path, required=True, help="bench row JSON of the grafted e2e")
    ap.add_argument("--backbone", type=Path, required=True,
                    help="bench row JSON of the backbone-only export (same arch, same session)")
    ap.add_argument("--e2e-meta", type=Path, default=None,
                    help="export .meta.json of the e2e ONNX (params/flops)")
    ap.add_argument("--backbone-meta", type=Path, default=None,
                    help="export .meta.json of the backbone ONNX (params/flops)")
    ap.add_argument("--lut-summed-ms", type=float, default=None,
                    help="the arch's raw @640 LUT block sum (default: winner.json latency_ms)")
    ap.add_argument("--winner", type=Path, default=ROOT / "state" / "winner_v1" / "winner.json",
                    help="fallback source for --lut-summed-ms")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    e2e_row = json.loads(args.e2e.read_text())
    backbone_row = json.loads(args.backbone.read_text())

    params = flops = 0
    if args.e2e_meta and args.backbone_meta:
        me, mb = json.loads(args.e2e_meta.read_text()), json.loads(args.backbone_meta.read_text())
        params = int(me["params"]) - int(mb["params"])
        if me.get("flops") is not None and mb.get("flops") is not None:
            flops = int(me["flops"]) - int(mb["flops"])

    lut_summed = args.lut_summed_ms
    if lut_summed is None and args.winner.exists():
        lut_summed = float(json.loads(args.winner.read_text())["latency_ms"])

    record = pose_offset_record(e2e_row, backbone_row, offset_params=params,
                                offset_flops=flops, lut_summed_ms=lut_summed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        print(f"[pose-offset] replacing existing {args.out} (derived artifact, regenerable)")
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"pose offset (adapter+head) = {record['latency_ms']:.4g} ms  "
          f"[e2e {record['components']['e2e']['latency_ms']:.4g} - backbone "
          f"{record['components']['backbone']['latency_ms']:.4g}]  -> {args.out}")
    if "backbone_measured_vs_lut_sum" in record:
        b = record["backbone_measured_vs_lut_sum"]
        print(f"@640 additivity point: measured/summed = {b['ratio']:.3f} "
              f"({b['delta_pct']:+.1f}% vs raw LUT sum {b['lut_summed_ms']:.4g} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
