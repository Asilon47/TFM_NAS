"""Prune-the-graft latency screen — "can pruning drag a searched OFA subnet under the baseline?"

User question (2026-07-10): the OFA graft loses end-to-end on the Nano — 17.69 ms fp32 / 12.38 ms
fp16 vs the yolo11n-pose baseline 12.75 / 7.75 — because it is memory-bound (the winner fp16 run
achieves 0.48 GB/s vs the baseline's 0.88). Pruning removes channels, cutting activation memory
traffic, so it *could* recover latency. This screens the graft's LATENCY across a pruning ladder
WITHOUT any recovery training: TensorRT latency is weight-value-independent, so the channel COUNT
sets latency, not which channels survive. Only if a rung clears the baseline do we spend a GPU
session on recovery (CP 6.2) to see whether accuracy survives — the same measure-the-floor-first
discipline as the OFA-ResNet50 screen (``expand/screen_r50.py``).

Budget (measured: ``data/pose_stem_head_offset.json``, ``data/e2e/*``): to beat baseline fp32 e2e
the backbone must drop 13.85 -> <8.92 ms (a 36 % cut) with the head fixed at 3.84 ms; fp16 needs
an even deeper backbone cut (~48 %) because the ~2.7 ms head floor is a larger share of the
7.75 ms target. The already-measured dense-net prune curve (``prune_base_r*``: 45 % params -> only
30 % latency) shows even a compute-bound net prunes latency sub-linearly, so the memory-bound
graft is expected to need aggressive pruning; the Nano bench settles whether it clears the bar at
any survivable sparsity.

Run (``.venv-nas``; CPU export is fine)::

    python -m prune.screen_prune_graft --out-dir models/screen_prune_graft

then bench each ONNX on the Nano (mode 0 / 612 MHz, clocks locked, one at a time)::

    python -m lut.orchestrate.bench_model \\
        --onnx models/screen_prune_graft/graft_prune_r40_e2e_640.onnx \\
        --imgsz 640 --precision fp16 --out data/e2e/graft_prune_r40_e2e_640_fp16.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured budget (fp32) — data/pose_stem_head_offset.json + data/e2e/baseline_recheck_640.json.
BACKBONE_MS_FP32 = 13.852
HEAD_OFFSET_MS_FP32 = 3.837
BASELINE_MS_FP32 = 12.755
BASELINE_MS_FP16 = 7.752
GRAFT_E2E_MS_FP16 = 12.382

# Bracket the ~45-50 % sparsity where fp16 parity is plausible: one mild, one near-threshold, one
# well past it (accuracy-doomed, but it exposes the measured latency floor).
RATIOS: tuple[float, ...] = (0.2, 0.4, 0.6)


def optimistic_backbone_ms(base_ms: float, channel_ratio: float) -> float:
    """Best-case memory-bound latency: perfectly linear in the CHANNEL pruning ratio.

    A memory-bound layer's time ~ bytes moved ~ H*W*C, so pruning a fraction ``channel_ratio`` of
    the channels removes ~that fraction of the time. This is the CHANNEL ratio, NOT param sparsity:
    params scale ~width^2 (a conv is in_c*out_c*k^2), so a 0.2 channel prune is ~0.36 param
    sparsity — but that width^2 saving does NOT translate to latency for activation-bound layers,
    so using param sparsity here would overstate the benefit. Even this channel-linear figure is an
    UPPER bound: the stem (few channels, high resolution) and the protected head output convs do
    not prune, and ``round_to=16`` leaves narrow layers uncut — so the MEASURED floor is higher.
    Bracket only; the Nano bench is the truth.
    """
    return base_ms * max(0.0, 1.0 - channel_ratio)


def onnx_name(ratio: float, imgsz: int) -> str:
    """Deterministic per-rung ONNX filename (``graft_prune_r40_e2e_640.onnx``)."""
    return f"graft_prune_r{int(round(ratio * 100)):02d}_e2e_{imgsz}.onnx"


def main(argv: list[str] | None = None) -> int:
    import torch

    from detect.export_grafted_onnx import DEFAULT_WINNER, export_grafted_onnx, load_arch
    from detect.pose_model import build_grafted_pose_model
    from prune.prune_graft import prune_graft
    from supernet.sampler import load_supernet

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--winner", type=Path, default=None,
                    help=f"winner record with an 'arch' key (default: {DEFAULT_WINNER})")
    ap.add_argument("--out-dir", type=Path, default=Path("models/screen_prune_graft"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ratios", default=",".join(str(r) for r in RATIOS),
                    help="comma list of pruning ratios in (0, 1)")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    winner = a.winner or DEFAULT_WINNER
    arch, prov = load_arch(winner=winner)
    ratios = [float(x) for x in a.ratios.split(",")]
    print(f"arch: d={arch['d']}  winner={winner}")

    sn = load_supernet()
    example = torch.randn(1, 3, a.imgsz, a.imgsz)
    rows: list[dict] = []
    for r in ratios:
        model = build_grafted_pose_model(arch, supernet=sn).eval()  # fresh: prune is in-place
        report = prune_graft(model, example, ratio=r)
        onnx = a.out_dir / onnx_name(r, a.imgsz)
        _, meta = export_grafted_onnx(
            arch, onnx, imgsz=a.imgsz, prebuilt=model,
            provenance={**prov, "prune_ratio": r,
                        "achieved_sparsity": report["params_sparsity"]})
        sparsity = report["params_sparsity"]
        # Memory-bound latency tracks the CHANNEL ratio r (activation traffic), not param
        # sparsity (~1-(1-r)^2, a width^2 effect that does not translate to memory-bound time).
        opt_e2e = optimistic_backbone_ms(BACKBONE_MS_FP32, r) + HEAD_OFFSET_MS_FP32
        rows.append({
            "channel_ratio": r,
            "achieved_param_sparsity": round(sparsity, 4),
            "params_before": report["params_before"],
            "params_after": report["params_after"],
            "flops": meta["flops"],
            "all_rounded": report["all_rounded"],
            "onnx": str(onnx),
            "optimistic_e2e_ms_fp32": round(opt_e2e, 2),
            "beats_baseline_fp32_even_optimistic": opt_e2e < BASELINE_MS_FP32,
        })
        print(f"  r={r:.2f} (channels)  param_sparsity={sparsity:.3f}  "
              f"params {report['params_before']:,}->{report['params_after']:,}  "
              f"opt_e2e(mem-bound)~{opt_e2e:.2f}ms  -> {onnx.name}")

    backbone_cut_needed = 1.0 - (BASELINE_MS_FP32 - HEAD_OFFSET_MS_FP32) / BACKBONE_MS_FP32
    manifest = {
        "winner": str(winner),
        "arch_d": arch["d"],
        "imgsz": a.imgsz,
        "ratios": ratios,
        "budget": {
            "baseline_ms_fp32": BASELINE_MS_FP32,
            "backbone_ms_fp32": BACKBONE_MS_FP32,
            "head_offset_ms_fp32": HEAD_OFFSET_MS_FP32,
            "backbone_cut_needed_fp32_pct": round(100 * backbone_cut_needed, 1),
            "baseline_ms_fp16": BASELINE_MS_FP16,
            "graft_e2e_ms_fp16": GRAFT_E2E_MS_FP16,
        },
        "rows": rows,
        "note": ("LATENCY IS MEASURED-ONLY: bench each onnx on the Nano (mode 0, locked clocks) "
                 "-> data/e2e/. channel_ratio is torch-pruning's pruning_ratio (fraction of "
                 "channels removed); achieved_param_sparsity is higher (~1-(1-r)^2, width^2). "
                 "optimistic_e2e = backbone*(1-channel_ratio) + fixed head, a best-case "
                 "memory-bound bound; the measured floor is higher (unprunable stem + round_to=16 "
                 "+ protected head). No recovery training here (TRT latency is weight-value-"
                 "independent); accuracy is only worth checking (CP 6.2) if a rung clears the "
                 "bar."),
    }
    (a.out_dir / "screen_prune_graft.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {a.out_dir / 'screen_prune_graft.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
