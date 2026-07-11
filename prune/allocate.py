"""CP 6.4 HALP-lite — latency-driven per-stage sparsity allocation for the winner graft.

HALP (NeurIPS'22 "Structural Pruning via Latency-Saliency Knapsack") allocates pruning where
the LATENCY table says it pays, not uniformly. This is the per-STAGE lite version wired to this
repo's own hardware model: stage base latencies come from the **@640 per-block LUT rows** (the
same rows that drove the stage-1 topology search), scaled by the measured backbone calibration
(LUT-sum × 1.236 = measured backbone, Stage 0), plus the measured pose adapter+head share
(3.84 ms fp32). Saliency = conv out-channel group-L2 mass, the data-free importance the ladder
already uses; the knapsack greedily takes the (stage, next-rung) step with the best
ms-saved-per-saliency-lost until the fp32 target is met.

Scope guards (why per-stage, why fp32): per-layer allocation needs per-layer TRT latencies we
do not have; fp32 is the calibrated axis (fp16 ≈ 0.70 × fp32 for this family, reported as an
estimate only — the deploy claim is verified on-device). Stage latencies are modeled linear in
retained channels (memory-bound family; the screen's e2e beat linear, so predictions are
conservative). The emitted spec is consumed by ``prune.recover_graft --ratio-spec``.

Run (laptop ``.venv-nas``; deterministic — OFA init + warm gate head)::

    python -m prune.allocate --target-fp32-ms 10.4 --target-fp32-ms 9.0
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "prune" / "specs"

# Measured anchors (Stage 0 + prune-graft screen, models/screen_prune_graft/ + procedure.md):
# winner e2e fp32 17.688 = LUT-sum 11.208 × 1.236 (measured backbone 13.85) + head 3.84.
BACKBONE_CALIBRATION = 1.236          # measured backbone / LUT-sum @640 (Stage 0)
HEAD_ADAPTER_MS_FP32 = 3.837          # data/pose_stem_head_offset.json (adapter + Pose head)
FP16_OVER_FP32 = 0.700                # graft-family measured pairs: 12.38/17.69, 8.41/11.81
STAGE_RUNGS = tuple(round(0.1 * i, 1) for i in range(8))   # 0.0 … 0.7 channel ratio


def stage_lut_ms(arch: Any, lut_path: Any = None, precision: str = "fp32") -> list[float]:
    """Per-stage LUT latency @640 (first block folded into stage 0), CALIBRATED ms."""
    from lut.loader import load_lut
    from search.arch_to_blocks import arch_to_keys

    lut_path = (ROOT / "data" / "lut.jsonl") if lut_path is None else lut_path
    lut = load_lut(lut_path, precision=precision)
    keys = arch_to_keys(arch, res=640)  # ArchDict-shaped (winner.json "arch")
    per_block = []
    for k in keys:
        if k not in lut:
            raise KeyError(f"no @640 LUT row for key {k}")
        per_block.append(float(lut[k]["latency_ms"]["mean"]))
    d = arch["d"]
    stages = []
    i = 1  # blocks[0] = the first block → stage 0
    for s, ds in enumerate(d):
        ms = sum(per_block[i:i + ds])
        if s == 0:
            ms += per_block[0]
        stages.append(ms * BACKBONE_CALIBRATION)
        i += ds
    return stages


def stage_saliency_curves(model: Any, depths: list[int],
                          rungs: tuple[float, ...] = STAGE_RUNGS) -> list[list[float]]:
    """Per-stage lost group-L2 mass at each rung (+ the adapter/head 'rest' pseudo-stage).

    Saliency of one out-channel = L2 norm of its conv-weight slice; lost(r) = sum of the
    smallest ceil(r·n) norms in the stage. Raw (un-normalized) sums keep stages comparable on
    one scale, exactly like global_l2 ranking.
    """
    import torch
    from torch import nn

    backbone = model.model[0]
    blocks = list(backbone.blocks)
    groups: list[list[Any]] = []
    i = 1
    for ds in depths:
        g = blocks[i:i + ds]
        if i == 1:
            g = [blocks[0], *g]
        groups.append(g)
        i += ds
    groups.append([m for m in (model.model[1], model.model[-1])])   # rest: adapter + head

    curves = []
    for g in groups:
        norms: list[float] = []
        for mod in g:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    w = m.weight.detach()
                    norms.extend(torch.linalg.vector_norm(
                        w.reshape(w.shape[0], -1), dim=1).tolist())
        norms.sort()
        n = len(norms)
        curve = []
        for r in rungs:
            k = int(round(r * n))
            curve.append(float(sum(norms[:k])))
        curves.append(curve)
    return curves


def greedy_allocate(stage_ms: list[float], saliency: list[list[float]],
                    target_ms: float, *, head_ms: float = HEAD_ADAPTER_MS_FP32,
                    rungs: tuple[float, ...] = STAGE_RUNGS) -> dict:
    """The knapsack: raise one stage a rung at a time, best Δms-saved / Δsaliency-lost first.

    ``saliency`` has one curve per stage PLUS the rest (adapter+head) pseudo-stage last.
    Latency model: linear in retained channels per stage; rest scales ``head_ms``.
    """
    if len(saliency) != len(stage_ms) + 1:
        raise ValueError("saliency needs one curve per stage + the rest pseudo-stage")
    base = list(stage_ms) + [head_ms]
    level = [0] * len(base)

    def predicted() -> float:
        return sum(b * (1.0 - rungs[lv]) for b, lv in zip(base, level, strict=True))

    while predicted() > target_ms:
        best, best_gain = None, -1.0
        for s in range(len(base)):
            if level[s] + 1 >= len(rungs):
                continue
            dms = base[s] * (rungs[level[s] + 1] - rungs[level[s]])
            dsal = saliency[s][level[s] + 1] - saliency[s][level[s]]
            gain = dms / max(dsal, 1e-12)
            if gain > best_gain:
                best, best_gain = s, gain
        if best is None:
            raise ValueError(f"target {target_ms} ms infeasible even at max rung "
                             f"{rungs[-1]} everywhere (floor {predicted():.2f} ms)")
        level[best] += 1

    pred = predicted()
    return {
        "stage_ratios": [rungs[lv] for lv in level[:-1]],
        "rest_ratio": rungs[level[-1]],
        "predicted_fp32_ms": round(pred, 3),
        "fp16_estimate_ms": round(pred * FP16_OVER_FP32, 3),
        "target_fp32_ms": target_ms,
        "rungs": list(rungs),
        "stage_base_ms": [round(b, 3) for b in stage_ms],
        "head_adapter_ms": head_ms,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-fp32-ms", type=float, action="append", required=True,
                   help="repeatable; fp16 deploy bar 7.75/6.6 ≈ fp32 10.4/9.0 for this family")
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--head-weights", type=Path,
                   default=ROOT / "runs/pose/experiments/gate_baseline/weights/best.pt")
    p.add_argument("--lut", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=SPEC_DIR)
    a = p.parse_args(argv)

    from detect.pose_model import build_grafted_pose_model
    from eval.shortft import _seed_everything
    from eval.verify_winner import load_winner

    arch = load_winner(a.winner_dir)["arch"]
    _seed_everything(0)  # adapter init → deterministic saliency
    model = build_grafted_pose_model(arch, head_weights=a.head_weights, freeze_head=False)
    stage_ms = stage_lut_ms(arch, lut_path=a.lut)
    curves = stage_saliency_curves(model, arch["d"])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for t in a.target_fp32_ms:
        spec = greedy_allocate(stage_ms, curves, t)
        spec["arch_tag"] = "winner"
        spec["saliency"] = "conv out-channel group-L2 (data-free)"
        spec["provenance"] = ("stage ms = @640 LUT rows × 1.236 measured calibration; "
                              "head/adapter = measured 3.837 ms fp32; linear-in-channels "
                              "(conservative — the screen's e2e beat linear)")
        out = a.out_dir / f"halp_fp32_{str(t).replace('.', 'p')}.json"
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"{out.name}: stages={spec['stage_ratios']} rest={spec['rest_ratio']} "
              f"pred={spec['predicted_fp32_ms']} ms (fp16≈{spec['fp16_estimate_ms']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
