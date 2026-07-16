"""Resolution screen: what does 640 -> N cost the pruned graft? (CPU, no AGX, no training)

The MCU leg stacks three accuracy drops on top of the measured pruning drop, and only one of them
decides the operating point:

    0.841 (winner-v1 donor)  ->  0.7625 (v2_act292 pruned+KD, MEASURED on the AGX @640 RGB)
                             ->  ??? @160        <- this screen: almost certainly the dominant drop
                             ->  ??? gray        <- small (the stem is 3 % of cycles)
                             ->  ??? int8 PTQ    <- known depthwise-MBv3 hazard, screened separately

CP 10.1 priced the *cycles* at 160 (25,541,274 = 6.85 FPS), and cycles do not depend on weights —
so the shape is settled and the only open question is what that shape costs in accuracy. This
answers it cheaply, before any AGX time is spent training at a resolution that might not be worth
training at.

**This is a LOWER BOUND, deliberately.** The checkpoint was trained at 640; evaluating it at 160 is
a train/test resolution mismatch, so a model actually *trained* at 160 can only do better. That
asymmetry is what makes it a useful screen: a passing lower bound is decisive (green-light the AGX),
and a collapsing one is decisive the other way (resolution is the wall — no amount of cycle work
helps, and that is itself the finding).

**The 640 control is not optional.** Rebuilding a pruned architecture and loading a trained
state_dict into it has several silent failure modes (wrong spec, wrong arch, wrong donor head). If
the 640 row does not reproduce the recorded 0.7625/0.7637, the load is wrong and every other row is
noise. The screen refuses to report without it.

Run under ``.venv-nas`` (needs ofa + ultralytics + torch_pruning), as a module::

    python -m mcu.res_screen --spec prune/specs/v2_act292.json \\
        --ckpt data/lightning_out/l4_v2_act292/recover_graft_v2_act292_kd.pt \\
        --expect-map 0.7637
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data/mcu/res_screen.json"
# The 640-trained reference resolutions, coarse->fine. 640 is the control; 160 is the CP 10.1 shape;
# 192/224 probe whether a *higher*-accuracy operating point is affordable (user bar: ~5 FPS is fine,
# so the question flipped from "how low can we go" to "can we afford to go up").
DEFAULT_RES = [640, 320, 224, 192, 160, 128, 96]
CONTROL_RES = 640
# The 640 control must land within this of the recorded number or the state_dict load is suspect.
# Wide enough to absorb validator-config noise (0.010 measured, CPU-vs-CUDA on the 140-img val).
CONTROL_TOL = 0.02


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True,
                    help="prune/specs/*.json — pins the shape")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="trained pruned-graft state_dict (.pt)")
    ap.add_argument("--arch-meta", type=Path,
                    default=REPO / "models/res224/graft_noneck_224_mcu.meta.json")
    ap.add_argument("--donor", type=Path,
                    default=REPO / "runs/pose/experiments/gate_baseline/weights/best.pt")
    ap.add_argument("--data-yaml", type=Path, default=REPO / "dataset/dataset.yaml")
    ap.add_argument("--res", type=int, nargs="+", default=DEFAULT_RES)
    ap.add_argument("--expect-map", type=float, default=None,
                    help="recorded mAP50-95 of --ckpt @640; the control row must match it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import torch

    from detect.evaluate import pose_map_model
    from mcu.export_pruned import build_pruned_graft, load_arch

    if CONTROL_RES not in args.res:
        raise SystemExit(f"--res must include the {CONTROL_RES} control (got {args.res}); "
                         f"without it a wrong state_dict load is indistinguishable from a cliff")

    arch = load_arch(args.arch_meta)
    spec = json.loads(args.spec.read_text())

    # Shape scaffold only — every weight below is overwritten by the trained state_dict.
    model, build = build_pruned_graft(arch, spec, args.donor)
    expect_params = spec.get("params_after")
    if expect_params and build["params_after"] != expect_params:
        raise SystemExit(f"rebuilt shape has {build['params_after']:,} params, spec says "
                         f"{expect_params:,} — the scaffold is not the trained architecture")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict does not match the rebuilt shape: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected (first missing: {list(missing)[:3]})")
    model.eval()
    print(f"loaded {args.ckpt.name} into the rebuilt shape ({build['params_after']:,} params, "
          f"0 missing / 0 unexpected)")

    rows = []
    for r in sorted(args.res, reverse=True):
        m = pose_map_model(model, data_yaml=args.data_yaml, imgsz=r,
                           device=args.device, batch=args.batch)
        rows.append({"imgsz": r, "map": m["map"], "map50": m["map50"]})
        print(f"  imgsz={r:>4}  mAP50-95={m['map']:.4f}  mAP50={m['map50']:.4f}")

    ctrl = next(x for x in rows if x["imgsz"] == CONTROL_RES)
    verdict = {"control_imgsz": CONTROL_RES, "control_map": ctrl["map"],
               "expect_map": args.expect_map}
    if args.expect_map is not None:
        delta = abs(ctrl["map"] - args.expect_map)
        verdict["control_delta"] = delta
        verdict["control_ok"] = bool(delta <= CONTROL_TOL)
        if not verdict["control_ok"]:
            print(f"\n  !! CONTROL FAILED: @640 gives {ctrl['map']:.4f}, recorded "
                  f"{args.expect_map:.4f} (delta {delta:.4f} > {CONTROL_TOL}). The state_dict "
                  f"load is wrong — every row above is meaningless. Do NOT act on this.")

    payload = {
        "note": "LOWER BOUND — the checkpoint is 640-trained; a model trained at each imgsz "
                "scores higher. Retention (train-at-res) is the open question this bounds.",
        "ckpt": str(args.ckpt), "spec": str(args.spec), "params": build["params_after"],
        "device": args.device, "data_yaml": str(args.data_yaml),
        "verdict": verdict, "rows": rows,
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
