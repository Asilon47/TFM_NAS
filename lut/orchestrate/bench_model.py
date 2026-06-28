"""Benchmark an arbitrary whole-model ONNX on the Jetson — the baseline anchor.

The owed CP 3.3 / D4 input: the **yolo11n-pose baseline-to-beat** latency @640 on the
Orin Nano. It sets the hard ceiling ``T_max = min(baseline, fps_to_ms(60)=16.7 ms)``
the search filters candidates against, and the iso-J λ calibration's reference scale.

Unlike ``run_sweep`` / ``measure_additivity`` these are NOT LUT rows — a whole model
has no per-block ``row_key`` — so the result lands in its own JSON
(``data/baseline_anchor.json``) rather than ``data/lut.jsonl``. The remote build +
benchmark path is reused verbatim from ``run_sweep`` (TRT engine in the lut-runner
docker, CUDA-event timing, the same preflight that refuses unlocked clocks).

**Precision.** Defaults to the config's ``sweep.precision`` (fp32) so the baseline is
measured against the *same* precision the @640 LUT (and therefore the costed
candidates) uses — a fp16 baseline would not be a like-for-like ceiling. Pass
``--precision fp16`` for the deploy-realistic figure (a separate, Phase-8/9 number).

Run on the laptop (``.venv``, fabric); the ONNX comes from
``python -m detect.export_baseline_onnx`` where ultralytics lives::

    python -m lut.orchestrate.bench_model --onnx yolo11n_pose_640.onnx \
        --name yolo11n_pose_640 --imgsz 640 --out data/baseline_anchor.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from lut.orchestrate.probe_device import probe, write_device_info
from lut.orchestrate.run_sweep import preflight_verdict, run_remote_bench
from lut.orchestrate.ssh_client import connect, load_config

ROOT = Path(__file__).resolve().parents[2]


def baseline_row(bench_result: dict, *, name: str, precision: str, imgsz: int,
                 device_info: dict) -> dict:
    """Normalize a remote bench result + device metadata into the anchor JSON row.

    Same enriched shape as a LUT row minus the per-block keys (``row_key``/``cfg``):
    the full latency distribution + working-set peak from the device, the achieved
    bandwidth derived laptop-side, and the device-state stamps that make the number
    comparable to the LUT (``power_mode``/``clocks_locked``/``trt_version``).
    """
    lat_mean_s = bench_result["latency_ms"]["mean"] / 1000.0
    achieved_bw_gbps = (bench_result["io_bytes"] / lat_mean_s) / 1e9 if lat_mean_s > 0 else 0.0
    return {
        "name": name,
        "precision": precision,
        "imgsz": imgsz,
        "latency_ms": bench_result["latency_ms"],
        "peak_mem_mib": bench_result["peak_mem_mib"],
        "achieved_bw_gbps": achieved_bw_gbps,
        "trt_version": bench_result.get("trt_version"),
        "power_mode": device_info.get("power_mode"),
        "jetpack": device_info.get("jetpack"),
        "clocks_locked": device_info.get("clocks_locked"),
        "source": "jetson_trt",
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, required=True, help="whole-model ONNX to benchmark")
    ap.add_argument("--name", default=None, help="anchor name (default: ONNX stem)")
    ap.add_argument("--imgsz", type=int, default=640, help="provenance only (stamped into the row)")
    ap.add_argument("--precision", default=None,
                    help="override; default is sweep.precision from config (fp32, LUT-consistent)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "baseline_anchor.json")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="trust device state instead of re-probing (bring-up/debug only)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args(argv)

    if not args.onnx.exists():
        raise SystemExit(
            f"ONNX not found: {args.onnx}. Export it where ultralytics lives:\n"
            "  python -m detect.export_baseline_onnx --weights yolo11n-pose.pt --imgsz 640")

    cfg, sweep_cfg = load_config(Path(args.config))
    precision = args.precision or sweep_cfg.get("precision", "fp16")
    name = args.name or args.onnx.stem

    conn = connect(cfg)
    if args.skip_preflight:
        device_info = json.loads(
            (ROOT / sweep_cfg.get("device_info_json", "data/device_info.json")).read_text())
    else:
        # Same gate as run_sweep: a latency measured under DVFS or the wrong power
        # mode is not comparable to the LUT the ceiling is enforced against.
        device_info = probe(conn, cfg)
        write_device_info(
            device_info, ROOT / sweep_cfg.get("device_info_json", "data/device_info.json"))
        reason = preflight_verdict(device_info, cfg.power_mode,
                                   require_locked_clocks=cfg.lock_clocks)
        if reason is not None:
            raise SystemExit(f"[preflight] {reason}")
        print(f"Preflight OK: power_mode={device_info.get('power_mode')!r} "
              f"clocks_locked={device_info.get('clocks_locked')}", flush=True)

    conn.run(f"mkdir -p {cfg.remote_workdir}/job {cfg.remote_workdir}/cache", hide=True)
    bench = run_remote_bench(
        conn, cfg, name, args.onnx, precision=precision,
        warmup=int(sweep_cfg.get("warmup_iters", 50)),
        iters=int(sweep_cfg.get("timed_iters", 200)),
        min_window_s=float(sweep_cfg.get("min_window_s", 0.5)),
    )
    row = baseline_row(bench, name=name, precision=precision, imgsz=args.imgsz,
                       device_info=device_info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(f"baseline {name} @{args.imgsz} ({precision}): "
          f"latency={row['latency_ms']['mean']:.4g} ms  peak={row['peak_mem_mib']:.4g} MiB "
          f"-> {args.out}")
    print(f"T_max = min({row['latency_ms']['mean']:.4g} ms baseline, 16.7 ms @60FPS) "
          "feeds `search.bo --t-max-ms`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
