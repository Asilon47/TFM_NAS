"""End-to-end frame benchmark — quantify what fraction of a deployed frame is inference.

The isolated bench (7.7 ms) excludes preprocessing + pose postprocess; the deployed yolo11n
ran at ~12 FPS (≈83 ms). This uploads a model, builds its TRT engine in the lut-runner
container, and runs ``lut/bench/e2e_run.py`` (preprocess + inference + postprocess, on the
Orin's own CPU/GPU) to report the per-stage split and the inference fraction.

Run (laptop ``.venv``; setup_jetson.sh first so /bench has e2e_run.py)::

    python -m lut.orchestrate.e2e_bench --onnx models/baseline/yolo11n_pose_640.onnx \
        --name baseline --precision fp16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from lut.orchestrate.probe_device import probe, write_device_info
from lut.orchestrate.run_sweep import preflight_verdict
from lut.orchestrate.ssh_client import connect, load_config

ROOT = Path(__file__).resolve().parents[2]


def run_remote_e2e(conn, cfg, name: str, local_onnx: Path, *, precision: str,
                   imgsz: int, iters: int) -> dict:
    """Build the engine + run e2e_run.py in the container; return the parsed stage split."""
    job = f"{cfg.remote_workdir}/job/e2e_{name}"
    conn.run(f"mkdir -p {job}", hide=True)
    try:
        conn.put(str(local_onnx), remote=f"{job}/model.onnx")
        cmd = (
            f"docker run --rm --runtime nvidia -v {job}:/job "
            f"-v {cfg.remote_workdir}/bench:/bench {cfg.docker_image} bash -c "
            f"'python3 /bench/build_engine.py --onnx /job/model.onnx --engine /job/model.plan "
            f"--precision {precision} --timing-cache /job/tc.cache "
            f"&& python3 /bench/e2e_run.py --engine /job/model.plan --imgsz {imgsz} "
            f"--iters {iters}'")
        res = conn.run(cmd, hide=True, warn=True)
        if res.return_code != 0:
            raise RuntimeError(f"remote e2e failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        line = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1]
        return json.loads(line)
    finally:
        conn.run(f"rm -rf {job}", hide=True, warn=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args(argv)

    if not a.onnx.exists():
        raise SystemExit(f"ONNX not found: {a.onnx}")
    name = a.name or a.onnx.stem
    cfg, sweep_cfg = load_config(Path(a.config))
    conn = connect(cfg)

    if not a.skip_preflight:
        di = probe(conn, cfg)
        write_device_info(di, ROOT / sweep_cfg.get("device_info_json", "data/device_info.json"))
        reason = preflight_verdict(di, cfg.power_mode, require_locked_clocks=cfg.lock_clocks)
        if reason is not None:
            raise SystemExit(f"[preflight] {reason}")
        print(f"Preflight OK: power_mode={di.get('power_mode')!r} "
              f"clocks_locked={di.get('clocks_locked')}", flush=True)

    r = run_remote_e2e(conn, cfg, name, a.onnx, precision=a.precision,
                       imgsz=a.imgsz, iters=a.iters)
    report = {"name": name, "precision": a.precision, "imgsz": a.imgsz,
              "backend": "jetson_trt+numpy_pre_post", **r,
              "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    out = a.out or ROOT / "data" / "contention" / f"e2e_{name}_{a.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    pre, inf, post = r["preprocess_ms"], r["inference_ms"], r["postprocess_ms"]
    print(f"\n{name} @{a.imgsz} ({a.precision}) — cv2={r['cv2']}, output {r['output_shape']}")
    print(f"  preprocess  {pre['mean']:7.2f} ms")
    print(f"  inference   {inf['mean']:7.2f} ms")
    print(f"  postprocess {post['mean']:7.2f} ms")
    print(f"  TOTAL       {r['total_ms']:7.2f} ms  = {r['fps']:.1f} FPS  "
          f"(inference is {100 * r['inference_fraction']:.0f}% of the frame)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
