"""Build a TensorRT engine from an ONNX file using trtexec.

Runs inside the lut-runner container. Called by orchestrate/run_sweep.py via
`docker run ... bash -c "python3 build_engine.py ... && python3 run_bench.py ..."`.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# trtexec ships with the l4t-tensorrt base image but under /usr/src/tensorrt/bin,
# which is NOT on the container's PATH — so a bare "trtexec" raises
# FileNotFoundError. Resolve it explicitly: prefer PATH (in case a future image
# fixes this), fall back to the canonical location.
_TRTEXEC_FALLBACK = "/usr/src/tensorrt/bin/trtexec"


def _resolve_trtexec() -> str:
    found = shutil.which("trtexec")
    if found:
        return found
    if os.access(_TRTEXEC_FALLBACK, os.X_OK):
        return _TRTEXEC_FALLBACK
    raise SystemExit(
        f"trtexec not found on PATH or at {_TRTEXEC_FALLBACK}. The l4t-tensorrt "
        "base image ships it under /usr/src/tensorrt/bin — add that dir to PATH "
        "in Dockerfile.runner, or rebuild the runner image."
    )


def build(onnx_path: Path, engine_path: Path, precision: str = "fp16",
          timing_cache: str | None = None) -> float:
    cmd = [_resolve_trtexec(),
           f"--onnx={onnx_path}",
           f"--saveEngine={engine_path}",
           "--skipInference",     # we time the engine ourselves in run_bench.py
           "--memPoolSize=workspace:512"]
    if timing_cache:
        # Persistent tactic-timing cache (created on first use): later builds
        # reuse measured tactic timings — much faster across a 2710-row sweep
        # — and identical layers resolve to identical tactics across rows,
        # which keeps per-block latencies mutually consistent.
        cmd.append(f"--timingCacheFile={timing_cache}")
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit(f"trtexec failed ({res.returncode}) for {onnx_path}")
    return dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "int8"])
    ap.add_argument("--timing-cache", default=None,
                    help="Path to a persistent trtexec timing cache file.")
    args = ap.parse_args()
    dt = build(Path(args.onnx), Path(args.engine), args.precision, args.timing_cache)
    print(f"engine_build_s={dt:.3f}", file=sys.stderr)
