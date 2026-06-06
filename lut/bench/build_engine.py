"""Build a TensorRT engine from an ONNX file using trtexec.

Runs inside the lut-runner container. Called by orchestrate/run_sweep.py via
`docker run ... bash -c "python3 build_engine.py ... && python3 run_bench.py ..."`.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def build(onnx_path: Path, engine_path: Path, precision: str = "fp16") -> float:
    cmd = ["trtexec",
           f"--onnx={onnx_path}",
           f"--saveEngine={engine_path}",
           "--skipInference",     # we time the engine ourselves in run_bench.py
           "--memPoolSize=workspace:512"]
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
    args = ap.parse_args()
    dt = build(Path(args.onnx), Path(args.engine), args.precision)
    print(f"engine_build_s={dt:.3f}", file=sys.stderr)
