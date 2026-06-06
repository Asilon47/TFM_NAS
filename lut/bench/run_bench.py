"""Benchmark one TRT engine and emit a single JSON result line on stdout.

Runs inside the lut-runner container. Uses CUDA events for timing (the only
reliable source on Jetson — python-side time.perf_counter adds ~100 us jitter).

Output is a single JSON object on stdout. stderr carries progress/debug.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def load_engine(path: Path) -> trt.ICudaEngine:
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_io(engine: trt.ICudaEngine):
    """Allocate host+device buffers for every binding. Returns (inputs, outputs, bindings)."""
    inputs, outputs, bindings = [], [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape)) if all(d > 0 for d in shape) else 0
        host = np.random.randn(*shape).astype(dtype) if size else np.empty(0, dtype)
        dev = cuda.mem_alloc(host.nbytes) if host.nbytes else None
        bindings.append(int(dev) if dev else 0)
        entry = {"name": name, "shape": shape, "dtype": dtype,
                 "host": host, "dev": dev, "bytes": host.nbytes}
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(entry)
        else:
            outputs.append(entry)
    return inputs, outputs, bindings


def bench(engine_path: Path, warmup: int, iters: int) -> dict:
    engine = load_engine(engine_path)
    ctx = engine.create_execution_context()

    inputs, outputs, bindings = allocate_io(engine)
    for name_idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(name_idx)
        ctx.set_tensor_address(name, bindings[name_idx])

    stream = cuda.Stream()

    # Upload inputs once (we're not measuring H2D cost)
    for x in inputs:
        if x["dev"] is not None:
            cuda.memcpy_htod_async(x["dev"], x["host"], stream)
    stream.synchronize()

    free_before, _ = cuda.mem_get_info()

    for _ in range(warmup):
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()

    samples_ms = []
    start_evt = cuda.Event()
    end_evt = cuda.Event()
    for _ in range(iters):
        start_evt.record(stream)
        ctx.execute_async_v3(stream.handle)
        end_evt.record(stream)
        end_evt.synchronize()
        samples_ms.append(end_evt.time_since(start_evt))

    free_after, _ = cuda.mem_get_info()
    peak_mem_mib = max(0.0, (free_before - free_after) / (1024 * 1024))

    samples_ms.sort()
    n = len(samples_ms)
    p50 = samples_ms[n // 2]
    p95 = samples_ms[min(n - 1, int(n * 0.95))]

    # Compute bytes moved (rough: sum of all IO tensor byte sizes). Enables a
    # derived achieved_bw_gbps in the orchestrator.
    io_bytes = sum(e["bytes"] for e in inputs + outputs)

    return {
        "latency_ms": {
            "mean": statistics.fmean(samples_ms),
            "std":  statistics.pstdev(samples_ms) if n > 1 else 0.0,
            "p50":  p50,
            "p95":  p95,
            "n":    n,
        },
        "peak_mem_mib": peak_mem_mib,
        "io_bytes": io_bytes,
        "trt_version": trt.__version__,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters",  type=int, default=200)
    args = ap.parse_args()

    result = bench(Path(args.engine), args.warmup, args.iters)
    sys.stdout.write(json.dumps(result) + "\n")
