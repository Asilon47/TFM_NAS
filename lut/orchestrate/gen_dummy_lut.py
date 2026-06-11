"""Generate a dummy data/lut.jsonl + data/device_info.json without a Jetson.

Stand-in for `orchestrate.run_sweep` when the real device isn't on hand.
Walks the same `iter_sweep()` the real pipeline uses, so every dummy row has
the row_key the real pipeline would have produced. Latencies come from a
roofline heuristic (FLOPs / peak compute, IO bytes / peak bandwidth) so they
correlate with op size — good enough to drive Phase 1+ NAS development; not
a substitute for measurement.

Usage:
  python -m lut.orchestrate.gen_dummy_lut                  # write data/lut.jsonl
  python -m lut.orchestrate.gen_dummy_lut --blocks mbconv conv3x3
  python -m lut.orchestrate.gen_dummy_lut --overwrite      # replace existing file
"""
import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from catalog.blocks import build_block, count_params
from catalog.flops import count_flops_forward
from catalog.sweep import iter_sweep, sweep_size


ROOT = Path(__file__).resolve().parents[2]

# Jetson Orin Nano (15W power mode), per public specs and docs/schema.md.
PEAK_FLOPS = 17e12       # FP16 dense, no sparsity
PEAK_BW = 51.2e9         # bytes/s; matches docs/schema.md `peak_dram_gbps_measured`
EFFICIENCY = 0.30        # real workloads hit 20-40% of peak
OVERHEAD_MS = 0.02       # CUDA kernel launch overhead floor
FP16_BYTES = 2

DEVICE_INFO = {
    "device": "Jetson Orin Nano",
    "power_mode": "15W",
    "gpu_clock_mhz_max": 1020,
    "emc_clock_mhz": 3200,
    "peak_dram_gbps_measured": 51.2,
    "trt_version": "8.6.2",
    "cuda_version": "12.2",
}

ROW_CONSTANTS = {
    "precision": "fp16",
    "trt_version": DEVICE_INFO["trt_version"],
    "power_mode": DEVICE_INFO["power_mode"],
    "jetpack": "6.0",
    "timed_iters": 200,
}


def measure_block(block: str, cfg: dict, input_shape) -> dict:
    """Build the module, run one forward pass, and capture flops + io_bytes + params."""
    module = build_block(block, cfg).eval()
    params = count_params(module)
    flops_total, y = count_flops_forward(module, input_shape)
    io_numel = int(np.prod(input_shape) + np.prod(y.shape))
    return {
        "params": params,
        "flops": flops_total,
        "io_bytes": io_numel * FP16_BYTES,
    }


def roofline_latency_ms(flops: int, io_bytes: int, rng: random.Random) -> float:
    compute_s = flops / (PEAK_FLOPS * EFFICIENCY)
    mem_s = io_bytes / (PEAK_BW * EFFICIENCY)
    base_ms = max(compute_s, mem_s) * 1000.0 + OVERHEAD_MS
    jittered = base_ms * (1.0 + rng.gauss(0.0, 0.02))
    return max(jittered, OVERHEAD_MS)


def make_row(block: str, cfg: dict, input_shape, row_key: str,
             rng: random.Random, timestamp: str) -> dict:
    m = measure_block(block, cfg, input_shape)
    flops, io_bytes = m["flops"], m["io_bytes"]

    lat_mean = roofline_latency_ms(flops, io_bytes, rng)
    latency_ms = {
        "mean": round(lat_mean, 4),
        "std": round(lat_mean * 0.02, 4),
        "p50": round(lat_mean * 0.99, 4),
        "p95": round(lat_mean * 1.04, 4),
        "n": ROW_CONSTANTS["timed_iters"],
    }

    peak_mem_mib = (io_bytes * 1.5) / (1024 * 1024) * (1.0 + rng.gauss(0.0, 0.05))
    peak_mem_mib = max(peak_mem_mib, 0.1)

    achieved_bw_gbps = io_bytes / (lat_mean * 1e-3) / 1e9

    return {
        "row_key": row_key,
        "block": block,
        "cfg": cfg,
        "input_shape": list(input_shape),
        "precision": ROW_CONSTANTS["precision"],
        "latency_ms": latency_ms,
        "peak_mem_mib": round(peak_mem_mib, 3),
        "params": m["params"],
        "flops": flops,
        "achieved_bw_gbps": round(achieved_bw_gbps, 3),
        "trt_version": ROW_CONSTANTS["trt_version"],
        "power_mode": ROW_CONSTANTS["power_mode"],
        "jetpack": ROW_CONSTANTS["jetpack"],
        "timestamp": timestamp,
        # Provenance: lets downstream consumers (and humans) tell roofline
        # estimates from real Jetson measurements in a mixed file.
        "source": "roofline_dummy",
    }


def write_device_info(path: Path, timestamp: str) -> None:
    info = dict(DEVICE_INFO, probed_at=timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "lut.jsonl"))
    ap.add_argument("--device-info-out",
                    default=str(ROOT / "data" / "device_info.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="Restrict to these block names (default: all).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists. Pass --overwrite to replace it.")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = sweep_size(args.blocks)
    print(f"Generating {total} dummy rows -> {out_path}")

    counts: Counter[str] = Counter()
    with open(out_path, "w") as f:
        for i, (block, cfg, input_shape, row_key) in enumerate(iter_sweep(args.blocks), 1):
            row = make_row(block, cfg, input_shape, row_key, rng, timestamp)
            f.write(json.dumps(row) + "\n")
            counts[block] += 1
            if i % 250 == 0 or i == total:
                print(f"  [{i}/{total}] {block}", flush=True)

    write_device_info(Path(args.device_info_out), timestamp)

    print(f"\nDone. Wrote {sum(counts.values())} rows.")
    print(f"  Device info: {args.device_info_out}")
    print("  By block:")
    for name in sorted(counts):
        print(f"    {name:14s} {counts[name]}")


if __name__ == "__main__":
    main()
