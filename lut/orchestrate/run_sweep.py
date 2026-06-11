"""Main LUT-building orchestrator.

For each (block, cfg) in the catalog:
  1. Export FP32 ONNX locally.
  2. scp ONNX to Jetson:$REMOTE/job/.
  3. docker run the lut-runner: builds FP16 engine, benchmarks it, prints JSON.
  4. Parse JSON stdout → append enriched row to data/lut.jsonl.
  5. Remove remote ONNX + engine.

Idempotent: rows already in lut.jsonl (matched by row_key) are skipped.

Usage:
  python -m lut.orchestrate.run_sweep                 # full sweep
  python -m lut.orchestrate.run_sweep --blocks mbconv conv3x3
  python -m lut.orchestrate.run_sweep --limit 5       # dry-run / smoke test
"""
import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from tqdm import tqdm

from catalog.blocks import build_block
from catalog.flops import count_flops
from catalog.sweep import iter_sweep, sweep_size
from lut.export.to_onnx import export_block
from lut.orchestrate.ssh_client import load_config, connect
from lut.orchestrate.resume import completed_keys


ROOT = Path(__file__).resolve().parents[2]


def run_remote_bench(conn, cfg, row_key: str, local_onnx: Path,
                     precision: str, warmup: int, iters: int) -> dict:
    remote_job = f"{cfg.remote_workdir}/job/{row_key}"
    conn.run(f"mkdir -p {remote_job}", hide=True)
    try:
        conn.put(str(local_onnx), remote=f"{remote_job}/model.onnx")
        # /job mounts the per-row dir; bench/ rides in via an explicit second
        # mount because `..` traversal in container paths is brittle.
        cmd = (
            f"docker run --rm --runtime nvidia "
            f"-v {remote_job}:/job "
            f"-v {cfg.remote_workdir}/bench:/bench "
            f"{cfg.docker_image} bash -c "
            f"'python3 /bench/build_engine.py "
            f"--onnx /job/model.onnx --engine /job/model.plan --precision {precision} "
            f"&& python3 /bench/run_bench.py "
            f"--engine /job/model.plan --warmup {warmup} --iters {iters}'"
        )
        res = conn.run(cmd, hide=True, warn=True)
        if res.return_code != 0:
            raise RuntimeError(f"remote bench failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        # run_bench.py emits one JSON object on stdout (last non-empty line).
        lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
        return json.loads(lines[-1])
    finally:
        conn.run(f"rm -rf {remote_job}", hide=True, warn=True)


def load_device_info(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="Only sweep these block names (default: all).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N new rows. 0 = unlimited.")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg, sweep_cfg = load_config(Path(args.config))
    precision = sweep_cfg.get("precision", "fp16")
    warmup = int(sweep_cfg.get("warmup_iters", 50))
    iters = int(sweep_cfg.get("timed_iters", 200))
    out_jsonl = ROOT / sweep_cfg.get("output_jsonl", "data/lut.jsonl")
    dev_info_path = ROOT / sweep_cfg.get("device_info_json", "data/device_info.json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device_info = load_device_info(dev_info_path)
    done = completed_keys(out_jsonl)
    total = sweep_size(args.blocks)

    print(f"Sweep size: {total} rows (catalog). Already complete: {len(done)}. "
          f"Pending: ~{max(0, total - len(done))}.", flush=True)

    conn = connect(cfg)
    conn.run(f"mkdir -p {cfg.remote_workdir}/job", hide=True)

    n_new = 0
    with tempfile.TemporaryDirectory(prefix="lut_onnx_") as tmp, \
         open(out_jsonl, "a") as out_f:
        tmp = Path(tmp)
        bar = tqdm(iter_sweep(args.blocks), total=total)
        for block, block_cfg, input_shape, row_key in bar:
            if row_key in done:
                continue
            onnx_path = tmp / f"{row_key}.onnx"
            try:
                meta = export_block(block, block_cfg, input_shape, onnx_path)
                flops = count_flops(build_block(block, block_cfg).eval(), input_shape)
                bench_result = run_remote_bench(
                    conn, cfg, row_key, onnx_path,
                    precision=precision, warmup=warmup, iters=iters,
                )
            except Exception as e:
                sys.stderr.write(f"\n[ERR] {block} {block_cfg} {input_shape}: {e}\n")
                traceback.print_exc(file=sys.stderr)
                continue

            lat_mean_s = bench_result["latency_ms"]["mean"] / 1000.0
            achieved_bw_gbps = (bench_result["io_bytes"] / lat_mean_s) / 1e9 \
                               if lat_mean_s > 0 else 0.0

            row = {
                "row_key": row_key,
                "block": block,
                "cfg": block_cfg,
                "input_shape": list(input_shape),
                "precision": precision,
                "latency_ms": bench_result["latency_ms"],
                "peak_mem_mib": bench_result["peak_mem_mib"],
                "params": meta["params"],
                "flops": flops,
                "achieved_bw_gbps": achieved_bw_gbps,
                "trt_version": bench_result.get("trt_version"),
                "power_mode": device_info.get("power_mode"),
                "jetpack": device_info.get("jetpack"),
                "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            done.add(row_key)
            n_new += 1
            bar.set_postfix_str(f"new={n_new} last={block}")

            if args.limit and n_new >= args.limit:
                break

    print(f"Done. Added {n_new} rows to {out_jsonl}.")


if __name__ == "__main__":
    main()
