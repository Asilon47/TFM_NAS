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
from lut.orchestrate.probe_device import probe, write_device_info
from lut.orchestrate.resume import completed_keys
from lut.orchestrate.ssh_client import connect, load_config

ROOT = Path(__file__).resolve().parents[2]


def run_remote_bench(conn, cfg, row_key: str, local_onnx: Path,
                     precision: str, warmup: int, iters: int,
                     min_window_s: float, fresh_timing_cache: bool = False) -> dict:
    remote_job = f"{cfg.remote_workdir}/job/{row_key}"
    conn.run(f"mkdir -p {remote_job}", hide=True)
    try:
        conn.put(str(local_onnx), remote=f"{remote_job}/model.onnx")
        # /job mounts the per-row dir; bench/ rides in via an explicit second
        # mount because `..` traversal in container paths is brittle. /cache
        # persists the trtexec timing cache across rows AND sweeps — which makes
        # a build reuse a prior run's autotuner tactics. For the LUT sweep that
        # is the intended speedup, but a median-of-N fp16 REBUILD study (winner-v2
        # variance de-risk) must NOT reuse tactics or every rebuild is identical;
        # fresh_timing_cache points each build at a unique cache file inside /job
        # (torn down with the row) so the autotuner re-searches from scratch.
        cache_flag = (f"--timing-cache /job/trt_timing_{row_key}.cache"
                      if fresh_timing_cache else "--timing-cache /cache/trt_timing.cache")
        cmd = (
            f"docker run --rm --runtime nvidia "
            f"-v {remote_job}:/job "
            f"-v {cfg.remote_workdir}/bench:/bench "
            f"-v {cfg.remote_workdir}/cache:/cache "
            f"{cfg.docker_image} bash -c "
            f"'python3 /bench/build_engine.py "
            f"--onnx /job/model.onnx --engine /job/model.plan --precision {precision} "
            f"{cache_flag} "
            f"&& python3 /bench/run_bench.py "
            f"--engine /job/model.plan --warmup {warmup} --iters {iters} "
            f"--min-window-s {min_window_s}'"
        )
        res = conn.run(cmd, hide=True, warn=True)
        if res.return_code != 0:
            raise RuntimeError(
                f"remote bench failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        return _parse_bench_stdout(res.stdout)
    finally:
        conn.run(f"rm -rf {remote_job}", hide=True, warn=True)


def _parse_bench_stdout(stdout: str) -> dict:
    """run_bench.py prints one JSON object as the last non-empty stdout line.

    Raises ValueError (never IndexError) with the offending output embedded,
    so a container that exits 0 without producing results is diagnosable
    straight from the sweep log.
    """
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            "remote bench produced no stdout despite exit code 0 — the "
            "container likely failed before run_bench.py started"
        )
    last = lines[-1]
    try:
        result = json.loads(last)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"remote bench's last stdout line is not JSON ({e}): {last[:200]!r}"
        ) from e
    if not isinstance(result, dict):
        raise ValueError(f"remote bench JSON is not an object: {last[:200]!r}")
    return result


def preflight_verdict(device_info: dict, expected_power_mode: int | None,
                      require_locked_clocks: bool = True) -> str | None:
    """Is the device state fit to measure? None = proceed; a string aborts.

    Called on a FRESH probe at sweep start. This function is policy, not
    plumbing: it owns the call on which device-state mismatches invalidate a
    measurement session. The two hard conditions ship fail-fast because they
    silently corrupt rows otherwise:

      * unlocked clocks — jetson_clocks does not survive a reboot, so a
        post-reboot sweep would measure with DVFS active while still
        stamping the configured power mode into every row;
      * power-mode mismatch — rows are only comparable within one mode.

    TODO(user): own/extend this policy. Open decisions left to you:
      - should a failed bandwidth probe (peak_dram_gbps_measured == 0) abort
        instead of warn? (it only affects sanity checks, not latencies)
      - should a trt_version/jetpack change vs. the previous
        data/device_info.json abort, to keep one LUT per software stack?
    """
    if require_locked_clocks and device_info.get("clocks_locked") is not True:
        return (
            "GPU clocks are not locked (jetson_clocks does not survive "
            "reboots). Run scripts/setup_jetson.sh, or set "
            "jetson.lock_clocks: false to knowingly measure with DVFS active."
        )
    if expected_power_mode is not None:
        actual = device_info.get("power_mode")
        if actual is None or str(actual) != str(expected_power_mode):
            return (
                f"power mode mismatch: device reports {actual!r}, config "
                f"expects {expected_power_mode!r}. Run scripts/setup_jetson.sh "
                "or update jetson.power_mode in config.yaml."
            )
    if not device_info.get("peak_dram_gbps_measured"):
        sys.stderr.write(
            "[preflight] WARN: bandwidthTest probe failed "
            "(peak_dram_gbps_measured is 0/absent) — achieved_bw sanity "
            "checks against DRAM peak will be meaningless.\n"
        )
    return None


def load_device_info(path: Path) -> dict:
    """Device metadata stamped into every LUT row (power_mode, jetpack, ...).

    Only consulted with ``--skip-preflight`` — the default path re-probes the
    device and never reads a stale file (see preflight_verdict).

    PROJECT_PLAN Phase 0's DoD requires data/device_info.json to exist for
    the locked power mode: rows stamped ``power_mode: None`` are ambiguous
    and can't be compared across sweeps, so degrading silently is a trap.

    TODO(user): implement the failure policy here (~8 lines). Requirements:
      (a) a row must never silently get power_mode=None,
      (b) a missing file and a corrupt file produce distinguishable errors,
      (c) the message names `python -m lut.orchestrate.probe_device` as the
          remedy.
    Recommended shape: fail fast (raise SystemExit) by default, with an
    `--allow-missing-device-info` escape hatch threaded through main() for
    bring-up on a fresh device. Legacy behavior (silent {}) kept below until
    the policy lands.
    """
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="Only sweep these block names (default: all).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N new rows. 0 = unlimited.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Trust data/device_info.json instead of re-probing "
                         "the device at sweep start (bring-up/debug only — "
                         "clock-lock state will NOT be verified).")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg, sweep_cfg = load_config(Path(args.config))
    precision = sweep_cfg.get("precision", "fp16")
    warmup = int(sweep_cfg.get("warmup_iters", 50))
    iters = int(sweep_cfg.get("timed_iters", 200))
    min_window_s = float(sweep_cfg.get("min_window_s", 0.5))
    out_jsonl = ROOT / sweep_cfg.get("output_jsonl", "data/lut.jsonl")
    dev_info_path = ROOT / sweep_cfg.get("device_info_json", "data/device_info.json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(cfg)
    if args.skip_preflight:
        device_info = load_device_info(dev_info_path)
    else:
        # Fresh state at every sweep start: jetson_clocks does not survive a
        # reboot and device_info.json may be days stale. Re-probe, persist,
        # and refuse to measure in a state that would corrupt rows.
        device_info = probe(conn, cfg)
        write_device_info(device_info, dev_info_path)
        reason = preflight_verdict(device_info, cfg.power_mode,
                                   require_locked_clocks=cfg.lock_clocks)
        if reason is not None:
            raise SystemExit(f"[preflight] {reason}")
        print(f"Preflight OK: power_mode={device_info.get('power_mode')!r} "
              f"clocks_locked={device_info.get('clocks_locked')} "
              f"gpu={device_info.get('gpu_clock_mhz_cur')} MHz", flush=True)

    # Precision-aware: row_key does not encode precision, so resuming under a
    # different precision must re-measure, not skip (see lut/docs/schema.md).
    done = completed_keys(out_jsonl, precision=precision)
    total = sweep_size(args.blocks)

    print(f"Sweep size: {total} rows (catalog). Already complete: {len(done)}. "
          f"Pending: ~{max(0, total - len(done))}.", flush=True)

    conn.run(f"mkdir -p {cfg.remote_workdir}/job {cfg.remote_workdir}/cache",
             hide=True)

    n_new = 0
    failures: list[tuple[str, str, str]] = []
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
                    min_window_s=min_window_s,
                )
            except Exception as e:
                # One bad row must not kill an overnight sweep: log, record,
                # move on. The end-of-run summary + exit code surface it.
                sys.stderr.write(f"\n[ERR] {block} {block_cfg} {input_shape}: {e}\n")
                traceback.print_exc(file=sys.stderr)
                failures.append((row_key, block, repr(e)))
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
                # None (not False) when unknown, e.g. --skip-preflight with an
                # old device_info.json that predates the clocks_locked probe.
                "clocks_locked": device_info.get("clocks_locked"),
                "source": "jetson_trt",
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
    if failures:
        sys.stderr.write(
            f"\n[run_sweep] {len(failures)} row(s) FAILED and were skipped:\n")
        for key, block, err in failures[:10]:
            sys.stderr.write(f"  {key} {block}: {err[:120]}\n")
        if len(failures) > 10:
            sys.stderr.write(f"  ... and {len(failures) - 10} more\n")
        sys.stderr.write(
            "Re-running the sweep retries only failed rows (resume skips "
            "completed ones).\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
