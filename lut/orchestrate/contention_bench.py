"""Contention experiment — does a co-running load close or WIDEN the graft's memory-bound gap?

Hypothesis under test (user, 2026-07-14): the OFA graft loses to yolo11n on the *isolated*
bench because DRAM bandwidth is fully free; on a *loaded* board the two would equalize. The
roofline prediction is the opposite — a bandwidth-bound net (the graft) is hurt MORE when
effective bandwidth drops, so contention should WIDEN the gap. This harness measures it.

For each stressor condition it benches each model (default fp16) with a co-running load and
records the latency, then reports the graft/baseline RATIO per condition. Ratio rising with
load ⇒ roofline (gap widens); ratio falling ⇒ the user's hypothesis (and a publishable
result). These are DELIBERATELY loaded numbers — NOT LUT rows, NOT the clean frontier; they
land in ``data/contention/`` and must never be compared to the isolated bench.

Stressors (co-process on the Jetson host, killed after each condition):
  * ``cpu``  — ``stress-ng --cpu 0`` (all 6 Orin cores): CPU + shared-memory-controller pressure
  * ``dram`` — ``stress-ng --vm``: saturates the shared LPDDR5 bandwidth (the roofline axis)
  * ``gpu``  — a detached ``trtexec`` loop on yolo11s: GPU compute + bandwidth contention
  * ``all``  — cpu+dram+gpu together (the realistic "loaded drone board")

Run (laptop ``.venv``; setup_jetson.sh first, teardown after)::

    python -m lut.orchestrate.contention_bench \
        --model baseline=models/baseline/yolo11n_pose_640.onnx \
        --model graft=data/e2e/winner_v1_e2e_640.onnx \
        --gpu-stressor models/anchor/yolo11s_pose_640.onnx \
        --conditions none,cpu,dram,gpu,all --precision fp16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from lut.orchestrate.probe_device import probe, write_device_info
from lut.orchestrate.run_sweep import preflight_verdict, run_remote_bench
from lut.orchestrate.ssh_client import connect, load_config

ROOT = Path(__file__).resolve().parents[2]
GPU_STRESS_NAME = "tfm_gpu_stress"      # detached docker container name (idempotent teardown)


def ensure_stress_ng(conn) -> bool:
    """True if stress-ng is available (install once via passwordless sudo if missing)."""
    if conn.run("command -v stress-ng", warn=True, hide=True).ok:
        return True
    print("[contention] installing stress-ng ...", flush=True)
    conn.run("sudo apt-get update -q && sudo apt-get install -y stress-ng",
             warn=True, hide=True)
    return conn.run("command -v stress-ng", warn=True, hide=True).ok


def start_stressor(conn, cfg, kind: str, *, gpu_onnx_remote: str | None,
                   duration_s: int = 1800) -> None:
    """Launch one host-side stressor detached (killed by ``stop_stressors``)."""
    if kind == "cpu":
        conn.run(f"nohup stress-ng --cpu 0 --timeout {duration_s}s "
                 ">/tmp/tfm_cpu_stress.log 2>&1 & echo started", hide=True, warn=True)
    elif kind == "dram":
        # --vm workers churning 80% RAM saturates the shared LPDDR5 controller (bandwidth).
        conn.run(f"nohup stress-ng --vm 4 --vm-bytes 80% --vm-method all "
                 f"--timeout {duration_s}s >/tmp/tfm_dram_stress.log 2>&1 & echo started",
                 hide=True, warn=True)
    elif kind == "gpu":
        if gpu_onnx_remote is None:
            raise ValueError("gpu stressor needs --gpu-stressor <onnx>")
        conn.run(f"docker rm -f {GPU_STRESS_NAME}", hide=True, warn=True)
        # one build, then continuous inference for duration_s → a steady GPU hog.
        conn.run(
            f"docker run -d --rm --runtime nvidia --name {GPU_STRESS_NAME} "
            f"-v {cfg.remote_workdir}/gpu_stress:/job {cfg.docker_image} "
            f"bash -c 'trtexec --onnx=/job/stressor.onnx --fp16 --duration={duration_s} "
            "--iterations=0 >/tmp/tfm_gpu_stress.log 2>&1'", hide=True, warn=True)
    else:
        raise ValueError(f"unknown stressor {kind!r}")


def gpu_stressor_ready(conn, timeout_s: int = 120) -> bool:
    """Wait until the trtexec hog leaves its build phase and is actually running inference."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = conn.run(f"docker logs {GPU_STRESS_NAME} 2>&1 | tail -5", hide=True, warn=True)
        if r.ok and ("Starting inference" in r.stdout or "GPU Compute Time" in r.stdout
                     or "Throughput" in r.stdout):
            return True
        time.sleep(5)
    return False


def stop_stressors(conn) -> None:
    """Kill every stressor (idempotent — safe to call in a finally block)."""
    conn.run("pkill -f 'stress-ng' || true", hide=True, warn=True)
    conn.run(f"docker rm -f {GPU_STRESS_NAME}", hide=True, warn=True)


def run_condition(conn, cfg, sweep_cfg, *, condition: str, models: dict[str, Path],
                  precision: str, gpu_onnx_remote: str | None, warmup_s: int) -> dict:
    """Bench every model under one stressor condition; returns {model: latency_ms}."""
    kinds = [] if condition == "none" else (
        ["cpu", "dram", "gpu"] if condition == "all" else [condition])
    try:
        for k in kinds:
            start_stressor(conn, cfg, k, gpu_onnx_remote=gpu_onnx_remote)
        if "gpu" in kinds and not gpu_stressor_ready(conn):
            print(f"[{condition}] WARN gpu stressor not confirmed running", flush=True)
        if kinds:
            time.sleep(warmup_s)   # let the load ramp before timing
        out: dict = {}
        for name, onnx in models.items():
            res = run_remote_bench(
                conn, cfg, f"contention_{condition}_{name}", onnx, precision=precision,
                warmup=int(sweep_cfg.get("warmup_iters", 50)),
                iters=int(sweep_cfg.get("timed_iters", 200)),
                min_window_s=float(sweep_cfg.get("min_window_s", 0.5)))
            ms = float(res["latency_ms"]["mean"])
            out[name] = ms
            print(f"[{condition}] {name:10s} {ms:7.3f} ms", flush=True)
        return out
    finally:
        stop_stressors(conn)


def analyze(rows: dict[str, dict], baseline: str, graft: str) -> list[dict]:
    """Per-condition graft/baseline ratio + Δ vs the unloaded control (the verdict table)."""
    base_ctrl = rows.get("none", {}).get(baseline)
    graft_ctrl = rows.get("none", {}).get(graft)
    ctrl_ratio = (graft_ctrl / base_ctrl) if base_ctrl and graft_ctrl else None
    table = []
    for cond, r in rows.items():
        b, g = r.get(baseline), r.get(graft)
        ratio = (g / b) if b and g else None
        table.append({
            "condition": cond,
            "baseline_ms": b, "graft_ms": g,
            "gap_ms": (g - b) if b and g else None,
            "ratio": ratio,
            "ratio_delta_vs_control": (ratio - ctrl_ratio)
            if ratio is not None and ctrl_ratio is not None else None,
            "baseline_slowdown_vs_control": (b / rows["none"][baseline])
            if b and rows.get("none", {}).get(baseline) else None,
            "graft_slowdown_vs_control": (g / rows["none"][graft])
            if g and rows.get("none", {}).get(graft) else None,
        })
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=ONNX",
                    help="repeatable; first is treated as the baseline, second as the graft")
    ap.add_argument("--gpu-stressor", type=Path, default=None,
                    help="ONNX for the GPU trtexec hog (e.g. yolo11s); required if 'gpu'/'all'")
    ap.add_argument("--conditions", default="none,cpu,dram,gpu,all")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--warmup-s", type=int, default=6, help="load ramp before timing")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args(argv)

    models: dict[str, Path] = {}
    for spec in a.model:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"model ONNX not found: {p}")
        models[name] = p
    names = list(models)
    if len(names) < 2:
        raise SystemExit("need >=2 --model (baseline + graft) to measure a gap")
    baseline, graft = names[0], names[1]

    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    cfg, sweep_cfg = load_config(Path(a.config))
    conn = connect(cfg)

    if not a.skip_preflight:
        device_info = probe(conn, cfg)
        write_device_info(device_info,
                          ROOT / sweep_cfg.get("device_info_json", "data/device_info.json"))
        reason = preflight_verdict(device_info, cfg.power_mode,
                                   require_locked_clocks=cfg.lock_clocks)
        if reason is not None:
            raise SystemExit(f"[preflight] {reason}")
        print(f"Preflight OK: power_mode={device_info.get('power_mode')!r} "
              f"clocks_locked={device_info.get('clocks_locked')}", flush=True)
    else:
        device_info = {}

    gpu_remote = None
    need_gpu = any(c in ("gpu", "all") for c in conditions)
    if need_gpu:
        if a.gpu_stressor is None or not a.gpu_stressor.exists():
            raise SystemExit("--gpu-stressor <onnx> required for the gpu/all conditions")
        conn.run(f"mkdir -p {cfg.remote_workdir}/gpu_stress", hide=True)
        conn.put(str(a.gpu_stressor), remote=f"{cfg.remote_workdir}/gpu_stress/stressor.onnx")
        gpu_remote = f"{cfg.remote_workdir}/gpu_stress/stressor.onnx"

    rows: dict[str, dict] = {}
    try:
        for cond in conditions:
            rows[cond] = run_condition(
                conn, cfg, sweep_cfg, condition=cond, models=models,
                precision=a.precision, gpu_onnx_remote=gpu_remote, warmup_s=a.warmup_s)
    finally:
        stop_stressors(conn)

    table = analyze(rows, baseline, graft)
    report = {
        "experiment": "memory-bound contention (loaded bench — NOT LUT/frontier)",
        "baseline": baseline, "graft": graft, "precision": a.precision,
        "regime": {"power_mode": device_info.get("power_mode"),
                   "clocks_locked": device_info.get("clocks_locked"),
                   "trt_version": device_info.get("trt_version")},
        "raw_ms": rows, "table": table,
        "hypothesis": "ratio rising with load => roofline (gap WIDENS); falling => equalizes",
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = a.out or ROOT / "data" / "contention" / f"contention_{a.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'condition':10s} {baseline:>10s} {graft:>10s} {'gap ms':>8s} "
          f"{'ratio':>7s} {'Δratio':>8s}")
    for r in table:
        print(f"{r['condition']:10s} {r['baseline_ms'] or 0:10.3f} {r['graft_ms'] or 0:10.3f} "
              f"{r['gap_ms'] or 0:8.3f} {r['ratio'] or 0:7.3f} "
              f"{(r['ratio_delta_vs_control'] or 0):+8.3f}")
    print(f"\nwrote {out}\n(Δratio>0 ⇒ gap widens under load = roofline; <0 ⇒ equalizes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
