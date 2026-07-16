"""Bench the pose frontier on this machine's CPU -- the cross-device rank check.

Every latency in this repo is a Jetson/TensorRT number, so the CP 6.2-G verdict ("grafts are
memory-bound and strictly dominated") has never been tested against a second memory system.
This driver measures the same 29 e2e models on x86 across a thread sweep; ``cpu_rank_report``
then asks whether the Orin's ordering survives and whether the graft penalty grows with
bandwidth pressure.

Run from the repo root in ``.venv``::

    python -m lut.orchestrate.cpu_bench                      # full 29x5 matrix, ~20-40 min
    python -m lut.orchestrate.cpu_bench --rounds 2 --iters 2 # smoke run
    python -m lut.orchestrate.cpu_bench --configs t1,t6      # subset

Resumable: rows already on disk are skipped unless ``--force``.

**Interleaving.** Rounds, not model-at-a-time. Each round times every model once with the order
rotated. A laptop heats up over the run -- the x86 analog of the unlocked-clock failure
``setup_jetson.sh`` prevents, with no equivalent lock available. Model-at-a-time would let that
drift track bench order; since the natural order is by family, the bias would correlate with
architecture family and could manufacture the very effect under test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import TypeVar

import numpy as np
import onnxruntime as ort

from lut.orchestrate.cpu_ort import (
    SYSFS,
    BenchConfig,
    LatencyStats,
    all_cpus,
    apply_affinity,
    make_feeds,
    make_session,
    physical_p_cores,
    summarize,
    time_iterations,
)
from lut.orchestrate.cpu_pairs import CANARY, resolve_pairs

_T = TypeVar("_T")

WARMUP = 5
DEFAULT_ROUNDS = 10
DEFAULT_ITERS = 6
DRIFT_TOL = 0.05


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def build_configs(p_cores: list[int]) -> list[BenchConfig]:
    """The 5-point matrix: a clean P-core sweep plus the practical all-threads number.

    t1..t6 stay on physical P-cores so core type is constant and thread count is an honest
    variable -- a 6->8 step would add *slower* E-cores, changing latency for a reason unrelated
    to bandwidth pressure and confounding the effect under test. all22 is deliberately
    confounded (slower cores AND more pressure) and is excluded from the scaling claim.
    """
    if len(p_cores) < 6:
        raise ValueError(
            f"expected 6 physical P-cores for the sweep, got {len(p_cores)}: {p_cores}"
        )
    sweep = [
        BenchConfig(name=f"t{n}", threads=n, affinity=tuple(p_cores[:n])) for n in (1, 2, 4, 6)
    ]
    return [*sweep, BenchConfig(name="all22", threads=0, affinity=())]


def rotate(seq: list[_T], offset: int) -> list[_T]:
    """Rotate left by offset (wraps).

    Explicit TypeVar rather than PEP-695 ``def rotate[T]`` — that syntax is 3.12+, and the whole
    tree is COPY'd into the AGX image (Python 3.10), where it is a SyntaxError, not a soft failure.
    """
    if not seq:
        return []
    k = offset % len(seq)
    return seq[k:] + seq[:k]


def schedule(names: list[str], rounds: int) -> list[tuple[int, str]]:
    """(round, name) pairs: every model once per round, order rotated between rounds."""
    out: list[tuple[int, str]] = []
    for r in range(rounds):
        out.extend((r, name) for name in rotate(names, r))
    return out


def drift_detected(canary_p50_by_round: dict[int, float], tol: float = DRIFT_TOL) -> bool:
    """True if the canary's p50 moved more than ``tol`` from the first round.

    Not fatal: a drifted run is still readable, it just must not be quoted as clean.
    """
    if len(canary_p50_by_round) < 2:
        return False
    ref = canary_p50_by_round[min(canary_p50_by_round)]
    if ref <= 0:
        return False
    return any(abs(v - ref) / ref > tol for v in canary_p50_by_round.values())


def _cpu_mhz_mean(cfg: BenchConfig, sysfs: Path) -> float:
    """Mean current clock over the CPUs this config actually runs on."""
    cpus = list(cfg.affinity) if cfg.affinity else all_cpus(sysfs)
    freqs: list[float] = []
    for cpu in cpus:
        p = sysfs / f"devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        if p.exists():
            freqs.append(int(p.read_text().strip()) / 1000.0)
    return sum(freqs) / len(freqs) if freqs else 0.0


def _governor(sysfs: Path) -> str:
    p = sysfs / "devices/system/cpu/cpu0/cpufreq/scaling_governor"
    return p.read_text().strip() if p.exists() else "unknown"


def _on_ac(sysfs: Path) -> bool:
    for name in ("ADP1", "AC", "ACAD"):
        p = sysfs / f"class/power_supply/{name}/online"
        if p.exists():
            return p.read_text().strip() == "1"
    return False


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def env_stamp(cfg: BenchConfig, sysfs: Path = SYSFS) -> dict[str, object]:
    """Machine state at row time -- the x86 analog of the Jetson's device_info stamps."""
    return {
        "governor": _governor(sysfs),
        "cpu_mhz_mean": _cpu_mhz_mean(cfg, sysfs),
        "loadavg_1m": os.getloadavg()[0],
        "on_ac": _on_ac(sysfs),
        "ort_version": ort.__version__,
        "cpu_model": _cpu_model(),
    }


def build_row(
    name: str,
    cfg: BenchConfig,
    stats: LatencyStats,
    env: dict[str, object],
    *,
    imgsz: int = 640,
    drift: bool = False,
) -> dict[str, object]:
    """One CPU row -- same shape as data/e2e/*.json, with x86 provenance instead of Jetson's.

    ``source: "x86_ort"`` keeps these unmistakable against ``jetson_trt``: no CPU number can
    ever be mistaken for a device latency in a later claim.
    """
    row: dict[str, object] = {
        "name": name,
        "config": cfg.name,
        "precision": "fp32",
        "imgsz": imgsz,
        "latency_ms": stats.as_dict(),
        "threads": cfg.threads,
        "affinity": list(cfg.affinity),
        "thermal_drift_detected": drift,
        "source": "x86_ort",
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    row.update(env)
    return row


def row_path(out_dir: Path, name: str, cfg_name: str) -> Path:
    return out_dir / f"{name}__{cfg_name}.json"


def _run_config(
    cfg: BenchConfig,
    resolved: list[tuple[str, Path]],
    out_dir: Path,
    rounds: int,
    iters: int,
) -> None:
    """Bench every model under one config, interleaved across rounds.

    Sessions are built once per (config, model) and held for the config's duration: rebuilding
    per round would spend more time in ORT graph optimisation than in measurement.
    """
    apply_affinity(cfg)
    names = [n for n, _ in resolved]
    paths = dict(resolved)

    print(f"[cpu_bench] {cfg.name}: building {len(names)} sessions ...", flush=True)
    sessions: dict[str, ort.InferenceSession] = {}
    feeds: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        sessions[name] = make_session(paths[name], cfg)
        feeds[name] = make_feeds(sessions[name])
        time_iterations(sessions[name], feeds[name], WARMUP)

    samples: dict[str, list[float]] = {n: [] for n in names}
    canary_by_round: dict[int, float] = {}
    t_start = time.perf_counter()

    for r, name in schedule(names, rounds):
        got = time_iterations(sessions[name], feeds[name], iters)
        samples[name].extend(got)
        if name == CANARY:
            canary_by_round[r] = summarize(got).p50
        if r == 0 and name == rotate(names, 0)[-1]:
            per_round = time.perf_counter() - t_start
            eta = per_round * (rounds - 1) / 60.0
            print(
                f"[cpu_bench] {cfg.name}: round 0 took {per_round:.0f}s, ETA {eta:.1f} min",
                flush=True,
            )

    drift = drift_detected(canary_by_round)
    if drift:
        print(
            f"[cpu_bench] WARNING {cfg.name}: canary drifted >{DRIFT_TOL:.0%} -- "
            "rows stamped thermal_drift_detected",
            flush=True,
        )

    env = env_stamp(cfg)
    for name in names:
        row = build_row(name, cfg, summarize(samples[name]), env, drift=drift)
        row_path(out_dir, name, cfg.name).write_text(json.dumps(row, indent=2) + "\n")
    print(f"[cpu_bench] {cfg.name}: wrote {len(names)} rows", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU cross-device rank check bench")
    ap.add_argument("--out", type=Path, default=root_dir() / "data" / "cpu")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--configs", type=str, default="", help="comma list, e.g. t1,t6")
    ap.add_argument("--force", action="store_true", help="re-bench rows already on disk")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = resolve_pairs(root_dir())  # hard-fails listing every missing ONNX
    configs = build_configs(physical_p_cores())
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            print(f"error: no configs matched {sorted(wanted)}", flush=True)
            return 2

    print(
        f"[cpu_bench] {len(pairs)} models x {len(configs)} configs, "
        f"{args.rounds} rounds x {args.iters} iters -> n={args.rounds * args.iters}/cell",
        flush=True,
    )

    for cfg in configs:
        resolved = [
            (p.jetson_name, path)
            for p, path in pairs
            if args.force or not row_path(args.out, p.jetson_name, cfg.name).exists()
        ]
        if not resolved:
            print(f"[cpu_bench] {cfg.name}: all rows present, skipping", flush=True)
            continue
        _run_config(cfg, resolved, args.out, args.rounds, args.iters)

    print(f"[cpu_bench] done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
