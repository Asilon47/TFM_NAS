"""Time an ONNX model on this machine's CPU under a pinned thread configuration.

The timing core for the cross-device rank check (see
``docs/superpowers/specs/2026-07-15-cpu-cross-device-rank-check-design.md``). Deliberately
dumb: it knows nothing about the sweep, the model frontier, or the Jetson. That keeps it
testable in milliseconds instead of requiring a 25-minute run.

Lives in ``lut/orchestrate/`` rather than ``lut/bench/`` because it runs locally under the
repo's own tooling; ``lut/bench/`` is excluded from ruff+mypy since it deploys to the Jetson.

**Hyperthread trap.** On this Core Ultra 9 185H the sibling pairs are {0,5} {1,2} {3,4} {6,7}
{8,9} {10,11} -- so ``taskset -c 0-5`` would occupy only THREE physical cores with every
hyperthread contended, measuring HT contention and calling it thread scaling.
``physical_p_cores()`` exists to avoid exactly that.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SYSFS = Path("/sys")

#: ORT's intra-op workers spin-wait after ``run`` returns instead of sleeping, which shaves
#: wake-up latency when ONE session is called in a loop -- the server case it was tuned for.
#: The rank check violates that assumption: it rotates ~29 live sessions, so every model's run
#: collides with the previous models' pools still burning the same pinned cores. Measured on the
#: 185H at t6 with 29 sessions held and interleaved access: **564.5 ms spinning vs 30.4 ms not**
#: -- an 18.6x artifact that grows with thread count, i.e. shaped exactly like the bandwidth
#: effect under test. Never re-enable this for an interleaved bench.
ALLOW_SPINNING = "0"


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux cpulist ("0-11", "0,5", "12-19,20-21") into sorted CPU ids."""
    out: set[int] = set()
    for raw in spec.strip().split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _read(path: Path) -> str:
    return path.read_text()


def all_cpus(sysfs: Path = SYSFS) -> list[int]:
    """Every online logical CPU."""
    return parse_cpu_list(_read(sysfs / "devices/system/cpu/online"))


def physical_p_cores(sysfs: Path = SYSFS) -> list[int]:
    """One logical CPU per physical performance core, ascending.

    On Intel hybrid parts ``devices/cpu_core/cpus`` lists the P-core logical CPUs; each
    physical core contributes two of them (SMT). We keep the lowest id of each sibling group.
    Non-hybrid machines have no ``cpu_core`` node -- fall back to all online CPUs, deduped by
    sibling group just the same.
    """
    core_node = sysfs / "devices/cpu_core/cpus"
    candidates = parse_cpu_list(_read(core_node)) if core_node.exists() else all_cpus(sysfs)

    seen: set[int] = set()
    out: list[int] = []
    for cpu in candidates:
        if cpu in seen:
            continue
        sib_path = sysfs / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        sibs = parse_cpu_list(_read(sib_path)) if sib_path.exists() else [cpu]
        seen.update(sibs)
        out.append(min(sibs))
    return sorted(out)


@dataclass(frozen=True)
class BenchConfig:
    """One thread configuration.

    ``threads=0`` => ORT default pool; ``affinity=()`` => unpinned.
    """

    name: str
    threads: int
    affinity: tuple[int, ...]


@dataclass(frozen=True)
class LatencyStats:
    """Latency distribution for one (model, config) cell -- same shape as data/e2e rows."""

    mean: float
    std: float
    p50: float
    p95: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "p50": self.p50, "p95": self.p95, "n": self.n}


def summarize(samples_ms: list[float]) -> LatencyStats:
    """Pool per-iteration timings into the row's distribution.

    p50 is the headline: it is what data/e2e rows report, and it is robust to the occasional
    scheduler-preemption outlier that a laptop will produce and a clock-locked Jetson will not.
    """
    if not samples_ms:
        raise ValueError("no samples to summarize")
    ordered = sorted(samples_ms)
    return LatencyStats(
        mean=statistics.fmean(ordered),
        std=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        p50=statistics.median(ordered),
        p95=float(np.percentile(np.asarray(ordered), 95)),
        n=len(ordered),
    )


def apply_affinity(cfg: BenchConfig, sysfs: Path = SYSFS) -> None:
    """Pin this process before session creation, so ORT's pool inherits the mask.

    Order matters: ORT builds its intra-op thread pool when the session is constructed, so
    setting affinity afterwards leaves the worker threads free to roam onto E-cores.
    """
    target = set(cfg.affinity) if cfg.affinity else set(all_cpus(sysfs))
    os.sched_setaffinity(0, target)


def make_session(onnx_path: Path, cfg: BenchConfig) -> ort.InferenceSession:
    """Build a CPU-EP session for one config. Call ``apply_affinity(cfg)`` first.

    Spinning is disabled (see ``ALLOW_SPINNING``): it is the difference between a valid
    interleaved measurement and an 18.6x thread-contention artifact.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", ALLOW_SPINNING)
    if cfg.threads:
        so.intra_op_num_threads = cfg.threads
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def make_feeds(
    session: ort.InferenceSession, imgsz: int = 640, seed: int = 0
) -> dict[str, np.ndarray]:
    """A fixed NCHW fp32 input, reused for every iteration and every model.

    Conv nets have no data-dependent control flow and these exports are network-only (no NMS),
    so a constant input is sound for latency and removes a noise source.
    """
    name = session.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    tensor = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
    return {name: tensor}


def time_iterations(
    session: ort.InferenceSession, feeds: dict[str, np.ndarray], k: int
) -> list[float]:
    """Time k forward passes; return per-iteration milliseconds."""
    out: list[float] = []
    for _ in range(k):
        t0 = time.perf_counter_ns()
        session.run(None, feeds)
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out
