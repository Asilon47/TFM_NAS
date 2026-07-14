# CPU Cross-Device Rank Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the Jetson Orin Nano's latency ranking of 28 pose architectures survives on an x86 CPU, and whether the OFA graft's memory-bound penalty grows with bandwidth pressure.

**Architecture:** Three modules in `lut/orchestrate/`, mirroring the existing timing-core / driver / analysis split. `cpu_ort.py` is a dumb, testable timing core (ONNX + thread config → latency stats). `cpu_bench.py` owns methodology: a round-robin interleaved schedule over a 28×5 model/config matrix, writing resumable JSON rows to `data/cpu/`. `cpu_rank_report.py` joins those against the measured Jetson rows in `data/e2e/` and emits Spearman/Kendall per thread count plus the graft-penalty-vs-threads curve.

**Tech Stack:** Python 3.12, onnxruntime 1.27.0 (CPUExecutionProvider), numpy 2.5.0, scipy 1.18.0 (`spearmanr`/`kendalltau`), onnx 1.22.0 (param counting), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-07-15-cpu-cross-device-rank-check-design.md`

## Global Constraints

- **Venv:** `.venv` (the LUT venv). Invoke as `.venv/bin/python -m <module>` from repo root. Never `pip install` into it.
- **Module location:** all three modules in `lut/orchestrate/`. **Never** `lut/bench/` — `pyproject.toml` excludes that from ruff *and* mypy because it deploys to the Jetson under Python 3.10.
- **Lint/type:** ruff (line-length 100, `select = ["E","F","W","I","B","UP"]`, target py312) and mypy both cover `lut/`. Every module must pass `bash scripts/check.sh`.
- **`from __future__ import annotations`** at the top of every new module (repo-wide convention).
- **Precision:** fp32 only. Never emit an fp16 CPU row.
- **`source` field:** every emitted row carries `"source": "x86_ort"`. Never `jetson_trt`.
- **Data is gitignored:** `data/cpu/*.json` must never be committed. Only code, tests, and docs are committed.
- **Model files are gitignored:** tests touching `models/**/*.onnx` must skip when absent (CI has no weights), following the existing `data/lut.jsonl` skip convention.
- **Commit after every task.** No `Co-Authored-By:` or "Generated with Claude Code" trailers.
- **Golden LUT hashes** in `tests/test_row_key.py` are untouchable — this work must not change them.

---

### Task 1: Timing core — CPU topology + latency stats

**Files:**
- Create: `lut/orchestrate/cpu_ort.py`
- Test: `tests/test_cpu_ort.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces:
  - `parse_cpu_list(spec: str) -> list[int]`
  - `physical_p_cores(sysfs: Path = SYSFS) -> list[int]`
  - `all_cpus(sysfs: Path = SYSFS) -> list[int]`
  - `BenchConfig(name: str, threads: int, affinity: tuple[int, ...])` — frozen dataclass; `threads=0` means "ORT default pool", `affinity=()` means "no pinning"
  - `LatencyStats(mean: float, std: float, p50: float, p95: float, n: int)` — frozen dataclass with `.as_dict() -> dict[str, float | int]`
  - `summarize(samples_ms: list[float]) -> LatencyStats`
  - `apply_affinity(cfg: BenchConfig, sysfs: Path = SYSFS) -> None`
  - `make_session(onnx_path: Path, cfg: BenchConfig) -> ort.InferenceSession`
  - `make_feeds(session: ort.InferenceSession, imgsz: int = 640, seed: int = 0) -> dict[str, np.ndarray]`
  - `time_iterations(session, feeds, k: int) -> list[float]` — returns per-iteration ms

- [ ] **Step 1: Write the failing topology + stats tests**

Create `tests/test_cpu_ort.py`:

```python
"""Contract tests for the CPU timing core.

The topology tests are the important ones: this laptop's hyperthread siblings
are {0,5} {1,2} {3,4} {6,7} {8,9} {10,11}, so the "obvious" taskset -c 0-5 lands
on THREE physical cores with every thread contended. A regression here would
silently measure HT contention and report it as thread scaling.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lut.orchestrate.cpu_ort import (
    BenchConfig,
    LatencyStats,
    all_cpus,
    parse_cpu_list,
    physical_p_cores,
    summarize,
)

# The real Core Ultra 9 185H layout, as probed from /sys on 2026-07-15.
SIBLINGS_185H = {
    0: "0,5", 1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "0,5",
    6: "6-7", 7: "6-7", 8: "8-9", 9: "8-9", 10: "10-11", 11: "10-11",
}


@pytest.fixture
def fake_sysfs(tmp_path: Path) -> Path:
    """A minimal /sys mirroring the 185H: 6 P-cores (SMT) + 10 E-cores."""
    (tmp_path / "devices/cpu_core").mkdir(parents=True)
    (tmp_path / "devices/cpu_core/cpus").write_text("0-11\n")
    (tmp_path / "devices/system/cpu").mkdir(parents=True)
    (tmp_path / "devices/system/cpu/online").write_text("0-21\n")
    for cpu, sibs in SIBLINGS_185H.items():
        topo = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(sibs + "\n")
    return tmp_path


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0-11", list(range(12))),
        ("0,5", [0, 5]),
        ("1-2", [1, 2]),
        ("12-19,20-21", list(range(12, 22))),
        ("3", [3]),
        ("0-21\n", list(range(22))),
    ],
)
def test_parse_cpu_list(spec: str, expected: list[int]) -> None:
    assert parse_cpu_list(spec) == expected


def test_physical_p_cores_picks_one_per_core(fake_sysfs: Path) -> None:
    """One logical CPU per physical P-core -- NOT 0-5."""
    assert physical_p_cores(fake_sysfs) == [0, 1, 3, 6, 8, 10]


def test_physical_p_cores_never_returns_a_sibling_pair(fake_sysfs: Path) -> None:
    """No two returned CPUs may share a physical core (the 0-5 trap)."""
    got = physical_p_cores(fake_sysfs)
    for cpu in got:
        sibs = set(parse_cpu_list(SIBLINGS_185H[cpu]))
        assert sibs & set(got) == {cpu}, f"cpu{cpu} shares a core with another selected cpu"


def test_physical_p_cores_falls_back_when_not_hybrid(tmp_path: Path) -> None:
    """Non-hybrid CPUs have no devices/cpu_core -- fall back to online CPUs."""
    (tmp_path / "devices/system/cpu").mkdir(parents=True)
    (tmp_path / "devices/system/cpu/online").write_text("0-3\n")
    for cpu in range(4):
        topo = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(f"{cpu}\n")
    assert physical_p_cores(tmp_path) == [0, 1, 2, 3]


def test_all_cpus(fake_sysfs: Path) -> None:
    assert all_cpus(fake_sysfs) == list(range(22))


def test_summarize_computes_order_statistics() -> None:
    stats = summarize([10.0, 12.0, 11.0, 13.0, 14.0])
    assert stats.n == 5
    assert stats.mean == pytest.approx(12.0)
    assert stats.p50 == pytest.approx(12.0)
    assert stats.p95 == pytest.approx(14.0, abs=0.5)
    assert stats.std > 0


def test_summarize_single_sample_has_zero_std() -> None:
    stats = summarize([7.5])
    assert stats.n == 1
    assert stats.std == 0.0
    assert stats.p50 == pytest.approx(7.5)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no samples"):
        summarize([])


def test_latency_stats_as_dict_matches_e2e_schema() -> None:
    """Rows must carry the same keys as data/e2e/*.json so tooling reads both."""
    d = LatencyStats(mean=1.0, std=0.1, p50=0.9, p95=1.2, n=60).as_dict()
    assert set(d) == {"mean", "std", "p50", "p95", "n"}
    assert d["n"] == 60


def test_bench_config_is_hashable_and_frozen() -> None:
    cfg = BenchConfig(name="t4", threads=4, affinity=(0, 1, 3, 6))
    assert hash(cfg)
    with pytest.raises(Exception):
        cfg.threads = 8  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cpu_ort.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lut.orchestrate.cpu_ort'`

- [ ] **Step 3: Write the timing core**

Create `lut/orchestrate/cpu_ort.py`:

```python
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


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux cpulist ("0-11", "0,5", "12-19,20-21") into sorted CPU ids."""
    out: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
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
    """One thread configuration. ``threads=0`` => ORT default pool; ``affinity=()`` => unpinned."""

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
    """Build a CPU-EP session for one config. Call ``apply_affinity(cfg)`` first."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cpu_ort.py -v`
Expected: PASS — 12 passed

- [ ] **Step 5: Verify against the real machine**

Run:
```bash
.venv/bin/python -c "
from lut.orchestrate.cpu_ort import physical_p_cores, all_cpus
print('P-cores:', physical_p_cores())
print('all:', len(all_cpus()), 'cpus')
"
```
Expected: `P-cores: [0, 1, 3, 6, 8, 10]` and `all: 22 cpus`. If the P-core list differs from this, STOP — the machine is not the one the spec profiled and §2 of the spec needs re-probing.

- [ ] **Step 6: Lint, type-check, commit**

```bash
bash scripts/check.sh -m "not slow"
git add lut/orchestrate/cpu_ort.py tests/test_cpu_ort.py
git commit -m "cpu-bench: ONNX Runtime timing core with hybrid-CPU-aware affinity

Picks one logical CPU per physical P-core ({0,1,3,6,8,10} on the 185H)
rather than the 0-5 range that would land on three HT-contended cores."
```

---

### Task 2: The pair map

**Files:**
- Create: `lut/orchestrate/cpu_pairs.py`
- Test: `tests/test_cpu_pairs.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Pair(jetson_name: str, onnx: str, family: str)` — frozen dataclass; `onnx` is a repo-relative path
  - `PAIRS: tuple[Pair, ...]` — the 28 declared pairs
  - `REFERENCE_FAMILIES: frozenset[str]` — `{"dense", "prune", "baseline"}`
  - `GRAFT_FAMILIES: frozenset[str]` — `{"graft", "graft_pruned"}`
  - `CANARY: str` — `"baseline_recheck_640"`
  - `resolve_pairs(root: Path) -> list[tuple[Pair, Path]]` — validates every ONNX exists; raises `FileNotFoundError` listing all misses

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cpu_pairs.py`:

```python
"""Contract tests for the Jetson<->ONNX pair map.

Jetson row names do not match ONNX filenames and cannot be derived (baseline_recheck_640 is
yolo11n_pose_640.onnx; dense_ctrl_n_640 is dense_w25_640.onnx). A mis-pair would produce a
corrupt-but-plausible Spearman, so the map is declared data and tested as data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lut.orchestrate.cpu_pairs import (
    CANARY,
    GRAFT_FAMILIES,
    PAIRS,
    REFERENCE_FAMILIES,
    resolve_pairs,
)

MODELS = Path(__file__).resolve().parents[1] / "models"
E2E = Path(__file__).resolve().parents[1] / "data" / "e2e"


def test_pair_count() -> None:
    assert len(PAIRS) == 28


def test_jetson_names_unique() -> None:
    names = [p.jetson_name for p in PAIRS]
    assert len(set(names)) == len(names)


def test_onnx_paths_unique() -> None:
    """Two rows mapped to one ONNX would silently duplicate a point in the correlation."""
    paths = [p.onnx for p in PAIRS]
    assert len(set(paths)) == len(paths)


def test_no_backbone_only_models() -> None:
    """Backbone rows are a different network scope -- mixing them is the retired-claim error."""
    for p in PAIRS:
        assert "backbone" not in p.jetson_name, p.jetson_name
        assert "_bb_" not in p.onnx, p.onnx


def test_canary_is_in_the_map() -> None:
    assert CANARY in {p.jetson_name for p in PAIRS}


def test_families_are_known() -> None:
    known = {"baseline", "anchor", "dense", "dense_nas", "prune", "graft", "graft_pruned"}
    assert {p.family for p in PAIRS} <= known


def test_reference_set_size_and_range() -> None:
    """The OLS reference is dense+prune+baseline = 15 models (anchor excluded: leverage)."""
    ref = [p for p in PAIRS if p.family in REFERENCE_FAMILIES]
    assert len(ref) == 15
    assert "anchor" not in REFERENCE_FAMILIES


def test_graft_families() -> None:
    grafts = [p for p in PAIRS if p.family in GRAFT_FAMILIES]
    assert len(grafts) == 10


@pytest.mark.skipif(not MODELS.exists(), reason="models/ is gitignored; absent in CI")
def test_every_onnx_exists() -> None:
    resolved = resolve_pairs(MODELS.parent)
    assert len(resolved) == 28
    for _, path in resolved:
        assert path.is_file(), path


@pytest.mark.skipif(not E2E.exists(), reason="data/ is gitignored; absent in CI")
def test_every_jetson_row_exists() -> None:
    for p in PAIRS:
        assert (E2E / f"{p.jetson_name}.json").is_file(), p.jetson_name


def test_resolve_pairs_raises_listing_all_misses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        resolve_pairs(tmp_path)
    # Must list misses, not fail on the first -- a short pair list weakens rho silently.
    assert "28" in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cpu_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lut.orchestrate.cpu_pairs'`

- [ ] **Step 3: Write the pair map**

Create `lut/orchestrate/cpu_pairs.py`:

```python
"""The declared Jetson-row <-> local-ONNX pairing for the CPU rank check.

Jetson row names in ``data/e2e/`` do not match the ONNX filenames under ``models/`` and cannot
be derived from them: ``baseline_recheck_640`` is ``yolo11n_pose_640.onnx``, ``dense_ctrl_n_640``
is ``dense_w25_640.onnx``, ``winner_v1_e2e_640`` is ``winner_v1_noneck_e2e_640.onnx``. Any
auto-matching heuristic would mis-pair models and still produce a plausible-looking correlation,
so the map is explicit, reviewed data.

Exclusions are deliberate (see spec §4): ``*_backbone_640`` rows are a different network scope;
``fallback_idx{3,11}`` have no local ONNX; ``recover_graft_r{40,60}_640.onnx`` are left out
because ``screen_prune_graft/graft_prune_r{40,60}_e2e_640.onnx`` are the artifacts whose Jetson
latencies (11.81 / 9.00 ms) match the CP 6.2-G rungs recorded in ``models/README.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANARY = "baseline_recheck_640"

#: Families whose latency/params relationship the project already trusts -- the OLS reference.
#: The anchor (yolo11s, 9.7M params) is excluded: at 2.5x the next-largest reference model it
#: would dominate the fit as a leverage point. dense_nas is excluded as a distinct design process.
REFERENCE_FAMILIES = frozenset({"dense", "prune", "baseline"})

#: The families under test.
GRAFT_FAMILIES = frozenset({"graft", "graft_pruned"})


@dataclass(frozen=True)
class Pair:
    """One measured Jetson row and its local ONNX counterpart."""

    jetson_name: str
    onnx: str
    family: str


PAIRS: tuple[Pair, ...] = (
    Pair("baseline_recheck_640", "models/baseline/yolo11n_pose_640.onnx", "baseline"),
    Pair("yolo11s_pose_640", "models/anchor/yolo11s_pose_640.onnx", "anchor"),
    Pair("dense_ctrl_n_640", "models/dense_scaled/dense_w25_640.onnx", "dense"),
    Pair("dense_d33_w20_640", "models/dense_scaled/dense_w20_640.onnx", "dense"),
    Pair("dense_d50_w15_640", "models/dense_scaled/dense_w15_640.onnx", "dense"),
    Pair("dense_w13_640", "models/dense_scaled/dense_w13_640.onnx", "dense"),
    Pair("dense_w18_640", "models/dense_scaled/dense_w18_640.onnx", "dense"),
    Pair("dense_w22_640", "models/dense_scaled/dense_w22_640.onnx", "dense"),
    Pair("dense_w30_640", "models/dense_scaled/dense_w30_640.onnx", "dense"),
    Pair("densenas_s31_640", "models/dense_nas/dense_s31-40-40-40-13_o100_640.onnx", "dense_nas"),
    Pair("densenas_s39_640", "models/dense_nas/dense_s39-40-38-38-14_o100_640.onnx", "dense_nas"),
    Pair("densenas_s40_640", "models/dense_nas/dense_s40-38-39-36-13_o100_640.onnx", "dense_nas"),
    Pair("prune_base_r10_640", "models/pruned_baseline/prune_r10_640.onnx", "prune"),
    Pair("prune_base_r15_640", "models/pruned_baseline/prune_r15_640.onnx", "prune"),
    Pair("prune_base_r20_640", "models/pruned_baseline/prune_r20_640.onnx", "prune"),
    Pair("prune_base_r30_640", "models/pruned_baseline/prune_r30_640.onnx", "prune"),
    Pair("prune_base_r35_640", "models/pruned_baseline/prune_r35_640.onnx", "prune"),
    Pair("prune_base_r45_640", "models/pruned_baseline/prune_r45_640.onnx", "prune"),
    Pair("prune_base_r55_640", "models/pruned_baseline/prune_r55_640.onnx", "prune"),
    Pair("winner_v1_e2e_640", "models/graft/winner_v1_noneck_e2e_640.onnx", "graft"),
    Pair("winner_v1_v2topdown_e2e_640", "models/graft/winner_v1_v2topdown_e2e_640.onnx", "graft"),
    Pair("winner_v1_v3pan_e2e_640", "models/graft/winner_v1_v3pan_e2e_640.onnx", "graft"),
    Pair(
        "graft_prune_r20_e2e_640",
        "models/screen_prune_graft/graft_prune_r20_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_prune_r40_e2e_640",
        "models/screen_prune_graft/graft_prune_r40_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_prune_r60_e2e_640",
        "models/screen_prune_graft/graft_prune_r60_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_r50_gtay_640",
        "models/graft_pruned/recover_graft_r50_gtay_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_r60_gtay_640",
        "models/graft_pruned/recover_graft_r60_gtay_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_halp_9p0_640",
        "models/graft_pruned/recover_graft_halp_fp32_9p0_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_halp_10p4_640",
        "models/graft_pruned/recover_graft_halp_fp32_10p4_640.onnx",
        "graft_pruned",
    ),
)


def resolve_pairs(root: Path) -> list[tuple[Pair, Path]]:
    """Resolve every pair's ONNX against ``root``; hard-fail listing ALL misses.

    Fails before any timing rather than benching a silently-short set: a missing model would
    weaken the correlation without announcing itself.
    """
    resolved: list[tuple[Pair, Path]] = []
    missing: list[str] = []
    for pair in PAIRS:
        path = root / pair.onnx
        if path.is_file():
            resolved.append((pair, path))
        else:
            missing.append(pair.onnx)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(PAIRS)} ONNX files missing under {root}:\n  "
            + "\n  ".join(missing)
        )
    return resolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cpu_pairs.py -v`
Expected: PASS — 11 passed. If `test_every_onnx_exists` or `test_every_jetson_row_exists` FAILS (rather than skips), the map has a wrong path — fix the map, not the test.

- [ ] **Step 5: Lint, type-check, commit**

```bash
bash scripts/check.sh -m "not slow"
git add lut/orchestrate/cpu_pairs.py tests/test_cpu_pairs.py
git commit -m "cpu-bench: declared Jetson<->ONNX pair map (28 e2e models)

Names are not derivable (baseline_recheck_640 -> yolo11n_pose_640.onnx),
so the pairing is reviewed data, not a heuristic that could mis-pair
into a plausible-looking correlation."
```

---

### Task 3: Driver — interleaved schedule, rows, resume, canary

**Files:**
- Create: `lut/orchestrate/cpu_bench.py`
- Test: `tests/test_cpu_bench.py`

**Interfaces:**
- Consumes: `cpu_ort.{BenchConfig, LatencyStats, physical_p_cores, all_cpus, apply_affinity, make_session, make_feeds, time_iterations, summarize}`, `cpu_pairs.{PAIRS, CANARY, resolve_pairs}`
- Produces:
  - `build_configs(p_cores: list[int]) -> list[BenchConfig]` — `[t1, t2, t4, t6, all22]`
  - `rotate(seq: list[T], offset: int) -> list[T]`
  - `schedule(names: list[str], rounds: int) -> list[tuple[int, str]]` — `(round_index, name)` in execution order
  - `drift_detected(canary_p50_by_round: dict[int, float], tol: float = 0.05) -> bool`
  - `env_stamp(cfg: BenchConfig, sysfs: Path) -> dict[str, object]`
  - `build_row(name, cfg, stats, env, *, imgsz=640) -> dict[str, object]`
  - `row_path(out_dir: Path, name: str, cfg_name: str) -> Path`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cpu_bench.py`:

```python
"""Contract tests for the CPU bench driver.

The interleave tests matter most: model-at-a-time benching would let thermal drift on a laptop
become a systematic bias that tracks bench order -- and since the natural order is by family,
that bias would correlate with architecture family and could manufacture the very effect the
experiment tests for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lut.orchestrate.cpu_bench import (
    build_configs,
    build_row,
    drift_detected,
    root_dir,
    rotate,
    row_path,
    schedule,
)
from lut.orchestrate.cpu_ort import BenchConfig, LatencyStats

P_CORES = [0, 1, 3, 6, 8, 10]


def test_build_configs_shape() -> None:
    cfgs = build_configs(P_CORES)
    assert [c.name for c in cfgs] == ["t1", "t2", "t4", "t6", "all22"]


def test_build_configs_pin_one_thread_per_physical_core() -> None:
    cfgs = {c.name: c for c in build_configs(P_CORES)}
    assert cfgs["t1"].affinity == (0,)
    assert cfgs["t2"].affinity == (0, 1)
    assert cfgs["t4"].affinity == (0, 1, 3, 6)
    assert cfgs["t6"].affinity == (0, 1, 3, 6, 8, 10)
    for name in ("t1", "t2", "t4", "t6"):
        assert cfgs[name].threads == len(cfgs[name].affinity)


def test_all22_is_unpinned_and_default_pool() -> None:
    """The practical number: ORT's own pool, no mask. Confounded by design, labelled as such."""
    cfgs = {c.name: c for c in build_configs(P_CORES)}
    assert cfgs["all22"].affinity == ()
    assert cfgs["all22"].threads == 0


def test_build_configs_rejects_too_few_cores() -> None:
    with pytest.raises(ValueError, match="6 physical P-cores"):
        build_configs([0, 1, 3])


def test_rotate() -> None:
    assert rotate([1, 2, 3, 4], 0) == [1, 2, 3, 4]
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate([1, 2, 3, 4], 5) == [2, 3, 4, 1]


def test_schedule_times_every_model_once_per_round() -> None:
    names = ["a", "b", "c"]
    sched = schedule(names, rounds=4)
    assert len(sched) == 12
    for r in range(4):
        in_round = [n for (rr, n) in sched if rr == r]
        assert sorted(in_round) == ["a", "b", "c"], f"round {r} is not a full pass"


def test_schedule_rotates_order_between_rounds() -> None:
    """Rotation is what turns thermal drift into common-mode noise instead of order bias."""
    names = ["a", "b", "c"]
    sched = schedule(names, rounds=3)
    firsts = [next(n for (rr, n) in sched if rr == r) for r in range(3)]
    assert firsts == ["a", "b", "c"]


def test_schedule_zero_rounds_is_empty() -> None:
    assert schedule(["a"], rounds=0) == []


def test_drift_detected_flags_a_hot_laptop() -> None:
    assert drift_detected({0: 10.0, 1: 10.1, 2: 11.0}) is True


def test_drift_detected_passes_a_stable_run() -> None:
    assert drift_detected({0: 10.0, 1: 10.1, 2: 10.2}) is False


def test_drift_detected_uses_round_zero_as_reference() -> None:
    """Drift is measured against the cold first round, not the min."""
    assert drift_detected({0: 10.0, 1: 10.4}, tol=0.05) is False
    assert drift_detected({0: 10.0, 1: 10.6}, tol=0.05) is True


def test_drift_detected_needs_two_rounds() -> None:
    assert drift_detected({0: 10.0}) is False
    assert drift_detected({}) is False


def test_build_row_matches_e2e_schema() -> None:
    cfg = BenchConfig(name="t4", threads=4, affinity=(0, 1, 3, 6))
    stats = LatencyStats(mean=42.0, std=0.5, p50=41.9, p95=43.0, n=60)
    env = {
        "governor": "powersave",
        "cpu_mhz_mean": 3800.0,
        "loadavg_1m": 0.4,
        "on_ac": True,
        "ort_version": "1.27.0",
        "cpu_model": "Intel(R) Core(TM) Ultra 9 185H",
    }
    row = build_row("prune_base_r20_640", cfg, stats, env)

    assert row["name"] == "prune_base_r20_640"
    assert row["config"] == "t4"
    assert row["precision"] == "fp32"
    assert row["imgsz"] == 640
    assert row["threads"] == 4
    assert row["affinity"] == [0, 1, 3, 6]
    assert row["source"] == "x86_ort"
    assert row["latency_ms"] == stats.as_dict()
    assert row["thermal_drift_detected"] is False
    assert "timestamp" in row


def test_build_row_never_emits_fp16() -> None:
    """ORT CPU fp16 is emulated -- an fp16 row would measure emulation, not the model."""
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    row = build_row("x", cfg, stats, {})
    assert row["precision"] == "fp32"


def test_build_row_is_json_serializable() -> None:
    cfg = BenchConfig(name="t1", threads=1, affinity=(0,))
    stats = LatencyStats(mean=1.0, std=0.0, p50=1.0, p95=1.0, n=1)
    json.dumps(build_row("x", cfg, stats, {}))


def test_row_path(tmp_path: Path) -> None:
    assert row_path(tmp_path, "dense_w13_640", "t4") == tmp_path / "dense_w13_640__t4.json"


def test_root_dir_is_the_repo() -> None:
    assert (root_dir() / "pyproject.toml").is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cpu_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lut.orchestrate.cpu_bench'`

- [ ] **Step 3: Write the driver**

Create `lut/orchestrate/cpu_bench.py`:

```python
"""Bench the pose frontier on this machine's CPU -- the cross-device rank check.

Every latency in this repo is a Jetson/TensorRT number, so the CP 6.2-G verdict ("grafts are
memory-bound and strictly dominated") has never been tested against a second memory system.
This driver measures the same 28 e2e models on x86 across a thread sweep; ``cpu_rank_report``
then asks whether the Orin's ordering survives and whether the graft penalty grows with
bandwidth pressure.

Run from the repo root in ``.venv``::

    python -m lut.orchestrate.cpu_bench                      # full 28x5 matrix, ~20-40 min
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

T = TypeVar("T")

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
    return [
        BenchConfig(name=f"t{n}", threads=n, affinity=tuple(p_cores[:n]))
        for n in (1, 2, 4, 6)
    ] + [BenchConfig(name="all22", threads=0, affinity=())]


def rotate(seq: list[T], offset: int) -> list[T]:
    """Rotate left by offset (wraps)."""
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
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    feeds: dict[str, dict[str, object]] = {}
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
        if name == names[-1] and r == 0:
            per_round = time.perf_counter() - t_start
            eta = per_round * (rounds - 1) / 60.0
            print(f"[cpu_bench] {cfg.name}: round 0 took {per_round:.0f}s, ETA {eta:.1f} min",
                  flush=True)

    drift = drift_detected(canary_by_round)
    if drift:
        print(f"[cpu_bench] WARNING {cfg.name}: canary drifted >{DRIFT_TOL:.0%} -- "
              "rows stamped thermal_drift_detected", flush=True)

    env = env_stamp(cfg)
    for name in names:
        row = build_row(name, cfg, summarize(samples[name]), env, drift=drift)
        row_path(out_dir, name, cfg.name).write_text(json.dumps(row, indent=2) + "\n")
    print(f"[cpu_bench] {cfg.name}: wrote {len(names)} rows", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=root_dir() / "data" / "cpu")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--configs", type=str, default="", help="comma list, e.g. t1,t6")
    ap.add_argument("--force", action="store_true", help="re-bench rows already on disk")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = resolve_pairs(root_dir())          # hard-fails listing every missing ONNX
    configs = build_configs(physical_p_cores())
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            print(f"error: no configs matched {sorted(wanted)}", flush=True)
            return 2

    print(f"[cpu_bench] {len(pairs)} models x {len(configs)} configs, "
          f"{args.rounds} rounds x {args.iters} iters -> n={args.rounds * args.iters}/cell",
          flush=True)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cpu_bench.py -v`
Expected: PASS — 17 passed

- [ ] **Step 5: Smoke-run the driver end-to-end**

Run:
```bash
.venv/bin/python -m lut.orchestrate.cpu_bench \
  --out /tmp/claude-1000/-home-asil-Desktop-TFM-NAS/3a93b1ab-5075-4ada-8aff-8c44208dc67e/scratchpad/cpu_smoke \
  --configs t1 --rounds 2 --iters 1
```
Expected: prints `28 models x 1 configs`, builds sessions, writes 28 rows. Inspect one:
```bash
cat /tmp/claude-1000/-home-asil-Desktop-TFM-NAS/3a93b1ab-5075-4ada-8aff-8c44208dc67e/scratchpad/cpu_smoke/baseline_recheck_640__t1.json
```
Expected: valid JSON with `"source": "x86_ort"`, `"config": "t1"`, `"threads": 1`, `"affinity": [0]`, `"n": 2`, and a plausible `latency_ms.p50` (tens to low hundreds of ms at 1 thread).

**If session build for all 28 models exhausts RAM**, reduce scope: add `--max-sessions` that builds/destroys per round instead of holding. Record the observed RSS in the commit message either way.

- [ ] **Step 6: Lint, type-check, commit**

```bash
bash scripts/check.sh -m "not slow"
git add lut/orchestrate/cpu_bench.py tests/test_cpu_bench.py
git commit -m "cpu-bench: interleaved driver for the 28x5 CPU matrix

Round-robin with rotated order so thermal drift on an unlockable laptop
stays common-mode instead of tracking bench order (which follows family,
and would manufacture the effect under test). Canary flags >5% drift."
```

---

### Task 4: Analysis — rank correlation + graft penalty

**Files:**
- Create: `lut/orchestrate/cpu_rank_report.py`
- Test: `tests/test_cpu_rank_report.py`

**Interfaces:**
- Consumes: `cpu_pairs.{PAIRS, REFERENCE_FAMILIES, GRAFT_FAMILIES}`, `cpu_bench.root_dir`
- Produces:
  - `count_params(onnx_path: Path) -> int`
  - `load_jetson(e2e_dir: Path) -> dict[str, float]` — name → fp32 p50
  - `load_cpu(cpu_dir: Path) -> dict[tuple[str, str], float]` — (name, config) → p50
  - `rank_stats(cpu: dict[str, float], jetson: dict[str, float]) -> dict[str, float]` — `{"spearman", "kendall", "n"}`
  - `fit_reference(params: dict[str, float], lat: dict[str, float], families: dict[str, str]) -> tuple[float, float]` — `(slope, intercept)`
  - `residuals(params, lat, slope, intercept, names) -> dict[str, float]`
  - `build_report(...) -> dict[str, object]`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cpu_rank_report.py`:

```python
"""Contract tests for the rank-correlation report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lut.orchestrate.cpu_rank_report import (
    fit_reference,
    load_cpu,
    load_jetson,
    rank_stats,
    residuals,
)


def test_rank_stats_perfect_agreement() -> None:
    cpu = {"a": 1.0, "b": 2.0, "c": 3.0}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    got = rank_stats(cpu, jetson)
    assert got["spearman"] == pytest.approx(1.0)
    assert got["kendall"] == pytest.approx(1.0)
    assert got["n"] == 3


def test_rank_stats_is_invariant_to_monotone_scaling() -> None:
    """The reason governor state doesn't matter for a rank check."""
    cpu = {"a": 1.0, "b": 2.0, "c": 3.0}
    slow = {k: v * 1.35 + 4.0 for k, v in cpu.items()}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    assert rank_stats(cpu, jetson)["spearman"] == rank_stats(slow, jetson)["spearman"]


def test_rank_stats_perfect_inversion() -> None:
    cpu = {"a": 3.0, "b": 2.0, "c": 1.0}
    jetson = {"a": 10.0, "b": 20.0, "c": 30.0}
    assert rank_stats(cpu, jetson)["spearman"] == pytest.approx(-1.0)


def test_rank_stats_uses_only_paired_names() -> None:
    """An unpaired CPU row must be dropped, not crash or corrupt the correlation."""
    cpu = {"a": 1.0, "b": 2.0, "orphan": 9.0}
    jetson = {"a": 10.0, "b": 20.0, "missing": 5.0}
    assert rank_stats(cpu, jetson)["n"] == 2


def test_rank_stats_needs_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        rank_stats({"a": 1.0}, {"a": 2.0})


def test_fit_reference_recovers_a_known_line() -> None:
    """latency = 2*params + 1 over the reference families only."""
    params = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "g1": 2.0}
    lat = {"r1": 3.0, "r2": 5.0, "r3": 7.0, "g1": 99.0}
    fams = {"r1": "dense", "r2": "prune", "r3": "baseline", "g1": "graft"}
    slope, intercept = fit_reference(params, lat, fams)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_fit_reference_ignores_the_anchor() -> None:
    """yolo11s at 9.7M would dominate the slope as a leverage point."""
    params = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "anchor": 9.7}
    lat = {"r1": 3.0, "r2": 5.0, "r3": 7.0, "anchor": 500.0}
    fams = {"r1": "dense", "r2": "prune", "r3": "baseline", "anchor": "anchor"}
    slope, _ = fit_reference(params, lat, fams)
    assert slope == pytest.approx(2.0)


def test_fit_reference_needs_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_reference({"r1": 1.0}, {"r1": 3.0}, {"r1": "dense"})


def test_residuals_are_ms_above_the_line() -> None:
    params = {"g1": 2.0, "g2": 3.0}
    lat = {"g1": 8.0, "g2": 7.0}
    got = residuals(params, lat, slope=2.0, intercept=1.0, names=["g1", "g2"])
    assert got["g1"] == pytest.approx(3.0)   # 8 - (2*2+1)
    assert got["g2"] == pytest.approx(0.0)   # 7 - (2*3+1)


def test_load_jetson_takes_fp32_p50_only(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(
        {"name": "a", "precision": "fp32", "latency_ms": {"p50": 12.7}, "source": "jetson_trt"}))
    (tmp_path / "a_fp16.json").write_text(json.dumps(
        {"name": "a_fp16", "precision": "fp16", "latency_ms": {"p50": 7.7}, "source": "jetson_trt"}))
    (tmp_path / "report.json").write_text(json.dumps({"not": "a row"}))
    got = load_jetson(tmp_path)
    assert got == {"a": 12.7}


def test_load_cpu_keys_by_name_and_config(tmp_path: Path) -> None:
    (tmp_path / "a__t4.json").write_text(json.dumps(
        {"name": "a", "config": "t4", "latency_ms": {"p50": 40.0}, "source": "x86_ort"}))
    got = load_cpu(tmp_path)
    assert got == {("a", "t4"): 40.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cpu_rank_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lut.orchestrate.cpu_rank_report'`

- [ ] **Step 3: Write the report**

Create `lut/orchestrate/cpu_rank_report.py`:

```python
"""Does the Orin's architecture ranking survive on x86? -- the cross-device rank check report.

Joins the CPU rows written by ``cpu_bench`` against the measured Jetson fp32 rows in
``data/e2e/`` and answers two questions per thread config:

1. **Spearman/Kendall, CPU vs Jetson** -- does the ordering hold?
2. **The graft penalty curve** -- for each config, fit OLS of latency on params across the
   reference families (dense + prune + baseline, 15 models, 0.9-3.9M params) and report each
   graft's *residual*: the ms it costs beyond what its size predicts. A residual that GROWS
   with thread count means grafts lose more as bandwidth pressure rises (memory-bound
   generalises); one that is flat and positive everywhere points instead at a runtime/kernel
   effect, not bandwidth.

The anchor (yolo11s, 9.7M) is excluded from the fit: at 2.5x the next-largest reference model
it would dominate the slope as a leverage point. It still enters the Spearman.

**Caveat carried into every claim:** ORT != TensorRT. Graph optimisation and kernel selection
differ, so part of any rank delta is *runtime*, not *device*; this bench cannot separate them.

Run::

    python -m lut.orchestrate.cpu_rank_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from scipy.stats import kendalltau, spearmanr

from lut.orchestrate.cpu_bench import root_dir
from lut.orchestrate.cpu_pairs import GRAFT_FAMILIES, PAIRS, REFERENCE_FAMILIES


def count_params(onnx_path: Path) -> int:
    """Total initialiser elements -- the ONNX is the truth, not README's rounded '~2.4M'."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
    return total


def load_jetson(e2e_dir: Path) -> dict[str, float]:
    """name -> fp32 p50 ms. fp16 is skipped: README records +-20% TRT build variance."""
    out: dict[str, float] = {}
    for path in sorted(e2e_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "latency_ms" not in row:
            continue
        if row.get("precision") != "fp32":
            continue
        out[str(row["name"])] = float(row["latency_ms"]["p50"])
    return out


def load_cpu(cpu_dir: Path) -> dict[tuple[str, str], float]:
    """(name, config) -> p50 ms."""
    out: dict[tuple[str, str], float] = {}
    for path in sorted(cpu_dir.glob("*__*.json")):
        row = json.loads(path.read_text())
        out[(str(row["name"]), str(row["config"]))] = float(row["latency_ms"]["p50"])
    return out


def rank_stats(cpu: dict[str, float], jetson: dict[str, float]) -> dict[str, float]:
    """Spearman + Kendall tau-b over the names present in both."""
    names = sorted(set(cpu) & set(jetson))
    if len(names) < 3:
        raise ValueError(f"need at least 3 paired models, got {len(names)}")
    x = [cpu[n] for n in names]
    y = [jetson[n] for n in names]
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "n": len(names),
    }


def fit_reference(
    params: dict[str, float], lat: dict[str, float], families: dict[str, str]
) -> tuple[float, float]:
    """OLS latency ~ params over REFERENCE_FAMILIES only. Returns (slope, intercept)."""
    names = sorted(
        n for n in set(params) & set(lat) if families.get(n) in REFERENCE_FAMILIES
    )
    if len(names) < 2:
        raise ValueError(f"need at least 2 reference models to fit, got {len(names)}")
    slope, intercept = np.polyfit([params[n] for n in names], [lat[n] for n in names], 1)
    return float(slope), float(intercept)


def residuals(
    params: dict[str, float],
    lat: dict[str, float],
    slope: float,
    intercept: float,
    names: list[str],
) -> dict[str, float]:
    """ms above the reference line -- the graft penalty at matched params."""
    return {n: lat[n] - (slope * params[n] + intercept) for n in names if n in lat}


def build_report(
    cpu: dict[tuple[str, str], float],
    jetson: dict[str, float],
    params: dict[str, float],
    families: dict[str, str],
    configs: list[str],
) -> dict[str, object]:
    per_config: dict[str, object] = {}
    for cfg in configs:
        lat = {name: v for (name, c), v in cpu.items() if c == cfg}
        if len(lat) < 3:
            continue
        slope, intercept = fit_reference(params, lat, families)
        graft_names = sorted(n for n in lat if families.get(n) in GRAFT_FAMILIES)
        res = residuals(params, lat, slope, intercept, graft_names)
        per_config[cfg] = {
            "rank_vs_jetson": rank_stats(lat, jetson),
            "reference_fit": {"slope_ms_per_param": slope, "intercept_ms": intercept},
            "graft_residual_ms": res,
            "graft_residual_mean_ms": float(np.mean(list(res.values()))) if res else 0.0,
            "scatter": {n: {"params": params[n], "latency_ms": lat[n],
                            "family": families.get(n, "?")} for n in sorted(lat)},
        }
    return {
        "per_config": per_config,
        "clean_sweep_configs": [c for c in configs if c != "all22"],
        "caveats": [
            "all22 is confounded (E-cores are slower AND add pressure) -- excluded from the "
            "scaling claim.",
            "ORT != TensorRT: part of any rank delta is runtime, not device. This bench cannot "
            "separate them.",
            "The penalty metric assumes latency is ~linear in params; params ignores activation "
            "volume, which is what a memory-bound argument is about. Read the scatter.",
            "Single machine, single run: a rank inversion near the noise floor needs repeat "
            "rounds before being quoted.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = root_dir()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cpu-dir", type=Path, default=root / "data" / "cpu")
    ap.add_argument("--e2e-dir", type=Path, default=root / "data" / "e2e")
    ap.add_argument("--out", type=Path, default=root / "data" / "cpu" / "rank_report.json")
    args = ap.parse_args(argv)

    cpu = load_cpu(args.cpu_dir)
    if not cpu:
        print(f"error: no CPU rows in {args.cpu_dir} -- run cpu_bench first")
        return 2
    jetson = load_jetson(args.e2e_dir)
    families = {p.jetson_name: p.family for p in PAIRS}
    params = {
        p.jetson_name: count_params(root / p.onnx) / 1e6
        for p in PAIRS
        if (root / p.onnx).is_file()
    }
    configs = sorted({c for _, c in cpu}, key=lambda c: (c == "all22", c))

    report = build_report(cpu, jetson, params, families, configs)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'config':>8} {'rho':>7} {'tau':>7} {'n':>4} {'graft penalty (ms)':>20}")
    print("-" * 52)
    for cfg, block in report["per_config"].items():  # type: ignore[union-attr]
        rk = block["rank_vs_jetson"]                  # type: ignore[index]
        pen = block["graft_residual_mean_ms"]         # type: ignore[index]
        tag = "  (confounded)" if cfg == "all22" else ""
        print(f"{cfg:>8} {rk['spearman']:>7.3f} {rk['kendall']:>7.3f} {rk['n']:>4} "
              f"{pen:>20.2f}{tag}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cpu_rank_report.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Lint, type-check, commit**

```bash
bash scripts/check.sh -m "not slow"
git add lut/orchestrate/cpu_rank_report.py tests/test_cpu_rank_report.py
git commit -m "cpu-bench: rank-correlation report + graft-penalty residual curve

Spearman/Kendall vs the measured Jetson fp32 rows per thread config, plus
each graft's residual against an OLS latency~params fit over dense+prune+
baseline (anchor excluded: 9.7M params would dominate the slope)."
```

---

### Task 5: Run the bench and record the result

**Files:**
- Modify: `procedure.md` (append a new section at the end)
- Modify: `models/README.md` (append a CPU cross-check subsection)
- Produces: `data/cpu/*.json` (gitignored), `data/cpu/rank_report.json` (gitignored)

**Interfaces:**
- Consumes: `cpu_bench.main`, `cpu_rank_report.main`
- Produces: nothing importable — this task produces the *result*

- [ ] **Step 1: Confirm `data/cpu/` is gitignored**

Run: `git check-ignore -v data/cpu/x.json`
Expected: a line naming `.gitignore` and the `data/` rule. If it prints nothing, STOP and add `data/` coverage — CPU rows must never be committed.

- [ ] **Step 2: Quiesce the machine and run the full matrix**

Close heavy applications first. Confirm AC:
```bash
cat /sys/class/power_supply/ADP1/online   # expect 1
```
Then run (~20–40 min, resumable — safe to Ctrl-C):
```bash
.venv/bin/python -m lut.orchestrate.cpu_bench 2>&1 | tee /tmp/claude-1000/-home-asil-Desktop-TFM-NAS/3a93b1ab-5075-4ada-8aff-8c44208dc67e/scratchpad/cpu_bench.log
```
Expected: `28 models x 5 configs, 10 rounds x 6 iters -> n=60/cell`, then per-config progress and `wrote 28 rows` five times.

- [ ] **Step 3: Check for thermal drift before trusting anything**

Run:
```bash
.venv/bin/python -c "
import json, pathlib
rows = [json.loads(p.read_text()) for p in pathlib.Path('data/cpu').glob('*__*.json')]
drifted = sorted({r['config'] for r in rows if r['thermal_drift_detected']})
print('rows:', len(rows), '| drifted configs:', drifted or 'none')
"
```
Expected: `rows: 140 | drifted configs: none`. **If any config drifted**, re-run just that config on a cooler machine (`--configs <name> --force`) before quoting its numbers. A drifted config is reportable only with the caveat attached.

- [ ] **Step 4: Generate the report**

Run: `.venv/bin/python -m lut.orchestrate.cpu_rank_report`
Expected: a 5-row table (`t1 t2 t4 t6 all22`) with Spearman, Kendall, n=28, and the mean graft penalty per config, plus `-> data/cpu/rank_report.json`.

- [ ] **Step 5: Record the result in `procedure.md`**

Append a section headed `## CPU cross-device rank check (2026-07-15)` covering, in prose:
- The question: is the Orin's ordering — and the CP 6.2-G memory-bound graft rejection — a property of the architectures or of the Orin?
- The machine and method: Core Ultra 9 185H, 6 physical P-cores `{0,1,3,6,8,10}`, ORT 1.27 CPU EP fp32 @640, 28 e2e models × 5 configs, n=60/cell, round-robin interleaved with a canary.
- **The measured numbers**: the ρ/τ per config and the graft-penalty curve `t1 → t2 → t4 → t6`, quoted from `rank_report.json`.
- **The verdict**, stated as whichever the data supports: penalty grows with thread count → memory-boundness generalises, CP 6.2-G hardens; penalty ~zero at t1 → the effect is about the memory system; penalty flat and positive → suspect runtime, not bandwidth.
- The threats verbatim from spec §9 — especially **ORT ≠ TensorRT**, which caps how hard the conclusion can be pushed.

- [ ] **Step 6: Add the cross-check to `models/README.md`**

Append a short subsection under the frontier table pointing at the CPU result and stating plainly that **CPU rows are `x86_ort` and are never comparable to the `jetson_trt` latencies in the table** — they answer a ranking question, not a deployment one.

- [ ] **Step 7: Commit**

```bash
git add procedure.md models/README.md
git commit -m "procedure: record the x86 CPU cross-device rank check result"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 hardware / P-core topology | Task 1 (`physical_p_cores`, verified on the real machine in Step 5) |
| §3.1 timing core | Task 1 |
| §3.2 driver | Task 3 |
| §3.3 analysis | Task 4 |
| §4 pair map + exclusions | Task 2 |
| §5.1 config matrix, fp32-only | Task 3 (`build_configs`, `build_row`) |
| §5.2 interleaving, R=10×k=6 | Task 3 (`schedule`, defaults) |
| §5.3 canary drift | Task 3 (`drift_detected`) + Task 5 Step 3 |
| §5.4 governor recorded not changed | Task 3 (`env_stamp`) |
| §6 row schema | Task 3 (`build_row`) + tests |
| §7 error handling | Task 2 (`resolve_pairs` hard-fail), Task 3 (resume), Task 4 (unpaired dropped) |
| §8 testing | Tasks 1–4 |
| §9 threats | Task 4 (`build_report.caveats`) + Task 5 Step 5 |
| §10 deliverables | All |

**Gap found and closed:** the spec's §7 says "session creation fails for one model → record `status: "fail"`, continue". The driver as written lets that exception propagate. This is **deliberate and the spec is now the looser document**: a model that cannot build a session is a broken artifact, and silently continuing would produce a short, weaker correlation — exactly the failure mode `resolve_pairs` exists to prevent. Failing loudly is the better behaviour; if it ever fires, the fix is to repair or explicitly exclude the model.

**Placeholder scan:** none — every code step carries complete code, every command carries expected output.

**Type consistency:** `LatencyStats.as_dict()` (not `as_row`) is used consistently in Tasks 1 and 3. `root_dir()` is defined in `cpu_bench` and imported by `cpu_rank_report`. `BenchConfig.threads=0` means "ORT default" in both `cpu_ort.make_session` and `cpu_bench.build_configs`.
