"""jetson/run_search.py MODE=prune_recover — the PG_* → prune.recover_graft contract.

The jetson entry is a script (not a package), so each test loads it fresh via importlib with
a controlled environment; the module reads its MODE/PG_* config at import time. The composed
command must mirror ``colab/run_prune_graft.compose_recover_cmd`` (the tested free-tier
contract): --technique always passed, spec wins over ratios, KD teacher defaults to the donor.
No torch / no network — the module's top level is stdlib-only.
"""
from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "jetson" / "run_search.py"

DONOR = Path("/data/gate_best.pt")
YAML = Path("/data/dataset/dataset.yaml")
OUT = Path("/data/out/prune_recover")

_PG_KEYS = ("MODE", "PG_SPEC", "PG_RATIOS", "PG_TECH", "PG_ARCH_JSON", "PG_KD",
            "PG_KD_ALPHA", "PG_TEACHER", "PG_SEED", "PG_EPOCHS", "PG_BATCH", "PG_LR",
            "PG_CKPT_EVERY")


@contextmanager
def _entry(env: dict[str, str]) -> Iterator[object]:
    """Load a fresh run_search module under exactly `env` (other PG_* keys cleared)."""
    saved = {k: os.environ.pop(k) for k in _PG_KEYS if k in os.environ}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("_rs_under_test", ENTRY)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k in _PG_KEYS:
            os.environ.pop(k, None)
        os.environ.update(saved)


def _cmd(env: dict[str, str]) -> str:
    with _entry(env) as mod:
        return mod.prune_recover_cmd(DONOR, YAML, OUT)  # type: ignore[attr-defined]


def test_defaults_match_the_free_tier_contract() -> None:
    cmd = _cmd({})
    assert cmd.startswith("python3 -m prune.recover_graft ")
    assert f"--head-weights {DONOR}" in cmd
    assert f"--data-yaml {YAML}" in cmd
    assert f"--out-dir {OUT}" in cmd
    assert "--device cuda" in cmd and "--imgsz 640" in cmd
    assert "--technique global_taylor" in cmd            # ALWAYS passed
    assert "--ratios 0.50" in cmd and "--ratio-spec" not in cmd
    assert f"--teacher {DONOR} --kd-alpha 1.0" in cmd    # KD on, teacher = donor
    assert "--seed 0" in cmd and "--epochs 100" in cmd and "--ckpt-every 10" in cmd


def test_spec_wins_over_ratios() -> None:
    cmd = _cmd({"PG_SPEC": "prune/specs/v2_act292.json", "PG_RATIOS": "0.40"})
    assert "--ratio-spec prune/specs/v2_act292.json" in cmd
    assert "--ratios" not in cmd


def test_arch_json_probe_passthrough() -> None:
    cmd = _cmd({"PG_ARCH_JSON": "prune/specs/minact_arch.json",
                "PG_SPEC": "prune/specs/u30.json"})
    assert "--arch-json prune/specs/minact_arch.json" in cmd
    assert "--ratio-spec prune/specs/u30.json" in cmd


def test_no_kd_drops_teacher() -> None:
    cmd = _cmd({"PG_KD": "0"})
    assert "--teacher" not in cmd and "--kd-alpha" not in cmd


def test_teacher_override_and_seed() -> None:
    cmd = _cmd({"PG_TEACHER": "/data/yolo11s_gate.pt", "PG_SEED": "2",
                "PG_KD_ALPHA": "0.5"})
    assert "--teacher /data/yolo11s_gate.pt --kd-alpha 0.5" in cmd
    assert "--seed 2" in cmd


def test_mode_selector_admits_prune_recover() -> None:
    with _entry({"MODE": "prune_recover"}) as mod:
        assert mod.MODE == "prune_recover"  # type: ignore[attr-defined]
