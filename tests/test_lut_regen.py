"""End-to-end (slow): gen_dummy_lut regenerates the exact key universe.

Runs the real CLI in a subprocess against a temp path and checks the emitted
keys equal the catalog sweep's — guarding the whole chain (grids, row_key,
measure_block) in one shot. The payload test additionally pins flops/params
against the committed artifact, so refactors of the FLOPs counter cannot
silently change row payloads.
"""
import json
import os
import subprocess
import sys

import pytest

from catalog.sweep import iter_sweep

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def regen_rows(tmp_path_factory, repo_root):
    out = tmp_path_factory.mktemp("regen") / "lut.jsonl"
    device_info = out.parent / "device_info.json"
    # Strip PYTHONPATH: a globally exported one (e.g. ROS) must not leak in.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run(
        [sys.executable, "-m", "lut.orchestrate.gen_dummy_lut",
         "--out", str(out), "--device-info-out", str(device_info)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, f"gen_dummy_lut failed:\n{res.stderr}"
    assert device_info.exists()
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def test_regen_writes_full_catalog(regen_rows):
    assert len(regen_rows) == 2710


def test_regen_keys_match_catalog(regen_rows):
    assert {r["row_key"] for r in regen_rows} == {k for *_, k in iter_sweep()}


def test_regen_payload_matches_existing_lut(regen_rows, lut_path):
    """flops/params identical to the committed artifact for a spread sample."""
    existing = {}
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                existing[row["row_key"]] = row
    sample = regen_rows[:: max(1, len(regen_rows) // 7)]
    for row in sample:
        old = existing.get(row["row_key"])
        if old is None:  # file may predate a grid widening; keydrift covers that
            continue
        assert row["flops"] == old["flops"], row["row_key"]
        assert row["params"] == old["params"], row["row_key"]
