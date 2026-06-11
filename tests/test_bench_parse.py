"""_parse_bench_stdout: remote bench output parsing must fail diagnosably.

A container can exit 0 yet produce no/garbage stdout (driver hiccup, image
drift); the parser must turn that into a ValueError whose message carries
the evidence — never an IndexError.
"""
import json

import pytest

pytest.importorskip("fabric", reason="run_sweep imports fabric via ssh_client")
pytest.importorskip("tqdm", reason="run_sweep imports tqdm")

from lut.orchestrate.run_sweep import _parse_bench_stdout  # noqa: E402

GOOD = {"latency_ms": {"mean": 0.4}, "peak_mem_mib": 18.7, "io_bytes": 100}


def test_single_json_line():
    assert _parse_bench_stdout(json.dumps(GOOD)) == GOOD


def test_noise_before_json_is_ignored():
    stdout = "TRT warning: blah\nbuilding engine...\n" + json.dumps(GOOD) + "\n"
    assert _parse_bench_stdout(stdout) == GOOD


def test_empty_stdout_raises_valueerror():
    with pytest.raises(ValueError, match="no stdout"):
        _parse_bench_stdout("")


def test_whitespace_only_stdout_raises_valueerror():
    with pytest.raises(ValueError, match="no stdout"):
        _parse_bench_stdout("  \n\n  \n")


def test_non_json_tail_raises_with_evidence():
    with pytest.raises(ValueError, match="Segmentation fault"):
        _parse_bench_stdout("building...\nSegmentation fault\n")


def test_json_scalar_is_rejected():
    with pytest.raises(ValueError, match="not an object"):
        _parse_bench_stdout("42\n")
