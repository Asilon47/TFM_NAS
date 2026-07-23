import importlib.util
import pathlib

import pytest

_MOD = pathlib.Path(__file__).resolve().parents[1] / "mcu" / "board" / "bench_receiver.py"
_spec = importlib.util.spec_from_file_location("bench_receiver", _MOD)
assert _spec and _spec.loader
bench_receiver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_receiver)
parse_bench_line = bench_receiver.parse_bench_line


def test_parses_bench_line_and_derives_ms_fps():
    line = (
        "BENCH model=cand_a5fddcc354bd res=192 cyc=58394932 "
        "clk_us=333000 n=20 fcl=175"
    )
    rec = parse_bench_line(line)
    assert rec is not None
    assert rec["model"] == "cand_a5fddcc354bd"
    assert rec["res"] == 192
    assert rec["cyc"] == 58394932
    assert rec["clk_us"] == 333000
    assert rec["n"] == 20
    assert rec["fcl"] == 175
    assert rec["ms"] == pytest.approx(333.0)
    assert rec["fps"] == pytest.approx(3.003, abs=1e-3)


def test_non_bench_line_returns_none():
    assert parse_bench_line("*** net_bench cand_x res=192 iters=20 ***") is None
    assert parse_bench_line("SMOKE 3 construct ok arena_base=5248064") is None
    assert parse_bench_line("") is None


def test_zero_clk_us_does_not_crash():
    rec = parse_bench_line("BENCH model=m res=160 cyc=1 clk_us=0 n=20 fcl=175")
    assert rec is not None
    assert rec["fps"] is None and rec["ms"] == 0.0
