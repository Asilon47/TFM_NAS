"""load_lut / iter_lut_rows: the validated input surface for CP 2.2 costing."""
import json
from pathlib import Path

import pytest

from lut.loader import iter_lut_rows, load_lut


def _row(key: str, precision: str | None = "fp16", **extra) -> str:
    row: dict = {"row_key": key, "block": "mbconv", **extra}
    if precision is not None:
        row["precision"] = precision
    return json.dumps(row)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_missing_file_names_the_remedies(tmp_path):
    with pytest.raises(FileNotFoundError, match="gen_dummy_lut"):
        load_lut(tmp_path / "absent.jsonl")


def test_load_all_when_unfiltered(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa"), _row("bbb")])
    lut = load_lut(p)
    assert set(lut) == {"aaa", "bbb"}
    assert lut["aaa"]["block"] == "mbconv"


def test_precision_filter_selects_subset(tmp_path):
    p = _write(tmp_path / "lut.jsonl",
               [_row("aaa", "fp16"), _row("bbb", "fp32"), _row("ccc", "fp16")])
    assert set(load_lut(p, precision="fp16")) == {"aaa", "ccc"}
    assert set(load_lut(p, precision="fp32")) == {"bbb"}


def test_same_key_across_precisions_collides_unfiltered(tmp_path):
    """row_key does not encode precision — unfiltered loads must not let
    file order decide which latency wins."""
    p = _write(tmp_path / "lut.jsonl", [_row("aaa", "fp16"), _row("aaa", "fp32")])
    with pytest.raises(ValueError, match="precision"):
        load_lut(p)


def test_same_key_across_precisions_fine_when_filtered(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa", "fp16"), _row("aaa", "fp32")])
    assert set(load_lut(p, precision="fp16")) == {"aaa"}


def test_duplicate_within_one_precision_collides(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa", "fp16"), _row("aaa", "fp16")])
    with pytest.raises(ValueError, match="duplicate row_key"):
        load_lut(p, precision="fp16")


def test_rows_without_precision_excluded_by_filter(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa", precision=None), _row("bbb")])
    assert set(load_lut(p, precision="fp16")) == {"bbb"}
    assert set(load_lut(p)) == {"aaa", "bbb"}


def test_malformed_lines_warned_once_with_count(tmp_path, capsys):
    p = _write(tmp_path / "lut.jsonl",
               [_row("aaa"), '{"row_key": "tru', "[1, 2]", _row("bbb")])
    rows = list(iter_lut_rows(p))
    assert [r["row_key"] for r in rows] == ["aaa", "bbb"]
    err = capsys.readouterr().err
    assert "skipped 2 malformed line(s)" in err
    assert str(p) in err


def test_clean_file_emits_no_warning(tmp_path, capsys):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa")])
    list(iter_lut_rows(p))
    assert capsys.readouterr().err == ""
