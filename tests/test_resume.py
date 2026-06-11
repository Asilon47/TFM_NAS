"""completed_keys(): resumability over partial or corrupt lut.jsonl files.

A truncated final line is *expected* after Ctrl-C / power loss mid-write —
skipping it (so the row gets re-measured) is the designed behavior, not an
error. These tests freeze that semantic.
"""
import json
from pathlib import Path

from lut.orchestrate.resume import completed_keys


def _row(key: str) -> str:
    return json.dumps({"row_key": key, "block": "mbconv"})


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_missing_file_is_empty(tmp_path):
    assert completed_keys(tmp_path / "absent.jsonl") == set()


def test_empty_file_is_empty(tmp_path):
    p = tmp_path / "lut.jsonl"
    p.write_text("")
    assert completed_keys(p) == set()


def test_valid_rows_collected(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa"), _row("bbb")])
    assert completed_keys(p) == {"aaa", "bbb"}


def test_blank_lines_skipped(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa"), "", "   ", _row("bbb")])
    assert completed_keys(p) == {"aaa", "bbb"}


def test_corrupt_line_skipped_others_kept(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa"), '{"row_key": "br', _row("ccc")])
    assert completed_keys(p) == {"aaa", "ccc"}


def test_truncated_tail_skipped(tmp_path):
    p = tmp_path / "lut.jsonl"
    p.write_text(_row("aaa") + "\n" + _row("bbb")[:10])
    assert completed_keys(p) == {"aaa"}


def test_rows_without_row_key_ignored(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [json.dumps({"block": "mbconv"}), _row("aaa")])
    assert completed_keys(p) == {"aaa"}


def test_duplicate_keys_dedupe(tmp_path):
    p = _write(tmp_path / "lut.jsonl", [_row("aaa"), _row("aaa")])
    assert completed_keys(p) == {"aaa"}
