"""Resumability: track which row_keys already live in data/lut.jsonl."""
from pathlib import Path

from lut.loader import iter_lut_rows


def completed_keys(jsonl_path: Path, precision: str | None = None) -> set[str]:
    """row_keys present in the file, optionally only rows at ``precision``.

    ``precision=None`` keeps the legacy any-precision behavior (used by the
    CP 2.1 DoD smoke test). ``run_sweep`` passes its configured precision so
    that switching precision re-measures rows instead of silently skipping
    them — ``row_key`` does not encode precision (lut/docs/schema.md), so
    without the filter a dummy/fp16 file would mask an fp32 sweep entirely.
    """
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    for row in iter_lut_rows(jsonl_path):
        if precision is not None and row.get("precision") != precision:
            continue
        key = row.get("row_key")
        if key:
            done.add(key)
    return done
