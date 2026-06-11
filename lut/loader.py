"""Read data/lut.jsonl into memory — the input surface for LUT-aware costing.

CP 2.2's cost function (``search/cost.py``) consumes :func:`load_lut`;
``lut.orchestrate.resume`` shares :func:`iter_lut_rows` so file-tolerance
policy lives in exactly one place.

``row_key`` does NOT encode precision (see lut/docs/schema.md): rows
measured at different precisions legally coexist under the same key in the
append-only file. Filter to one precision before keying — :func:`load_lut`
refuses to let two rows silently collide.
"""
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from catalog.contracts import LutRow


def iter_lut_rows(path: Path) -> Iterator[dict]:
    """Yield parsed rows; skip malformed lines with one stderr warning.

    A truncated tail is *expected* after an interrupted ``run_sweep`` write
    — the affected row is simply re-measured on resume. Malformed lines are
    therefore skipped, but counted and surfaced (instead of vanishing) so
    real corruption is noticed.
    """
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if not isinstance(row, dict):
                n_bad += 1
                continue
            yield row
    if n_bad:
        sys.stderr.write(
            f"[lut] skipped {n_bad} malformed line(s) in {path} — expected if "
            "a previous run was interrupted mid-write; the affected rows will "
            "be re-measured on the next sweep.\n"
        )


def load_lut(path: Path, precision: str | None = None) -> dict[str, LutRow]:
    """Load rows keyed by ``row_key``, optionally filtered to one precision.

    Raises:
        FileNotFoundError: if the LUT hasn't been generated/measured yet.
        ValueError: if two rows (after filtering) share a ``row_key`` —
            pass ``precision="fp16"``/``"fp32"`` to disambiguate a
            multi-precision file instead of letting last-write-win decide
            latencies silently.

    Note: rows written before the ``precision`` field existed are excluded
    by any precision filter (their precision is unknown).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — generate it with "
            "`python -m lut.orchestrate.gen_dummy_lut` (offline dummy) or "
            "`python -m lut.orchestrate.run_sweep` (real Jetson sweep)"
        )
    rows: dict[str, LutRow] = {}
    for row in iter_lut_rows(path):
        key = row.get("row_key")
        if not key:
            continue
        if precision is not None and row.get("precision") != precision:
            continue
        if key in rows:
            hint = ("" if precision is not None else
                    " — the file may mix precisions (row_key does not encode "
                    "precision); pass precision='fp16'/'fp32' to filter")
            raise ValueError(f"{path}: duplicate row_key {key!r}{hint}")
        rows[key] = row  # type: ignore[assignment]
    return rows
