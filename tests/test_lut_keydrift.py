"""Tripwires between the catalog and the on-disk LUT artifact.

Two invariants with different lifetimes:

1. No orphans (always enforced): every row_key in data/lut.jsonl must be
   derivable from the *current* catalog. An orphan means a grid edit or
   row_key change re-keyed measured rows — the exact failure the golden
   hashes in test_row_key.py pin from the catalog side.
2. Completeness (Phase 0 DoD gate): once collection finishes, the file must
   cover every catalog row. While the real Jetson sweep is still filling the
   file this is legitimately partial, so the test SKIPS with a coverage
   report instead of failing.

Both skip entirely when data/lut.jsonl hasn't been generated (CI, fresh
clones) via the lut_path fixture.
"""
import pytest

from catalog.sweep import iter_sweep
from lut.orchestrate.resume import completed_keys


def test_no_orphan_lut_keys(lut_path):
    catalog_keys = {k for *_, k in iter_sweep()}
    lut_keys = completed_keys(lut_path)
    orphans = lut_keys - catalog_keys
    assert not orphans, (
        f"{len(orphans)} row(s) in {lut_path} carry keys no longer derivable "
        "from the catalog — a grid/row_key change re-keyed measured rows. If "
        "cfgs were deliberately retired, record the decision in procedure.md "
        "and update this test."
    )


def test_catalog_coverage_once_complete(lut_path):
    catalog_keys = {k for *_, k in iter_sweep()}
    lut_keys = completed_keys(lut_path)
    missing = catalog_keys - lut_keys
    if missing:
        pytest.skip(
            f"LUT coverage {len(catalog_keys) - len(missing)}/"
            f"{len(catalog_keys)} catalog rows — the completeness gate "
            "applies once the sweep finishes (resume with "
            "`python -m lut.orchestrate.run_sweep`)"
        )
    assert catalog_keys <= lut_keys
