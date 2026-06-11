"""Tripwire: every catalog row has a row in the generated LUT.

Catches any change that re-keys rows (grid edits, row_key changes, cfg type
drift) the moment the catalog diverges from the on-disk artifact. The
assertion is subset, not equality: append-only means the file may legally
hold rows for retired cfgs, but every *current* catalog entry must be
covered (Phase 0 DoD).

Skips when data/lut.jsonl hasn't been generated (CI, fresh clones).
"""
from catalog.sweep import iter_sweep
from lut.orchestrate.resume import completed_keys


def test_every_catalog_key_is_in_lut_file(lut_path):
    catalog_keys = {k for *_, k in iter_sweep()}
    lut_keys = completed_keys(lut_path)
    missing = catalog_keys - lut_keys
    assert not missing, (
        f"{len(missing)} catalog rows missing from {lut_path} — regenerate "
        "the dummy LUT or run the sweep after widening a grid"
    )
