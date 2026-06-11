"""Root conftest — anchors pytest imports at the repo root.

The project is intentionally unpackaged: runtime invocation is
``python -m <pkg>.<mod>`` from the repo root (see CLAUDE.md). pytest's
default "prepend" import mode puts this directory on ``sys.path`` because
this conftest.py lives here, which makes ``catalog``, ``search``, ``lut``,
and ``supernet`` importable from tests without an install step.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def lut_path() -> Path:
    """Path to data/lut.jsonl; skips the requesting test when absent (e.g. CI)."""
    p = REPO_ROOT / "data" / "lut.jsonl"
    if not p.exists():
        pytest.skip("data/lut.jsonl not present — generate it with "
                    "`python -m lut.orchestrate.gen_dummy_lut`")
    return p
