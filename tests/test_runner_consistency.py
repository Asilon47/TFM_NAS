"""The three remote runner kits pin the same (RES, T_MAX_MS) regime — by copy, on purpose.

``kaggle/run.py``, ``jetson/run_search.py`` and ``colab/run_colab.py`` each hard-code
RES / T_MAX_MS because every kit must run standalone on its remote (no shared import), and
the per-seed resume caches are RES-namespaced (``cp33_bo_cache_r{RES}``) with the ceiling
baked into every cached feasibility verdict — one silently diverging copy invalidates caches
and DoDs. This gate pins all three to the committed @640 regime instead of refactoring the
duplication away (the duplication is the crash-safe design; see procedure.md "Plan pivot").

ast-parsed, not imported: the kits are remote entry scripts (top-level side effects), and the
``kaggle`` directory name collides with the pip-installed kaggle package.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KITS = ["kaggle/run.py", "jetson/run_search.py", "colab/run_colab.py"]
EXPECTED: dict[str, float] = {"RES": 640, "T_MAX_MS": 12.75}


def _resolve_ifexp(val: ast.IfExp, known: dict[str, float]) -> float:
    """Resolve kaggle's ``12.75 if RES == 640 else 16.7`` against the already-parsed RES."""
    t = val.test
    if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id in known
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.comparators[0], ast.Constant)
            and isinstance(val.body, ast.Constant) and isinstance(val.orelse, ast.Constant)):
        taken = val.body if known[t.left.id] == t.comparators[0].value else val.orelse
        if isinstance(taken.value, int | float):
            return float(taken.value)
    raise AssertionError(f"unresolvable conditional regime constant: {ast.dump(val)}")


def _regime(path: Path) -> dict[str, float]:
    """Module-level RES / T_MAX_MS assignments, in source order."""
    out: dict[str, float] = {}
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in EXPECTED):
            continue
        name, val = node.targets[0].id, node.value
        if isinstance(val, ast.Constant):
            assert isinstance(val.value, int | float), f"{path.name}: {name} is not numeric"
            out[name] = float(val.value)
        elif isinstance(val, ast.IfExp):
            out[name] = _resolve_ifexp(val, out)
        else:
            raise AssertionError(
                f"{path.name}: {name} must stay a literal (or `X if RES == n else Y`) "
                f"for this regime gate, got: {ast.dump(val)}")
    return out


@pytest.mark.parametrize("kit", KITS)
def test_kit_pins_the_committed_regime(kit: str) -> None:
    regime = _regime(ROOT / kit)
    assert regime == EXPECTED, (
        f"{kit} regime {regime} != {EXPECTED}. Changing RES/T_MAX_MS invalidates the "
        f"RES-namespaced seed caches and every cached feasibility verdict — if intentional, "
        f"record the decision in procedure.md and update EXPECTED here in the same commit.")
