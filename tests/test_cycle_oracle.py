"""mcu/cycle_oracle.py — pure parts (the sim itself needs docker + .venv-nas)."""
import json

import pytest

from mcu.cycle_oracle import (
    MATCHED_L2,
    candidate_key,
    canonical_candidate,
    cycles_for,
)

ARCH = {"ks": [3, 5, 3], "e": [4, 6, 4], "d": [2, 2, 3]}
SPEC = {"stage_ratios": [0.6, 0.0, 0.6, 0.6, 0.6], "rest_ratio": 0.3}


def _cand(**over):
    c = {"arch": ARCH, "spec": SPEC, "neck": "pan", "imgsz": 160}
    c.update(over)
    return c


def test_candidate_key_is_content_addressed() -> None:
    k1 = candidate_key(_cand())
    # field order / int-vs-float noise must not change the key
    reordered = {"imgsz": 160, "neck": "pan",
                 "spec": {"rest_ratio": 0.30, "stage_ratios": [0.6, 0, 0.6, 0.6, 0.6]},
                 "arch": {"d": [2, 2, 3], "e": [4, 6, 4], "ks": [3, 5, 3]}}
    assert candidate_key(reordered) == k1
    # every graph-determining field moves the key
    assert candidate_key(_cand(neck=None)) != k1
    assert candidate_key(_cand(imgsz=192)) != k1
    assert candidate_key(_cand(spec={"stage_ratios": [0.5, 0, 0.6, 0.6, 0.6],
                                     "rest_ratio": 0.3})) != k1


def test_canonical_candidate_guards_neck() -> None:
    with pytest.raises(ValueError, match="unknown neck"):
        canonical_candidate(_cand(neck="bifpn"))


def test_matched_l2_is_the_recorded_budget() -> None:
    """Every recorded CP 10.1 number is at 84 KB; the oracle must default to it."""
    assert MATCHED_L2 == 84000


def test_cycles_for_hits_cache_without_subprocess(monkeypatch, tmp_path) -> None:
    """A cached result short-circuits before any export/sim subprocess."""
    import mcu.cycle_oracle as co

    monkeypatch.setattr(co, "ORACLE_DIR", tmp_path)
    cand = _cand()
    cached = {"status": "ok", "cycles": 123, "key": candidate_key(cand)}
    (tmp_path / f"cand_{candidate_key(cand)}.json").write_text(json.dumps(cached))

    def boom(*a, **k):  # any subprocess call = cache miss = failure
        raise AssertionError("subprocess must not run on a cache hit")

    monkeypatch.setattr(co.subprocess, "run", boom)
    assert cycles_for(cand) == cached


def test_infeasible_result_is_cached(monkeypatch, tmp_path) -> None:
    """A failed export caches status=infeasible so the search never re-prices it."""
    import types

    import mcu.cycle_oracle as co

    monkeypatch.setattr(co, "ORACLE_DIR", tmp_path)
    monkeypatch.setattr(co.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="",
                                                              stderr="boom"))
    cand = _cand()
    res = cycles_for(cand)
    assert res["status"] == "infeasible" and res["stage"] == "export"
    on_disk = json.loads((tmp_path / f"cand_{res['key']}.json").read_text())
    assert on_disk["status"] == "infeasible"
