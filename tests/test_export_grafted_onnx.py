"""detect/export_grafted_onnx.py — the pure arch-selection layer (Stage 0).

The export itself needs ofa + ultralytics (.venv-nas); what must be airtight everywhere is the
candidate selection — d values repeat in the top-12, so index is the identity — and the
one-source rule. Import must stay ultralytics-free under .venv (lazy imports).
"""
import json

import pytest

from detect.export_grafted_onnx import load_arch

WINNER = {"arch": {"ks": [3] * 20, "e": [4] * 20, "d": [2, 2, 4, 3, 3]}, "latency_ms": 11.2}
CANDS = {
    "candidates": [
        {"arch": {"ks": [3] * 20, "e": [4] * 20, "d": [2, 2, 4, 3, 2]},
         "latency_ms": 11.097, "acc": 0.6275, "method": "tpe", "seed": 3},
        {"arch": {"ks": [5] * 20, "e": [6] * 20, "d": [2, 2, 4, 3, 2]},
         "latency_ms": 10.727, "acc": 0.6226, "method": "bo", "seed": 3},
    ]
}


@pytest.fixture
def files(tmp_path):
    w, c = tmp_path / "winner.json", tmp_path / "cands.json"
    w.write_text(json.dumps(WINNER))
    c.write_text(json.dumps(CANDS))
    return w, c


def test_winner_source(files) -> None:
    w, _ = files
    arch, prov = load_arch(winner=w)
    assert arch["d"] == [2, 2, 4, 3, 3]
    assert prov["source"] == str(w)


def test_candidate_by_index_disambiguates_duplicate_d(files) -> None:
    _, c = files
    a0, p0 = load_arch(candidates=c, index=0)
    a1, p1 = load_arch(candidates=c, index=1)
    assert a0["d"] == a1["d"] == [2, 2, 4, 3, 2]          # the duplicate-d trap
    assert a0["ks"] != a1["ks"]                            # but different archs
    assert p0["cached_latency_ms"] == 11.097 and p1["cached_latency_ms"] == 10.727
    assert p1["index"] == 1


def test_source_rules(files) -> None:
    w, c = files
    with pytest.raises(ValueError, match="exactly one"):
        load_arch()
    with pytest.raises(ValueError, match="exactly one"):
        load_arch(winner=w, candidates=c, index=0)
    with pytest.raises(ValueError, match="--index"):
        load_arch(candidates=c)
    with pytest.raises(ValueError, match="out of range"):
        load_arch(candidates=c, index=2)


def test_winner_without_arch_key_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"latency_ms": 1.0}))
    with pytest.raises(ValueError, match="no 'arch' key"):
        load_arch(winner=bad)
