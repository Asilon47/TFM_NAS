"""CP 3.5 — winner-v1 verification tests (pure verdict + stubbed-fine-tune driver).

The verdict logic and the reloader are pure, so they run under ``.venv`` / CI. The driver calls
``short_finetune`` (GPU + dataset), so it is covered by stubbing that primitive — proving the
reload → 3-seed loop → average → repro.json orchestration without a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.verify_winner import REPRO_BAND, ReproVerdict, load_winner, verify_winner

# A minimal winner.json (only the fields the verifier reads).
_WINNER = {
    "arch": {"ks": [3] * 20, "e": [3] * 20, "d": [2, 2, 4, 4, 3]},
    "acc": 0.650,
    "method": "bo",
    "seed": 0,
    "latency_ms": 11.744,
    "t_max_ms": 12.75,
}


def _write_winner(tmp_path: Path, acc: float = 0.650) -> Path:
    d = tmp_path / "winner_v1"
    d.mkdir()
    (d / "winner.json").write_text(json.dumps({**_WINNER, "acc": acc}))
    return d


# ---- reloader ----------------------------------------------------------------

def test_load_winner_reads_arch_and_acc(tmp_path: Path) -> None:
    d = _write_winner(tmp_path)
    rec = load_winner(d)
    assert rec["arch"]["d"] == [2, 2, 4, 4, 3]
    assert rec["acc"] == pytest.approx(0.650)


# ---- ReproVerdict (the pure DoD logic) ---------------------------------------

def test_verdict_mean_within_band_passes() -> None:
    # cached 0.650, fresh mean = 0.655 -> delta 0.005 < 0.020 band.
    v = ReproVerdict(cached_acc=0.650, fresh_seeds=[0.66, 0.65, 0.655])
    assert v.fresh_mean == pytest.approx(0.655)
    assert v.delta == pytest.approx(0.005)
    assert v.passes


def test_verdict_mean_outside_band_fails() -> None:
    # fresh mean = 0.70 -> delta 0.05 > 0.020 band.
    v = ReproVerdict(cached_acc=0.650, fresh_seeds=[0.70, 0.70, 0.70])
    assert not v.passes


def test_verdict_worst_delta_is_max_single_seed_gap() -> None:
    # mean reproduces (delta 0), but one seed is far off — worst_delta surfaces it.
    v = ReproVerdict(cached_acc=0.650, fresh_seeds=[0.60, 0.70, 0.650])
    assert v.delta == pytest.approx(0.0)
    assert v.passes  # mean-based rule tolerates the spread
    assert v.worst_delta == pytest.approx(0.050)  # strict reading would flag it


def test_verdict_band_is_configurable() -> None:
    v_strict = ReproVerdict(cached_acc=0.650, fresh_seeds=[0.665] * 3, band=0.005)
    assert not v_strict.passes  # delta 0.015 > 0.005
    v_loose = ReproVerdict(cached_acc=0.650, fresh_seeds=[0.665] * 3, band=0.020)
    assert v_loose.passes


def test_default_band_constant() -> None:
    assert ReproVerdict(cached_acc=0.0, fresh_seeds=[0.0]).band == REPRO_BAND


# ---- driver (stubbed fine-tune) ----------------------------------------------

def test_verify_winner_averages_seeds_and_writes_repro(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path, acc=0.650)
    calls: list[dict] = []

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        calls.append({"seed": kw["seed"], "save_to": kw.get("save_to")})
        # return a per-seed map so the average is checkable: 0.64, 0.66, 0.65 -> mean 0.650
        return {"map": {1: 0.64, 2: 0.66, 3: 0.65}[kw["seed"]], "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)

    out = verify_winner(d, head_weights="gate.pt", seeds=(1, 2, 3), device="cpu")

    assert out["fresh_seeds"] == [0.64, 0.66, 0.65]
    assert out["fresh_mean"] == pytest.approx(0.650)
    assert out["delta"] == pytest.approx(0.0)
    assert out["passes"] is True
    # repro.json persisted beside the winner
    saved = json.loads((d / "repro.json").read_text())
    assert saved["cached_acc"] == pytest.approx(0.650)
    assert saved["method"] == "bo"


def test_verify_winner_saves_weights_from_first_seed_only(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)
    save_targets: list = []

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        save_targets.append(kw.get("save_to"))
        return {"map": 0.65, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)

    out = verify_winner(d, head_weights="gate.pt", seeds=(1, 2, 3), save_weights=True)

    # exactly one seed (the first) writes weights.pt; the rest pass save_to=None
    non_null = [t for t in save_targets if t is not None]
    assert len(non_null) == 1
    assert Path(non_null[0]).name == "weights.pt"
    assert out["weights"].endswith("weights.pt")


def test_verify_winner_no_save_weights(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        assert kw.get("save_to") is None  # never saves when save_weights=False
        return {"map": 0.65, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    out = verify_winner(d, head_weights="gate.pt", seeds=(1,), save_weights=False)
    assert out["weights"] is None
