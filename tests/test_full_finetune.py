"""Side experiment (full fine-tune of winner-v1) — pure wiring tests via a stubbed
``short_finetune``. This is not a DoD (no verdict logic to test), just the reload -> loop ->
average -> JSON orchestration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.full_finetune import full_finetune

_WINNER = {
    "arch": {"ks": [3] * 20, "e": [3] * 20, "d": [2, 2, 4, 3, 3]},
    "acc": 0.610,
    "anchors": {
        "a": {"acc": 0.8774, "latency_ms": 12.75},
        "b": {"acc": 0.8819, "latency_ms": 21.69},
    },
}


def _write_winner(tmp_path: Path) -> Path:
    d = tmp_path / "winner_v1"
    d.mkdir()
    (d / "winner.json").write_text(json.dumps(_WINNER))
    return d


def test_full_finetune_single_seed_writes_json(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        return {"map": 0.80, "map50": 0.95}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    out = full_finetune(d, seeds=(0,), epochs=100, device="cpu")

    assert out["maps"] == [0.80]
    assert out["mean"] == pytest.approx(0.80)
    assert out["std"] == pytest.approx(0.0)
    assert out["proxy_acc"] == pytest.approx(0.610)
    assert out["delta_vs_proxy"] == pytest.approx(0.19)
    assert out["anchors"]["a"]["acc"] == pytest.approx(0.8774)
    saved = json.loads((d / "full_finetune.json").read_text())
    assert saved["mean"] == pytest.approx(0.80)


def test_full_finetune_multi_seed_averages(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        return {"map": {0: 0.78, 1: 0.82}[kw["seed"]], "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    out = full_finetune(d, seeds=(0, 1), device="cpu")

    assert out["maps"] == [0.78, 0.82]
    assert out["mean"] == pytest.approx(0.80)
    assert out["std"] == pytest.approx(0.02)


def test_full_finetune_saves_weights_from_first_seed_only(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)
    save_targets: list = []

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        save_targets.append(kw.get("save_to"))
        return {"map": 0.8, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    out = full_finetune(d, seeds=(0, 1), save_weights=True)

    non_null = [t for t in save_targets if t is not None]
    assert len(non_null) == 1
    assert Path(non_null[0]).name == "full_finetune_weights.pt"
    assert out["weights"].endswith("full_finetune_weights.pt")


def test_full_finetune_no_save_weights(tmp_path: Path, monkeypatch) -> None:
    d = _write_winner(tmp_path)

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        assert kw.get("save_to") is None  # never saves when save_weights=False
        return {"map": 0.8, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    out = full_finetune(d, seeds=(0,), save_weights=False)
    assert out["weights"] is None


def test_full_finetune_default_is_warm_start_capable_and_unfrozen(
    tmp_path: Path, monkeypatch
) -> None:
    d = _write_winner(tmp_path)
    seen: dict = {}

    def fake_short_finetune(arch, **kw):  # noqa: ANN001, ANN003
        seen["freeze_head"] = kw["freeze_head"]
        seen["head_weights"] = kw["head_weights"]
        return {"map": 0.8, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake_short_finetune)
    full_finetune(d, seeds=(0,))

    assert seen["freeze_head"] is False
    assert seen["head_weights"] is None
