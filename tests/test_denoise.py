"""CP 3.5 de-noise tests — pure candidate/selection helpers + stubbed resumable driver."""
from __future__ import annotations

from pathlib import Path

import pytest

from search.denoise import (
    arch_key,
    denoise_archs,
    select_denoised,
    top_candidates,
)

_ARCH = {"ks": [3] * 20, "e": [3] * 20, "d": [2, 2, 4, 4, 3]}


def _pt(acc: float, lat: float, arch: dict | None = None, **kw) -> dict:
    return {"arch": arch or _ARCH, "acc": acc, "latency_ms": lat, "acc_eff": acc,
            "method": kw.get("method", "bo"), "seed": kw.get("seed", 0)}


# ---- candidates --------------------------------------------------------------

def test_top_candidates_by_acc_within_ceiling() -> None:
    frontier = [_pt(0.60, 11.0), _pt(0.65, 12.0), _pt(0.63, 10.0),
                _pt(0.70, 20.0)]  # 0.70 is infeasible (>12.75)
    cands = top_candidates(frontier, t_max=12.75, k=2)
    assert [c["acc"] for c in cands] == [0.65, 0.63]  # feasible, acc-desc; 0.70 excluded


def test_arch_key_is_stable_and_distinguishes() -> None:
    a2 = {**_ARCH, "d": [4, 4, 4, 4, 4]}
    assert arch_key(_ARCH) == arch_key(dict(_ARCH))  # same arch → same key
    assert arch_key(_ARCH) != arch_key(a2)           # different depth → different key


# ---- selection (the honest fastest-in-tie rule) ------------------------------

def test_select_denoised_fastest_within_tie() -> None:
    # top mean 0.62; the 0.61 arch is within one std (0.03) of it -> tie -> the faster wins.
    denoised = [
        {"arch": _ARCH, "latency_ms": 11.7, "denoised_mean": 0.62, "denoised_std": 0.03},
        {"arch": _ARCH, "latency_ms": 10.1, "denoised_mean": 0.61, "denoised_std": 0.03},
        {"arch": _ARCH, "latency_ms": 9.0, "denoised_mean": 0.50, "denoised_std": 0.02},
    ]
    w = select_denoised(denoised, t_max=12.75)
    assert w["latency_ms"] == 10.1  # tied with the top on accuracy but faster


def test_select_denoised_zero_band_is_plain_argmax() -> None:
    denoised = [
        {"arch": _ARCH, "latency_ms": 11.7, "denoised_mean": 0.62, "denoised_std": 0.03},
        {"arch": _ARCH, "latency_ms": 10.1, "denoised_mean": 0.61, "denoised_std": 0.03},
    ]
    w = select_denoised(denoised, t_max=12.75, tie_band=0.0)
    assert w["denoised_mean"] == 0.62  # no tie window -> the most accurate, ignoring latency


def test_select_denoised_respects_ceiling() -> None:
    denoised = [
        {"arch": _ARCH, "latency_ms": 13.0, "denoised_mean": 0.70, "denoised_std": 0.01},
        {"arch": _ARCH, "latency_ms": 12.0, "denoised_mean": 0.60, "denoised_std": 0.01},
    ]
    w = select_denoised(denoised, t_max=12.75)  # the 0.70 arch is infeasible
    assert w["latency_ms"] == 12.0


# ---- driver (stubbed fine-tune; averaging + resume) --------------------------

def test_denoise_archs_averages_seeds(monkeypatch) -> None:
    def fake(arch, **kw):  # noqa: ANN001, ANN003
        return {"map": {1: 0.60, 2: 0.66, 3: 0.63}[kw["seed"]], "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake)
    out = denoise_archs([_pt(0.65, 11.7)], head_weights="g.pt", seeds=(1, 2, 3), device="cpu")
    assert out[0]["denoised_maps"] == [0.60, 0.66, 0.63]
    assert out[0]["denoised_mean"] == pytest.approx(0.63)
    assert out[0]["cached_acc"] == 0.65  # the biased single-seed value it replaces


def test_denoise_archs_resumes_from_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "denoise_cache.jsonl"
    calls = {"n": 0}

    def fake(arch, **kw):  # noqa: ANN001, ANN003
        calls["n"] += 1
        return {"map": 0.62, "map50": 0.9}

    monkeypatch.setattr("eval.shortft.short_finetune", fake)
    denoise_archs([_pt(0.65, 11.7)], head_weights="g.pt", seeds=(1, 2, 3), cache=cache)
    assert calls["n"] == 3           # one arch × three seeds
    assert cache.exists() and len(cache.read_text().splitlines()) == 3

    # second run reads the cache -> zero new fine-tunes
    denoise_archs([_pt(0.65, 11.7)], head_weights="g.pt", seeds=(1, 2, 3), cache=cache)
    assert calls["n"] == 3           # unchanged: fully resumed
