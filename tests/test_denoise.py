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


# ---- winner-v1 serialization (CP 3.5 close) ----------------------------------

def _enriched(acc: float, lat: float, d: list[int], *, mean: float, std: float) -> dict:
    arch = {"ks": [3] * 20, "e": [3] * 20, "d": d}
    return {"arch": arch, "acc": acc, "latency_ms": lat, "acc_eff": acc, "cached_acc": acc,
            "seeds": [1, 2, 3], "denoised_maps": [mean] * 3, "denoised_mean": mean,
            "denoised_std": std, "method": "bo", "seed": 0}


def test_denoised_winner_record_uses_mean_and_passes_repro() -> None:
    from search.denoise import denoised_winner_record
    from search.space import decode

    top = _enriched(0.6376, 12.65, [2, 2, 4, 4, 2], mean=0.6236, std=0.0032)
    knee = _enriched(0.6238, 11.21, [2, 2, 4, 3, 3], mean=0.6101, std=0.0049)
    fast = _enriched(0.6336, 10.15, [2, 4, 3, 4, 4], mean=0.5706, std=0.0019)
    payload = {"t_max_ms": 12.75, "candidates": [top, knee, fast]}

    rec = denoised_winner_record(knee, payload, baseline_latency_ms=12.75,
                                 old_alpha_arch=top["arch"])
    assert rec["acc"] == 0.6101                 # reference acc = the de-noised MEAN
    assert rec["cached_acc"] == 0.6238          # the biased single-seed value it replaces
    assert rec["reproduction"]["passes"] is True   # |0.6238-0.6101| = 0.0137 <= 0.020
    assert rec["denoised_rank"] == 1            # knee is 2nd by mean (top 0.6236 is 1st)
    assert decode(rec["vector"]) == rec["arch"]    # winner vector round-trips
    # provenance of the corrected curse
    assert rec["winners_curse"]["rejected_single_seed_alpha"]["arch_d"] == [2, 2, 4, 4, 2]
    assert rec["winners_curse"]["averted_second_curse"]["arch_d"] == [2, 4, 3, 4, 4]
    assert rec["vs_yolo11n"]["latency_speedup_pct"] > 10   # 11.21 vs 12.75 ~ 12%


def test_denoised_winner_record_repro_fails_when_far() -> None:
    from search.denoise import denoised_winner_record
    # cached 0.65 but de-noised mean 0.61 -> |delta| 0.04 > band 0.02 -> repro fails
    cursed = _enriched(0.65, 11.7, [2, 2, 4, 4, 3], mean=0.61, std=0.025)
    rec = denoised_winner_record(cursed, {"t_max_ms": 12.75, "candidates": [cursed]},
                                 repro_band=0.02)
    assert rec["reproduction"]["passes"] is False
