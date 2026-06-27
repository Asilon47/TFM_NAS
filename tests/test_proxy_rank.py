"""Tests for eval/proxy_rank.py — the CP 2.4 proxy-rank-fidelity protocol driver.

Pure pieces only (scipy/stdlib → ``.venv`` / CI): the PASS/FAIL verdict assembly, the
span-corner archs, and the JSON resume round-trip. The actual proxy/full fine-tunes
(``run_protocol``) need a GPU + dataset and are integration-smoked under ``.venv-nas``
(``eval/proxy_rank.py``'s ``--max-steps`` CPU path).
"""
import json

import pytest

pytest.importorskip("scipy")

from eval.proxy_rank import (  # noqa: E402
    ArchResult,
    assemble_verdict,
    corner_archs,
    full_noise_verdict,
    load_results,
    reverdict,
    run_full_diagnostic,
    run_protocol,
    save_results,
)


def _r(index: int, proxy: float | None, full: float | None) -> ArchResult:
    return ArchResult(index=index, arch={"d": [index]}, proxy_map=proxy, full_map=full)


# --- assemble_verdict: reframed search-relevant DoD (Spearman + top-1 regret; CP 2.4 close). The
#     Kendall-τ-on-10 + Δ≤0.005 gate was superseded — τ-on-10 mis-measures (size descriptors fail τ
#     yet pick the true best), so τ/precision@k/reproducibility are now reported diagnostics. ---

def test_verdict_passes_when_rank_concordant_and_regret_zero():
    results = [_r(0, 0.11, 0.10), _r(1, 0.22, 0.20), _r(2, 0.29, 0.30), _r(3, 0.42, 0.40)]
    v = assemble_verdict(results, repro_pair=(0.30, 0.302))
    assert v["n_complete"] == 4
    assert v["spearman"] == pytest.approx(1.0)     # gate half 1: monotone agreement
    assert v["top1_regret"] == pytest.approx(0.0)  # gate half 2: proxy's #1 IS the true #1
    assert v["kendall_tau"] == pytest.approx(1.0)  # diagnostic (superseded gate)
    assert v["rank_passes"] is True
    assert v["reproducibility"]["passes"] is True  # reported diagnostic
    assert v["dod_passes"] is True


def test_verdict_fails_when_ranking_discordant():
    results = [_r(0, 0.40, 0.10), _r(1, 0.30, 0.20), _r(2, 0.20, 0.30), _r(3, 0.10, 0.40)]
    v = assemble_verdict(results, repro_pair=(0.30, 0.30))
    assert v["spearman"] == pytest.approx(-1.0)
    assert v["rank_passes"] is False
    assert v["dod_passes"] is False  # ranking fails → whole DoD fails


def test_verdict_reproducibility_is_diagnostic_not_gate():
    # The reframe change: rank is good (concordant, regret 0) but the proxy is noisy run-to-run.
    # Under the old gate this failed the DoD; under the reframe, reproducibility is a REPORTED
    # diagnostic, not a gate, so the DoD still PASSES.
    results = [_r(0, 0.11, 0.10), _r(1, 0.22, 0.20), _r(2, 0.29, 0.30), _r(3, 0.42, 0.40)]
    v = assemble_verdict(results, repro_pair=(0.30, 0.40))  # 10 pts apart → repro "fails"
    assert v["rank_passes"] is True
    assert v["reproducibility"]["passes"] is False  # still computed + reported
    assert v["dod_passes"] is True  # reframe: reproducibility does not gate the DoD


def test_verdict_handles_incomplete_results():
    # fewer than 2 archs have BOTH proxy+full → no rank correlation yet.
    results = [_r(0, 0.11, 0.10), _r(1, 0.22, None)]
    v = assemble_verdict(results)
    assert v["n_complete"] == 1
    assert v["kendall_tau"] is None
    assert v["spearman"] is None
    assert v["top1_regret"] is None
    assert v["dod_passes"] is False


# --- reverdict: re-stamp a prior run's verdict under the reframe gate, no GPU (CP 2.4 close) ---

def test_reverdict_recomputes_gate_and_preserves_repro(tmp_path):
    out = tmp_path / "cp24.json"
    save_results(out, [_r(0, 0.11, 0.10), _r(1, 0.22, 0.20), _r(2, 0.29, 0.30), _r(3, 0.42, 0.40)])
    # a stale verdict from the OLD gate, carrying a (failing) reproducibility diagnostic to preserve
    vpath = tmp_path / "cp24.json.verdict.json"
    vpath.write_text(json.dumps({
        "n_complete": 4, "kendall_tau": 1.0, "rank_passes": False,
        "reproducibility": {"run_a": 0.30, "run_b": 0.40, "delta": 0.10, "passes": False},
        "dod_passes": False,
    }))

    v = reverdict(out=out)

    assert v["spearman"] == pytest.approx(1.0)
    assert v["top1_regret"] == pytest.approx(0.0)
    assert v["rank_passes"] is True
    assert v["dod_passes"] is True  # reframe gate flips the stale FAIL to PASS
    assert v["reproducibility"]["delta"] == pytest.approx(0.10)  # carried from the prior verdict
    on_disk = json.loads(vpath.read_text())
    assert on_disk["dod_passes"] is True
    assert on_disk["reproducibility"]["passes"] is False  # preserved as a diagnostic


def test_reverdict_missing_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        reverdict(out=tmp_path / "nope.json")


# --- corner_archs: pin the min/max corners so the ranking spans the space ---

def test_corner_archs_min_and_max():
    lo, hi = corner_archs(n_ks=20, n_d=5)
    assert lo == {"ks": [3] * 20, "e": [3] * 20, "d": [2] * 5}
    assert hi == {"ks": [7] * 20, "e": [6] * 20, "d": [4] * 5}


# --- resume: JSON round-trip so a timed-out Kaggle run continues, not restarts ---

def test_results_json_round_trip(tmp_path):
    results = [_r(0, 0.11, 0.10), _r(1, 0.22, None)]
    out = tmp_path / "cp24.json"
    save_results(out, results)
    loaded = load_results(out)
    assert loaded == results


def test_load_results_missing_file_is_empty(tmp_path):
    assert load_results(tmp_path / "nope.json") == []


# --- full_noise_verdict: is the full-train ranking itself reliable enough to rank the cluster? ---
# (CP 2.4 failed at tau=0.20 with full-train mAPs clustered in 0.823-0.850; before repairing the
#  proxy we must know whether the full-train ranking of those clustered archs is even stable.)

def test_full_noise_verdict_discriminates_when_noise_small():
    # full-train barely moves on reseed (<=0.002) but the cluster spans 0.023 → high SNR.
    reseed = {7: (0.822, 0.823), 4: (0.835, 0.833), 8: (0.846, 0.847)}
    cluster = [0.823, 0.846, 0.840, 0.836, 0.835, 0.831, 0.830, 0.829]  # spread 0.023
    v = full_noise_verdict(reseed, cluster)
    assert v["noise_floor"] == pytest.approx(0.001)  # median(0.001, 0.002, 0.001)
    assert v["cluster_spread"] == pytest.approx(0.023)
    assert v["snr"] == pytest.approx(0.023 / 0.001)
    assert v["verdict"] == "discriminates"
    assert v["discriminates"] is True


def test_full_noise_verdict_flat_when_noise_swamps_spread():
    # full-train moves ~0.025 on reseed — bigger than the cluster spread → the ranking is noise.
    reseed = {7: (0.822, 0.847), 4: (0.835, 0.860), 8: (0.846, 0.821)}
    cluster = [0.823, 0.846, 0.840, 0.836, 0.835, 0.831, 0.830, 0.829]  # spread 0.023
    v = full_noise_verdict(reseed, cluster)
    assert v["noise_floor"] == pytest.approx(0.025)
    assert v["snr"] < 1.0
    assert v["verdict"] == "flat"
    assert v["discriminates"] is False


def test_full_noise_verdict_ambiguous_between_thresholds():
    # SNR in (1, 2): noise comparable to the spread it must rank → neither clean call.
    reseed = {7: (0.822, 0.835), 4: (0.835, 0.848)}  # deltas 0.013, 0.013
    cluster = [0.823, 0.846]  # spread 0.023; snr = 0.023 / 0.013 = 1.77
    v = full_noise_verdict(reseed, cluster)
    assert 1.0 < v["snr"] < 2.0
    assert v["verdict"] == "ambiguous"
    assert v["discriminates"] is False


def test_full_noise_verdict_reports_per_arch_deltas_and_n():
    reseed = {7: (0.822, 0.823), 4: (0.835, 0.833)}
    v = full_noise_verdict(reseed, [0.82, 0.85])
    assert v["deltas"][7] == pytest.approx(0.001)
    assert v["deltas"][4] == pytest.approx(0.002)
    assert v["n_reseed"] == 2
    assert v["cluster_spread"] == pytest.approx(0.03)


def test_full_noise_verdict_requires_at_least_one_reseed():
    with pytest.raises(ValueError):
        full_noise_verdict({}, [0.82, 0.85])


# --- resume back-compat: the new full_map_reseed field must round-trip + load on old records ---

def test_arch_result_round_trip_with_reseed_field(tmp_path):
    results = [
        ArchResult(index=0, arch={"d": [2]}, proxy_map=0.5, full_map=0.78, full_map_reseed=0.77)
    ]
    out = tmp_path / "r.json"
    save_results(out, results)
    assert load_results(out) == results


def test_load_results_back_compat_without_reseed_field(tmp_path):
    # records written before full_map_reseed existed must still load (field defaults to None).
    out = tmp_path / "old.json"
    out.write_text('[{"index": 0, "arch": {"d": [2]}, "proxy_map": 0.5, "full_map": 0.78}]')
    loaded = load_results(out)
    assert loaded[0].full_map_reseed is None


# --- run_full_diagnostic orchestration: the heavy fine-tune is stubbed so the resume + guard +
#     verdict-writing path runs under .venv / CI (the real fine-tune is the Kaggle GPU step). ---

def test_run_full_diagnostic_missing_prior_raises(tmp_path):
    # nothing to compare against → fail loudly rather than silently re-running everything.
    with pytest.raises(FileNotFoundError):
        run_full_diagnostic(indices=[0], out=tmp_path / "nope.json", supernet=object())


def test_run_full_diagnostic_unknown_index_raises(tmp_path):
    out = tmp_path / "p.json"
    save_results(out, [_r(0, 0.5, 0.78)])
    with pytest.raises(ValueError):
        run_full_diagnostic(indices=[5], out=out, supernet=object())  # idx 5 not in the prior file


def test_run_full_diagnostic_reseeds_resumes_and_writes_verdict(tmp_path, monkeypatch):
    import eval.shortft as shortft_mod

    seeds_seen = []

    def _stub_finetune(arch, **kw):  # stand in for the GPU fine-tune
        seeds_seen.append(kw["seed"])
        return {"map": 0.80, "map50": 0.9}

    monkeypatch.setattr(shortft_mod, "short_finetune", _stub_finetune)

    out = tmp_path / "p.json"
    # idx0 = min corner (dropped from the cluster); idx1 already reseeded (resume must skip it);
    # idx2 still pending.
    save_results(out, [
        ArchResult(index=0, arch={"d": [2]}, proxy_map=0.5, full_map=0.70),
        ArchResult(index=1, arch={"d": [3]}, proxy_map=0.6, full_map=0.84, full_map_reseed=0.83),
        ArchResult(index=2, arch={"d": [4]}, proxy_map=0.6, full_map=0.85),
    ])

    v = run_full_diagnostic(indices=[1, 2], out=out, full_epochs=3, seed=0, supernet=object())

    # resume: idx1 already had a reseed → not retrained; only idx2 reseeded, at seed+1 = 1.
    assert seeds_seen == [1]
    reread = {r.index: r for r in load_results(out)}
    assert reread[2].full_map_reseed == 0.80
    assert reread[1].full_map_reseed == 0.83  # untouched by the resume
    # verdict persisted + structured (JSON stringifies the deltas keys, so compare scalar fields).
    diag = json.loads((tmp_path / "p.json.diagnostic.json").read_text())
    assert diag["n_reseed"] == v["n_reseed"] == 2
    assert diag["verdict"] == v["verdict"]
    assert diag["verdict"] in {"discriminates", "flat", "ambiguous"}


# --- warm-head re-test (CP 2.4 repair): --reset-proxy recomputes the proxy with a warm-started,
#     frozen head while preserving the expensive seed-0 full maps. Fine-tune stubbed → .venv/CI. ---

def test_run_protocol_reset_proxy_recomputes_warm_keeps_full(tmp_path, monkeypatch):
    import eval.proxy_rank as pr
    import eval.shortft as shortft_mod

    # stub the arch sampler (no ofa under .venv) and the fine-tune (no GPU).
    monkeypatch.setattr(pr, "sample_archs", lambda sn, n, seed: [{"d": [2]}, {"d": [4]}][:n])
    calls: list[dict] = []

    def _stub_finetune(arch, **kw):
        calls.append(kw)
        return {"map": 0.42, "map50": 0.5}

    monkeypatch.setattr(shortft_mod, "short_finetune", _stub_finetune)

    out = tmp_path / "p.json"
    save_results(out, [  # a prior run: both proxy AND full present
        ArchResult(index=0, arch={"d": [2]}, proxy_map=0.10, full_map=0.70),
        ArchResult(index=1, arch={"d": [4]}, proxy_map=0.20, full_map=0.85),
    ])

    v = run_protocol(n_archs=2, out=out, run_full=True, run_repro=False, reset_proxy=True,
                     head_weights="gate.pt", freeze_head=True, supernet=object())

    reread = {r.index: r for r in load_results(out)}
    # proxy recomputed with the warm head; the expensive seed-0 full maps are untouched.
    assert reread[0].proxy_map == 0.42 and reread[1].proxy_map == 0.42
    assert reread[0].full_map == 0.70 and reread[1].full_map == 0.85
    # only the 2 proxies were (re)run — full was skipped (present), no repro requested.
    assert len(calls) == 2
    assert all(c["head_weights"] == "gate.pt" and c["freeze_head"] is True for c in calls)
    assert v["n_complete"] == 2


# --- seed-averaging (CP 2.4 repair, "Variation Matters"): the proxy map is the mean over N seeds
#     to cut run-to-run variance. Resumable per-seed; backward-compatible (proxy_seeds=1). ---

def test_run_protocol_proxy_seeds_averages_over_seeds(tmp_path, monkeypatch):
    import eval.proxy_rank as pr
    import eval.shortft as shortft_mod

    monkeypatch.setattr(pr, "sample_archs", lambda sn, n, seed: [{"d": [2]}, {"d": [4]}][:n])
    seeds_seen: dict[tuple, list[int]] = {}

    def _stub_finetune(arch, **kw):  # seed-dependent so the mean is non-trivial
        seeds_seen.setdefault(tuple(arch["d"]), []).append(kw["seed"])
        return {"map": 0.10 * kw["seed"], "map50": 0.5}

    monkeypatch.setattr(shortft_mod, "short_finetune", _stub_finetune)

    out = tmp_path / "p.json"
    run_protocol(n_archs=2, out=out, run_full=False, run_repro=False, proxy_seeds=3,
                 seed=0, supernet=object())

    reread = {r.index: r for r in load_results(out)}
    # mean over seeds 0,1,2 of 0.1*seed = mean(0.0, 0.1, 0.2) = 0.1
    assert reread[0].proxy_map == pytest.approx(0.1)
    assert reread[1].proxy_map == pytest.approx(0.1)
    assert reread[0].proxy_seed_maps == pytest.approx([0.0, 0.1, 0.2])  # per-seed kept for resume
    assert seeds_seen[(2,)] == [0, 1, 2]


def test_run_protocol_proxy_seeds_resumes_midway(tmp_path, monkeypatch):
    import eval.proxy_rank as pr
    import eval.shortft as shortft_mod

    monkeypatch.setattr(pr, "sample_archs", lambda sn, n, seed: [{"d": [2]}][:n])
    seeds_seen: list[int] = []

    def _stub_finetune(arch, **kw):
        seeds_seen.append(kw["seed"])
        return {"map": 0.10 * kw["seed"], "map50": 0.5}

    monkeypatch.setattr(shortft_mod, "short_finetune", _stub_finetune)

    out = tmp_path / "p.json"
    # one of three seeds already done (proxy_map still None until all seeds land)
    save_results(out, [ArchResult(index=0, arch={"d": [2]}, proxy_seed_maps=[0.0])])

    run_protocol(n_archs=1, out=out, run_full=False, run_repro=False, proxy_seeds=3,
                 seed=0, supernet=object())

    assert seeds_seen == [1, 2]  # only the two missing seeds ran
    reread = {r.index: r for r in load_results(out)}
    assert reread[0].proxy_seed_maps == pytest.approx([0.0, 0.1, 0.2])
    assert reread[0].proxy_map == pytest.approx(0.1)
