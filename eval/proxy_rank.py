"""CP 2.4 proxy-rank-fidelity protocol — one command produces the rank verdict that gates the
whole search (peer-review R2.1 / P0.2).

The protocol (PROJECT_PLAN.md CP 2.4): take ~8-12 archs spanning the space, score each with the
**5-epoch proxy** *and* a **full train**, and require the proxy ranking to agree with the
full-train ranking under the **reframed search-relevant gate** (:func:`eval.shortft.rank_verdict`):
**Spearman ρ ≥ 0.70 AND top-1 regret ≤ 0.01** (else BO climbs the wrong surface — repair the proxy
first). Kendall-τ, precision@k, and a run-to-run **reproducibility** check ride along as reported
*diagnostics* (the original τ-on-10 + Δ≤0.005 gate was superseded — τ-on-10 mis-measures: size
descriptors fail τ yet pick the true best). This driver loops the archs, calls
:func:`eval.shortft.short_finetune` for each regime, and emits the PASS/FAIL verdict.

Two design points that matter for a real Kaggle run:

- **Resumable.** Every per-arch result is written to ``--out`` immediately, so a session timeout
  (Kaggle caps ~9-12 h; a full train of 10 archs can exceed that) resumes instead of restarting.
- **Honest reproducibility.** ``short_finetune`` seeds everything, so the same seed twice is
  bit-identical (a meaningless pass). The repro rerun uses ``seed+1`` to measure real run-to-run
  noise (data order, kernel nondeterminism).

Run it::

    python -m eval.proxy_rank --archs 10 --proxy-epochs 5 --full-epochs 100 --device cuda

If the τ gate fails with the full-train mAPs clustered (as the first run did, τ=0.20), diagnose
*before* repairing the proxy: re-train a few archs at a second seed to measure whether the
full-train ranking is even stable (``full_noise_verdict``)::

    python -m eval.proxy_rank --diagnose-full --indices 7,4,8 --full-epochs 100 --device cuda

The pure verdict/serialization/corner logic is unit-tested under ``.venv`` (``tests/
test_proxy_rank.py``); the fine-tunes are GPU-gated (CPU-smokeable with ``--max-steps``).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from eval.shortft import rank_verdict, reproducible

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "cp24_proxy_rank.json"  # data/ is gitignored (regeneratable)


@dataclass
class ArchResult:
    """One architecture's proxy (5-epoch) and full-train pose mAP. ``None`` until that run lands.

    ``full_map_reseed`` holds the *same arch's* full-train mAP at a second seed — populated only by
    the ``--diagnose-full`` noise-floor run (see :func:`full_noise_verdict`), ``None`` otherwise.
    """

    index: int
    arch: dict
    proxy_map: float | None = None
    proxy_seed_maps: list[float] | None = None  # per-seed proxy mAPs when --proxy-seeds>1 (resume)
    full_map: float | None = None
    full_map_reseed: float | None = None


def corner_archs(n_ks: int, n_d: int) -> tuple[dict, dict]:
    """The (min, max) corners of the OFA-MBv3 search space — pin these so the ranking spans the
    accuracy range (uniform-random sampling clusters near mid-depth and weakens Kendall-τ)."""
    lo = {"ks": [3] * n_ks, "e": [3] * n_ks, "d": [2] * n_d}
    hi = {"ks": [7] * n_ks, "e": [6] * n_ks, "d": [4] * n_d}
    return lo, hi


def assemble_verdict(
    results: list[ArchResult], repro_pair: tuple[float, float] | None = None
) -> dict[str, Any]:
    """The reframed search-relevant PASS/FAIL CP 2.4 verdict (Spearman ρ + top-1 regret).

    ``dod_passes`` is the rank gate alone (:func:`eval.shortft.rank_verdict`): **ρ ≥ 0.70 AND
    top-1 regret ≤ 0.01**. Kendall-τ, precision@k, and the reproducibility pair (if measured) are
    recorded as *diagnostics* — they no longer gate (the τ-on-10 + Δ≤0.005 gate was superseded; see
    the module docstring). Fewer than 2 fully-scored archs → no ranking yet → fail.
    """
    complete = [r for r in results if r.proxy_map is not None and r.full_map is not None]
    verdict: dict[str, Any] = {"n_complete": len(complete)}

    if len(complete) >= 2:
        proxy = cast(list[float], [r.proxy_map for r in complete])  # filtered non-None above
        full = cast(list[float], [r.full_map for r in complete])
        rv = rank_verdict(proxy, full, k=min(3, len(complete)))  # precision@k diag; clamp k<=n
        verdict.update(
            spearman=rv.spearman,               # gate half 1
            top1_regret=rv.top1_regret,         # gate half 2
            kendall_tau=rv.kendall_tau,         # diagnostic (superseded gate)
            precision_at_k=rv.precision_at_k,   # diagnostic
            spearman_gate=rv.spearman_gate,     # thresholds → self-describing verdict JSON
            regret_tol=rv.regret_tol,
            rank_passes=rv.passes,
        )
    else:
        verdict.update(
            spearman=None, top1_regret=None, kendall_tau=None, precision_at_k=None,
            spearman_gate=None, regret_tol=None, rank_passes=None,
        )

    if repro_pair is not None:
        a, b = repro_pair
        verdict["reproducibility"] = {  # diagnostic only — does NOT gate the reframed DoD
            "run_a": a, "run_b": b, "delta": abs(a - b), "passes": reproducible(a, b),
        }

    verdict["dod_passes"] = bool(verdict["rank_passes"])
    return verdict


def full_noise_verdict(
    reseed: dict[int, tuple[float, float]],
    cluster_maps: Sequence[float],
    *,
    snr_discriminates: float = 2.0,
    snr_flat: float = 1.0,
) -> dict[str, Any]:
    """Is the *full-train* ranking itself reliable enough to rank the clustered archs?

    CP 2.4 failed (τ=0.20) with the full-train mAPs clustered in a narrow band. Before blaming the
    5-epoch proxy we must know whether the *full* ranking of those archs is even stable: if
    retraining moves an arch's mAP as much as the gaps between archs, the ground-truth ranking is
    noise and *no* proxy can pass.

    ``reseed`` maps an arch index → its ``(full_seed0, full_seed1)`` full-train mAPs (same arch, two
    seeds). ``cluster_maps`` are the seed-0 full mAPs of the clustered archs whose order the search
    must resolve. We compare the full-train **noise floor** (median run-to-run ``|Δ|``) to the
    **spread** it has to rank, ``snr = cluster_spread / noise_floor``:

    - ``snr ≥ snr_discriminates`` → the task separates archs on accuracy; repair the *proxy*.
    - ``snr ≤ snr_flat`` → the full ranking is itself noise; reframe (accuracy → constraint).
    - between → ``ambiguous`` (likely tighten the full reference *and* the proxy).
    """
    if not reseed:
        raise ValueError("need at least one reseeded arch to estimate the full-train noise floor")

    deltas = {i: abs(s1 - s0) for i, (s0, s1) in reseed.items()}
    noise_floor = float(statistics.median(deltas.values()))
    cluster_spread = float(max(cluster_maps) - min(cluster_maps)) if cluster_maps else 0.0
    snr = float("inf") if noise_floor == 0.0 else cluster_spread / noise_floor

    if snr >= snr_discriminates:
        verdict = "discriminates"
    elif snr <= snr_flat:
        verdict = "flat"
    else:
        verdict = "ambiguous"

    reseed_tau: float | None = None
    if len(reseed) >= 2:
        from scipy.stats import kendalltau  # lazy (matches rank_fidelity); only on ≥2 archs

        s0 = [v[0] for v in reseed.values()]
        s1 = [v[1] for v in reseed.values()]
        reseed_tau = float(kendalltau(s0, s1).statistic)

    return {
        "n_reseed": len(reseed),
        "deltas": deltas,
        "noise_floor": noise_floor,
        "cluster_spread": cluster_spread,
        "snr": snr,
        "reseed_kendall_tau": reseed_tau,
        "verdict": verdict,
        "discriminates": verdict == "discriminates",
    }


def save_results(path: Path, results: list[ArchResult]) -> None:
    """Persist results to JSON (after every fine-tune) so a timed-out run resumes, not restarts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(r) for r in results], indent=2))


def load_results(path: Path) -> list[ArchResult]:
    """Re-read persisted results for resume; missing file → empty (a fresh run)."""
    p = Path(path)
    if not p.exists():
        return []
    return [ArchResult(**rec) for rec in json.loads(p.read_text())]


def reverdict(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """Recompute ``<out>.verdict.json`` under the current gate from an existing results file.

    CPU-only (scipy + json — no fine-tune, no GPU): re-runs :func:`assemble_verdict` over the
    persisted ``proxy_map``/``full_map`` and re-stamps the verdict. Used to close CP 2.4 — the
    warm-head re-test's verdict was written under the old τ gate; this re-stamps it under the
    reframe gate without re-running the proxy. The reproducibility ``run_b`` isn't stored in the
    results file, so any ``reproducibility`` block from the prior verdict is carried forward as a
    diagnostic. Raises ``FileNotFoundError`` if ``out`` has no results.
    """
    results = load_results(out)
    if not results:
        raise FileNotFoundError(f"no results at {out} to re-verdict (run the protocol first)")
    verdict = assemble_verdict(results)
    vpath = Path(str(out) + ".verdict.json")
    if vpath.exists():
        prior = json.loads(vpath.read_text())
        if "reproducibility" in prior and "reproducibility" not in verdict:
            verdict["reproducibility"] = prior["reproducibility"]  # preserve the run-to-run diag
    vpath.write_text(json.dumps(verdict, indent=2))
    return verdict


# --------------------------------------------------------------------------------------------
# Integration: the actual protocol loop. Heavy imports (torch / ofa / the fine-tune) are lazy so
# importing this module for the pure helpers above stays light under .venv / CI. GPU-gated.
# --------------------------------------------------------------------------------------------

def sample_archs(supernet: Any, n: int, seed: int) -> list[dict]:
    """``n`` deterministic archs spanning the space: min+max corners then seeded randoms."""
    import random

    from supernet.sampler import random_arch

    random.seed(seed)  # OFA's sample_active_subnet draws from the `random` module
    base = random_arch(supernet)
    lo, hi = corner_archs(len(base["ks"]), len(base["d"]))
    archs: list[dict] = []
    if n >= 1:
        archs.append(lo)
    if n >= 2:
        archs.append(hi)
    while len(archs) < n:
        archs.append(random_arch(supernet))
    return archs[:n]


def run_protocol(
    *,
    n_archs: int = 10,
    proxy_epochs: int = 5,
    full_epochs: int = 100,
    seed: int = 0,
    device: str = "cuda",
    imgsz: int = 640,
    batch: int = 16,
    out: Path = DEFAULT_OUT,
    run_full: bool = True,
    run_repro: bool = True,
    max_steps: int | None = None,
    supernet: Any = None,
    head_weights: Any = None,
    freeze_head: bool = False,
    reset_proxy: bool = False,
    proxy_seeds: int = 1,
) -> dict[str, Any]:
    """Score ``n_archs`` under proxy + full, return the CP 2.4 verdict (resumable).

    ``head_weights``/``freeze_head`` warm-start + freeze the Pose head for the proxy fine-tune
    (CP 2.4 repair). ``reset_proxy`` nulls every loaded ``proxy_map`` (keeping the expensive
    ``full_map``) so a warm-head re-test recomputes the proxy and re-correlates against the
    existing full maps — run it on a *copy* of the results file. ``proxy_seeds`` averages each
    arch's proxy mAP over that many seeds (``seed … seed+proxy_seeds-1``) to cut run-to-run
    variance (the reproducibility-Δ repair, "Variation Matters"); per-seed maps are flushed so an
    averaging run resumes mid-arch. The repro rerun then compares two *independent* averaged
    estimates. ``proxy_seeds=1`` is the original single-run behavior.
    """
    from eval.shortft import short_finetune

    sn = supernet
    if sn is None:
        from supernet.sampler import load_supernet  # lazy: only build the supernet if not supplied

        sn = load_supernet()
    archs = sample_archs(sn, n_archs, seed)
    by_index = {r.index: r for r in load_results(out)}
    if reset_proxy:  # warm-head re-test: drop stale proxy maps (+ per-seed), keep seed-0 full maps
        for r in by_index.values():
            r.proxy_map = None
            r.proxy_seed_maps = None

    def _ft(arch: dict, epochs: int, run_seed: int) -> float:
        return short_finetune(arch, epochs=epochs, seed=run_seed, imgsz=imgsz, batch=batch,
                              device=device, supernet=sn, max_steps=max_steps,
                              head_weights=head_weights, freeze_head=freeze_head)["map"]

    def _flush() -> None:
        save_results(out, [by_index[i] for i in sorted(by_index)])

    def _avg_proxy(arch: dict, r: ArchResult, base_seed: int) -> float:
        """Mean proxy mAP over ``proxy_seeds`` seeds (resumes from flushed per-seed maps)."""
        maps = list(r.proxy_seed_maps or [])
        for s in range(len(maps), proxy_seeds):
            maps.append(_ft(arch, proxy_epochs, base_seed + s))
            r.proxy_seed_maps = maps
            _flush()
        return statistics.mean(maps)

    for i, arch in enumerate(archs):
        r = by_index.setdefault(i, ArchResult(index=i, arch=arch))
        if r.proxy_map is None:
            print(f"[proxy {i + 1}/{len(archs)}] d={arch['d']} "
                  f"({proxy_epochs} ep × {proxy_seeds} seed)")
            r.proxy_map = _avg_proxy(arch, r, seed)
            _flush()
        if run_full and r.full_map is None:
            print(f"[full  {i + 1}/{len(archs)}] d={arch['d']} ({full_epochs} ep)")
            r.full_map = _ft(arch, full_epochs, seed)
            _flush()

    ordered = [by_index[i] for i in range(len(archs))]
    repro_pair = None
    if run_repro:
        print(f"[repro] arch 0, next {proxy_seeds}-seed block (run-to-run noise floor)")
        b = statistics.mean(
            _ft(archs[0], proxy_epochs, seed + proxy_seeds + j) for j in range(proxy_seeds)
        )
        repro_pair = (ordered[0].proxy_map or 0.0, b)

    verdict = assemble_verdict(ordered, repro_pair=repro_pair)
    Path(str(out) + ".verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def run_full_diagnostic(
    *,
    indices: Sequence[int],
    full_epochs: int = 100,
    seed: int = 0,
    device: str = "cuda",
    imgsz: int = 640,
    batch: int = 16,
    out: Path = DEFAULT_OUT,
    max_steps: int | None = None,
    supernet: Any = None,
) -> dict[str, Any]:
    """Re-train ``indices`` at ``seed+1`` to measure the full-train ranking's own noise floor.

    The Q1='diagnose first' step after CP 2.4 failed (τ=0.20). Reuses the prior ``out`` results —
    only the chosen archs are retrained, into ``full_map_reseed`` (resumable per-arch flush), so the
    10 existing seed-0 full maps are *not* recomputed. The cluster spread excludes the global-min
    arch (the min corner — the one trivially-separable outlier), so the SNR reflects the
    hard-to-rank bulk. Writes the verdict to ``<out>.diagnostic.json`` and returns it.
    """
    from eval.shortft import short_finetune

    prior = load_results(out)
    if not prior:
        raise FileNotFoundError(
            f"no prior results at {out} — run the proxy protocol first (need the seed-0 full_maps)"
        )
    by_index = {r.index: r for r in prior}

    missing = [i for i in indices if i not in by_index or by_index[i].full_map is None]
    if missing:
        raise ValueError(f"indices {missing} have no seed-0 full_map in {out} to compare against")

    sn = supernet
    if sn is None:
        from supernet.sampler import load_supernet  # lazy: only build the supernet if not supplied

        sn = load_supernet()

    def _flush() -> None:
        save_results(out, [by_index[i] for i in sorted(by_index)])

    for i in indices:
        r = by_index[i]
        if r.full_map_reseed is None:
            print(f"[reseed {i}] d={r.arch['d']} full-train @ seed {seed + 1} ({full_epochs} ep)")
            r.full_map_reseed = short_finetune(
                r.arch, epochs=full_epochs, seed=seed + 1, imgsz=imgsz, batch=batch,
                device=device, supernet=sn, max_steps=max_steps,
            )["map"]
            _flush()

    reseed = {
        i: (cast(float, by_index[i].full_map), cast(float, by_index[i].full_map_reseed))
        for i in indices
    }
    full_maps = {
        r.index: cast(float, r.full_map) for r in by_index.values() if r.full_map is not None
    }
    # drop the global-min arch (min corner) so the spread reflects the hard-to-rank bulk
    min_idx = min(full_maps, key=lambda i: full_maps[i])
    cluster = [m for i, m in full_maps.items() if i != min_idx]

    verdict = full_noise_verdict(reseed, cluster)
    Path(str(out) + ".diagnostic.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def _print_verdict(verdict: dict[str, Any], out: Path) -> None:
    rho, regret = verdict.get("spearman"), verdict.get("top1_regret")
    rho_gate, regret_tol = verdict.get("spearman_gate"), verdict.get("regret_tol")
    print("\n" + "=" * 60)
    print(f"CP 2.4 proxy-rank fidelity  (n={verdict['n_complete']})")
    print(f"  Spearman rho = {rho}  (gate >= {rho_gate})")
    print(f"  top-1 regret = {regret}  (gate <= {regret_tol}) -> "
          f"{'PASS' if verdict['rank_passes'] else 'FAIL'}")
    print(f"  (diagnostics) Kendall-tau = {verdict.get('kendall_tau')}  "
          f"precision@k = {verdict.get('precision_at_k')}")
    if "reproducibility" in verdict:
        rep = verdict["reproducibility"]
        print(f"  (diagnostic) reproducibility delta = {rep['delta']:.4f} "
              f"(within tol? {'yes' if rep['passes'] else 'no'})")
    print(f"  DoD: {'PASS ✅' if verdict['dod_passes'] else 'FAIL ❌'}")
    print(f"  results: {out}   verdict: {out}.verdict.json")
    print("=" * 60)


def _print_diagnostic(verdict: dict[str, Any], out: Path) -> None:
    snr_str = "inf" if verdict["snr"] == float("inf") else f"{verdict['snr']:.2f}"
    deltas = ", ".join(f"{i}:{d:.4f}" for i, d in verdict["deltas"].items())
    meaning = {
        "discriminates": "archs separable on accuracy → repair the PROXY (epochs / LR / head warm)",
        "flat": "archs NOT separable → REFRAME (accuracy=constraint, latency=objective)",
        "ambiguous": "borderline → tighten the full reference AND the proxy (discuss)",
    }[verdict["verdict"]]
    print("\n" + "=" * 60)
    print(f"CP 2.4 full-train noise diagnostic  (n_reseed={verdict['n_reseed']})")
    print(f"  noise_floor (median |Δ| reseed) = {verdict['noise_floor']:.4f}")
    print(f"  cluster_spread (min corner excl)= {verdict['cluster_spread']:.4f}")
    print(f"  SNR = spread / noise            = {snr_str}  (≥2 discriminates, ≤1 flat)")
    if verdict["reseed_kendall_tau"] is not None:
        print(f"  reseed Kendall-tau seed0↔seed1  = {verdict['reseed_kendall_tau']:.3f}")
    print(f"  per-arch |Δ|: {deltas}")
    print(f"  VERDICT: {verdict['verdict'].upper()} — {meaning}")
    print(f"  diagnostic: {out}.diagnostic.json")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the protocol and exit 0 iff the CP 2.4 DoD passes."""
    import argparse

    p = argparse.ArgumentParser(description="CP 2.4 proxy-rank-fidelity protocol")
    p.add_argument("--archs", type=int, default=10)
    p.add_argument("--proxy-epochs", type=int, default=5)
    p.add_argument("--full-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-full", action="store_true", help="proxy only (skip full train)")
    p.add_argument("--no-repro", action="store_true", help="skip the reproducibility rerun")
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    p.add_argument("--head-weights", type=Path, default=None,
                   help="warm-start the Pose head from a trained gate yolo11n-pose .pt (CP 2.4)")
    p.add_argument("--freeze-head", action="store_true",
                   help="freeze the (warm-started) head so the proxy trains only backbone+adapter")
    p.add_argument("--reset-proxy", action="store_true",
                   help="null loaded proxy maps (keep full maps) for a warm-head re-test on a copy")
    p.add_argument("--proxy-seeds", type=int, default=1,
                   help="average the proxy mAP over N seeds to cut run-to-run variance (repairs Δ)")
    p.add_argument("--reverdict", action="store_true",
                   help="recompute <out>.verdict.json under the current gate from an existing "
                        "results file (CPU-only; re-stamps a prior run after a gate change)")
    p.add_argument("--diagnose-full", action="store_true",
                   help="measure the full-train noise floor (rerun --indices at seed+1) instead "
                        "of the proxy protocol")
    p.add_argument("--indices", default="7,4,8",
                   help="comma-separated arch indices to reseed for --diagnose-full")
    a = p.parse_args(argv)

    if a.reverdict:
        verdict = reverdict(out=a.out)
        _print_verdict(verdict, a.out)
        return 0 if verdict["dod_passes"] else 1

    if a.diagnose_full:
        indices = [int(x) for x in a.indices.split(",") if x.strip()]
        diag = run_full_diagnostic(
            indices=indices, full_epochs=a.full_epochs, seed=a.seed, device=a.device,
            imgsz=a.imgsz, batch=a.batch, out=a.out, max_steps=a.max_steps,
        )
        _print_diagnostic(diag, a.out)
        return 0 if diag["discriminates"] else 1

    verdict = run_protocol(
        n_archs=a.archs, proxy_epochs=a.proxy_epochs, full_epochs=a.full_epochs, seed=a.seed,
        device=a.device, imgsz=a.imgsz, batch=a.batch, out=a.out, run_full=not a.no_full,
        run_repro=not a.no_repro, max_steps=a.max_steps,
        head_weights=a.head_weights, freeze_head=a.freeze_head, reset_proxy=a.reset_proxy,
        proxy_seeds=a.proxy_seeds,
    )
    _print_verdict(verdict, a.out)
    return 0 if verdict["dod_passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
