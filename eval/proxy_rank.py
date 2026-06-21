"""CP 2.4 proxy-rank-fidelity protocol — one command produces the Kendall-τ verdict that gates
the whole search (peer-review R2.1 / P0.2).

The protocol (PROJECT_PLAN.md CP 2.4): take ~8-12 archs spanning the space, score each with the
**5-epoch proxy** *and* a **full train**, and require the two rankings to agree at
**Kendall-τ ≥ 0.7** (else BO climbs the wrong surface — repair the proxy first). Plus a
**reproducibility** check (one arch run twice within 0.5 pts). This driver loops the archs,
calls :func:`eval.shortft.short_finetune` for each regime, and emits the PASS/FAIL verdict.

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

from eval.shortft import rank_fidelity, reproducible

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
    """Combine rank fidelity (the gate) + reproducibility into a single PASS/FAIL CP 2.4 verdict.

    ``dod_passes`` requires the Kendall-τ gate to pass; if a reproducibility pair was measured it
    must pass too. Fewer than 2 fully-scored archs → no ranking yet → fail.
    """
    complete = [r for r in results if r.proxy_map is not None and r.full_map is not None]
    verdict: dict[str, Any] = {"n_complete": len(complete)}

    if len(complete) >= 2:
        proxy = cast(list[float], [r.proxy_map for r in complete])  # filtered non-None above
        full = cast(list[float], [r.full_map for r in complete])
        rf = rank_fidelity(proxy, full)
        verdict.update(kendall_tau=rf.kendall_tau, spearman=rf.spearman, rank_passes=rf.passes)
    else:
        verdict.update(kendall_tau=None, spearman=None, rank_passes=None)

    repro_ok: bool | None = None
    if repro_pair is not None:
        a, b = repro_pair
        repro_ok = reproducible(a, b)
        verdict["reproducibility"] = {
            "run_a": a, "run_b": b, "delta": abs(a - b), "passes": repro_ok,
        }

    verdict["dod_passes"] = bool(verdict["rank_passes"]) and (repro_ok is None or repro_ok)
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
) -> dict[str, Any]:
    """Score ``n_archs`` under proxy + full, return the CP 2.4 verdict (resumable).

    ``head_weights``/``freeze_head`` warm-start + freeze the Pose head for the proxy fine-tune
    (CP 2.4 repair). ``reset_proxy`` nulls every loaded ``proxy_map`` (keeping the expensive
    ``full_map``) so a warm-head re-test recomputes the proxy and re-correlates against the
    existing full maps — run it on a *copy* of the results file.
    """
    from eval.shortft import short_finetune

    sn = supernet
    if sn is None:
        from supernet.sampler import load_supernet  # lazy: only build the supernet if not supplied

        sn = load_supernet()
    archs = sample_archs(sn, n_archs, seed)
    by_index = {r.index: r for r in load_results(out)}
    if reset_proxy:  # warm-head re-test: drop stale proxy maps, keep the seed-0 full maps
        for r in by_index.values():
            r.proxy_map = None

    def _ft(arch: dict, epochs: int, run_seed: int) -> float:
        return short_finetune(arch, epochs=epochs, seed=run_seed, imgsz=imgsz, batch=batch,
                              device=device, supernet=sn, max_steps=max_steps,
                              head_weights=head_weights, freeze_head=freeze_head)["map"]

    def _flush() -> None:
        save_results(out, [by_index[i] for i in sorted(by_index)])

    for i, arch in enumerate(archs):
        r = by_index.setdefault(i, ArchResult(index=i, arch=arch))
        if r.proxy_map is None:
            print(f"[proxy {i + 1}/{len(archs)}] d={arch['d']} ({proxy_epochs} ep)")
            r.proxy_map = _ft(arch, proxy_epochs, seed)
            _flush()
        if run_full and r.full_map is None:
            print(f"[full  {i + 1}/{len(archs)}] d={arch['d']} ({full_epochs} ep)")
            r.full_map = _ft(arch, full_epochs, seed)
            _flush()

    ordered = [by_index[i] for i in range(len(archs))]
    repro_pair = None
    if run_repro:
        print("[repro] arch 0, seed+1 (run-to-run noise floor)")
        repro_pair = (ordered[0].proxy_map or 0.0, _ft(archs[0], proxy_epochs, seed + 1))

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
    tau, rho = verdict.get("kendall_tau"), verdict.get("spearman")
    print("\n" + "=" * 60)
    print(f"CP 2.4 proxy-rank fidelity  (n={verdict['n_complete']})")
    print(f"  Kendall-tau = {tau}  (gate >= 0.7) -> {'PASS' if verdict['rank_passes'] else 'FAIL'}")
    print(f"  Spearman    = {rho}")
    if "reproducibility" in verdict:
        rep = verdict["reproducibility"]
        print(f"  reproducibility delta = {rep['delta']:.4f} (<= 0.005) -> "
              f"{'PASS' if rep['passes'] else 'FAIL'}")
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
    p.add_argument("--diagnose-full", action="store_true",
                   help="measure the full-train noise floor (rerun --indices at seed+1) instead "
                        "of the proxy protocol")
    p.add_argument("--indices", default="7,4,8",
                   help="comma-separated arch indices to reseed for --diagnose-full")
    a = p.parse_args(argv)

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
    )
    _print_verdict(verdict, a.out)
    return 0 if verdict["dod_passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
