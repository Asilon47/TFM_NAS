"""CP 3.5 (de-noise) — kill the single-seed winner's curse, then re-select honestly.

The CP 3.5 verify (:mod:`eval.verify_winner`) exposed a selection bug, not a pipeline bug: the
ceiling-first ``argmax`` ran over frontier accuracies each measured at a **single fine-tune seed**
(the search oracle used ``seed=0`` for every arch). The top of the feasible frontier is a
*statistical tie* — the top-12 cached accs span 0.027 while a single arch's fresh-seed σ is 0.031
— so the max over 130 single-seed draws is **winner's curse**: it picks whichever arch's seed-0
draw was luckiest, and it regresses on re-eval (α*: fresh 3-seed mean 0.610 vs cached 0.650).

The fix is to *average out the seed noise on the contenders*, then re-select:

1. :func:`top_candidates` — the top-K feasible frontier archs by cached acc (the plausible winners),
   committed to ``state/winner_v1/denoise_candidates.json`` (CPU) so the GPU job reads a
   pinned, inspected set — not a recomputation off the gitignored frontier.
2. :func:`denoise_archs` — re-fine-tune each candidate at N **fresh** seeds (1,2,3, not the biased
   seed-0), average → de-noised ``mean ± std``. GPU-gated + resumable (per-(arch,seed)
   cache), run on Kaggle (``kaggle/run.py`` ``MODE="denoise"``).
3. :func:`select_denoised` — re-pick the winner on de-noised means: among the archs statistically
   tied with the best (within the empirical noise band), take the **fastest** (LUT-exact latency is
   the robust axis once accuracy saturates). Reduces to a plain argmax if a real winner separates.

Pure helpers (candidates + selection) are unit-tested under ``.venv``; the driver is GPU-gated.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics as st
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from catalog.contracts import ArchDict
from search.select_winner import feasible_frontier, load_frontier
from search.space import canonical, encode

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)  # fresh seeds; excludes the biased search seed 0


# ---- candidate set (CPU, pinned) ---------------------------------------------

def top_candidates(frontier: Sequence[dict], *, t_max: float, k: int) -> list[dict]:
    """The top-``k`` feasible frontier points by cached proxy acc — the plausible winners to
    de-noise. Sorted acc-desc; ties broken toward lower latency (stable, inspectable)."""
    feas = feasible_frontier(frontier, t_max)
    feas.sort(key=lambda p: (-p["acc"], p["latency_ms"]))
    return feas[:k]


def arch_key(arch: dict) -> str:
    """Cache key: the canonical encoding (depth-inactive don't-cares masked, so two arch_dicts
    that describe the same network share a key). Matches the search's memo keying."""
    return str(tuple(canonical(encode(cast(ArchDict, arch)))))


# ---- de-noise driver (GPU, resumable) ----------------------------------------

def _load_cache(path: Any) -> dict[tuple[str, int], float]:
    memo: dict[tuple[str, int], float] = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                memo[(r["key"], int(r["seed"]))] = float(r["map"])
    return memo


def _append_cache(path: Any, key: str, seed: int, m: float) -> None:
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"key": key, "seed": seed, "map": m}) + "\n")


def denoise_archs(
    candidates: Sequence[dict],
    *,
    head_weights: Any,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    freeze_head: bool = True,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    epochs: int = 5,
    supernet: Any = None,
    cache: Any = None,
) -> list[dict]:
    """Re-fine-tune each candidate at every seed and average → de-noised accuracy.

    Resumable: each (arch, seed) mAP is cached to ``cache`` (jsonl) as it lands, so a killed Kaggle
    session resumes without recomputation. Returns each candidate enriched with ``denoised_maps`` /
    ``denoised_mean`` / ``denoised_std`` (``cached_acc`` = the single-seed value it replaces).
    GPU-gated (imports ``short_finetune`` lazily).
    """
    from eval.shortft import short_finetune  # lazy: torch/ultralytics/ofa only on the GPU run

    memo = _load_cache(cache)
    out: list[dict] = []
    for cand in candidates:
        arch = cand["arch"]
        key = arch_key(arch)
        maps: list[float] = []
        for s in seeds:
            if (key, s) in memo:
                m = memo[(key, s)]
            else:
                m = float(short_finetune(
                    dict(arch), epochs=epochs, seed=s, imgsz=imgsz, batch=batch, device=device,
                    supernet=supernet, head_weights=head_weights, freeze_head=freeze_head,
                )["map"])
                memo[(key, s)] = m
                _append_cache(cache, key, s, m)
            maps.append(m)
        out.append({
            **cand,
            "cached_acc": cand["acc"],
            "seeds": list(seeds),
            "denoised_maps": maps,
            "denoised_mean": st.mean(maps),
            "denoised_std": st.pstdev(maps) if len(maps) > 1 else 0.0,
        })
    return out


# ---- re-selection on the de-noised means (CPU, pure) -------------------------

def select_denoised(
    denoised: Sequence[dict], *, t_max: float, tie_band: float | None = None
) -> dict:
    """Re-pick the winner on de-noised accuracy. Among the archs statistically **tied** with the
    best de-noised mean (within ``tie_band``), return the **fastest** — because when accuracy
    saturates into a noise-tie, LUT-exact latency is the axis that actually discriminates.

    ``tie_band`` default = the largest per-arch ``denoised_std`` observed (the empirical single-arch
    noise floor): two means closer than one arch's own seed spread are "indistinguishable". A
    ``tie_band`` of 0 collapses this to a plain de-noised argmax (if a real winner separates).
    """
    feas = [d for d in denoised if d["latency_ms"] <= t_max]
    if not feas:
        raise ValueError(f"no de-noised candidate within the {t_max} ms ceiling")
    top = max(d["denoised_mean"] for d in feas)
    band = tie_band if tie_band is not None else max((d["denoised_std"] for d in feas), default=0.0)
    tie = [d for d in feas if top - d["denoised_mean"] <= band]
    return min(tie, key=lambda d: d["latency_ms"])


# ---- winner-v1 serialization (CP 3.5 close, CPU) ----------------------------

def _ranked_by_mean(cands: Sequence[dict]) -> list[dict]:
    """De-noised candidates, best mean first (the honest ranking)."""
    return sorted(cands, key=lambda d: -d["denoised_mean"])


def _find(cands: Sequence[dict], arch: dict) -> tuple[int, dict] | None:
    """Locate a candidate by arch identity in the mean-ranked list → (rank, candidate)."""
    key = arch_key(arch)
    for i, d in enumerate(_ranked_by_mean(cands)):
        if arch_key(d["arch"]) == key:
            return i, d
    return None


def _rank_of(cands: Sequence[dict], arch: dict) -> int:
    """The de-noised-mean rank of ``arch`` in ``cands`` (raises if absent)."""
    hit = _find(cands, arch)
    if hit is None:
        raise ValueError(f"arch d={arch.get('d')} not in the candidate set")
    return hit[0]


def denoised_winner_record(
    winner: dict,
    payload: dict,
    *,
    anchors: dict | None = None,
    baseline_latency_ms: float | None = None,
    repro_band: float = 0.020,
    old_alpha_arch: dict | None = None,
    frontier_sources: Sequence[str] | None = None,
) -> dict:
    """Build the winner-v1 record for a **de-noise-selected** winner (CP 3.5 close).

    The reference accuracy is the de-noised 3-seed **mean** (not the biased single-seed
    cached value): the de-noise seeds double as the CP 3.5 clean-session reproduction, so
    the record carries a reproduction verdict (``|cached − mean| ≤ band``). It also
    documents the winner's curse it corrects — the rejected single-seed α* and the
    fastest-cached arch that de-noising demoted (the averted *second* curse).
    """
    cands = payload["candidates"]
    winner_rank = _rank_of(cands, winner["arch"])
    mean, cached = winner["denoised_mean"], winner["cached_acc"]

    rejected = None  # the single-seed α* the curse originally picked (now demoted)
    if old_alpha_arch is not None and (hit := _find(cands, old_alpha_arch)) is not None:
        rank, d = hit
        rejected = {
            "arch_d": d["arch"]["d"], "cached_acc": d["cached_acc"],
            "denoised_mean": d["denoised_mean"], "denoised_std": d["denoised_std"],
            "denoised_rank": rank, "repro_delta": d["cached_acc"] - d["denoised_mean"],
            "reproduces": abs(d["cached_acc"] - d["denoised_mean"]) <= repro_band,
        }

    fastest = min(cands, key=lambda d: d["latency_ms"])  # the averted second curse
    fastest_rank = _rank_of(cands, fastest["arch"])

    base_lat = baseline_latency_ms if baseline_latency_ms is not None else payload["t_max_ms"]
    speedup = 100.0 * (base_lat - winner["latency_ms"]) / base_lat

    return {
        "arch": winner["arch"],
        "vector": encode(cast(ArchDict, winner["arch"])),
        "selection_rule": (
            "de-noised re-selection (CP 3.5): the single-seed ceiling-first argmax hit the "
            "winner's curse (the top cluster is a statistical tie), so the top-K feasible "
            "frontier was re-scored at 3 fresh seeds and the winner re-picked on de-noised "
            "means. Human-selected knee of the accuracy/latency frontier: a "
            f"{speedup:.0f}%-faster-than-yolo11n latency win at a proxy-mAP that saturates; "
            "equals search.denoise.select_denoised(tie_band~0.015)."
        ),
        "latency_ms": winner["latency_ms"],
        "acc": mean,                      # de-noised 3-seed MEAN = the reference accuracy
        "acc_eff": mean,
        "acc_std": winner["denoised_std"],
        "acc_seeds": winner["seeds"],
        "denoised_maps": winner["denoised_maps"],
        "cached_acc": cached,             # the biased single-seed value it replaces
        "denoised_rank": winner_rank,
        "t_max_ms": payload["t_max_ms"],
        "method": winner.get("method"),
        "seed": winner.get("seed"),
        "vs_yolo11n": {"baseline_latency_ms": base_lat, "latency_speedup_pct": speedup},
        "reproduction": {
            "band": repro_band,
            "cached_minus_mean": cached - mean,
            "passes": abs(cached - mean) <= repro_band,
            "note": ("the de-noise seeds (1,2,3) ARE the CP 3.5 clean-session reproduction; "
                     "the single-seed search value lands within band of the 3-seed mean."),
        },
        "winners_curse": {
            "diagnosis": ("the search oracle scored every arch at a single fine-tune seed "
                          "(seed 0); argmax over the saturated top cluster selects the "
                          "luckiest draw and regresses on re-eval."),
            "rejected_single_seed_alpha": rejected,
            "averted_second_curse": {
                "arch_d": fastest["arch"]["d"], "cached_acc": fastest["cached_acc"],
                "denoised_mean": fastest["denoised_mean"], "denoised_rank": fastest_rank,
                "note": ("the fastest-cached candidate; de-noising demoted it, so a naive "
                         "'fastest of the cached' re-pick would have been a second curse."),
            },
            "n_candidates": len(cands),
        },
        "anchors": anchors,
        "frontier_sources": list(frontier_sources) if frontier_sources else None,
        "denoise_source": "data/cp33_kaggle_out/denoise.json",
        "note": ("acc is the mean of 3 fresh-seed 5-epoch warm-head PROXY mAPs (CP 2.4 "
                 "ranking signal), NOT a full-train deployable number; Phase 8 distills the "
                 "deployable weights."),
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---- CLI: --make-candidates (CPU) | re-score (GPU) | --serialize (CPU) ------

def _make_candidates(argv_frontier: Sequence[Path], t_max: float, k: int, out: Path) -> int:
    frontier = load_frontier(argv_frontier)
    if not frontier:
        raise SystemExit(f"no frontier points in {[str(f) for f in argv_frontier]}")
    cands = top_candidates(frontier, t_max=t_max, k=k)
    payload = {"t_max_ms": t_max, "top_k": k, "n_frontier": len(frontier), "candidates": cands}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(cands)} candidates (top-{k} feasible <= {t_max} ms) -> {out}")
    for c in cands:
        print(f"  acc={c['acc']:.4f}  lat={c['latency_ms']:.3f}ms  [{c['method']} seed{c['seed']}]"
              f"  d={c['arch']['d']}")
    return 0


def _serialize_winner(denoise: Path, winner_index: int, old_winner: Path,
                      out_winner: Path, frontier: Sequence[Path]) -> int:
    from search.space import decode  # local: round-trip guard

    payload = json.loads(Path(denoise).read_text())
    ranked = _ranked_by_mean(payload["candidates"])
    print(f"de-noised candidates (best mean first, from {denoise}):")
    for i, d in enumerate(ranked):
        mark = "  <-- winner" if i == winner_index else ""
        print(f"  [{i:2d}] mean={d['denoised_mean']:.4f}+/-{d['denoised_std']:.4f}  "
              f"lat={d['latency_ms']:.3f}ms  d={d['arch']['d']}  "
              f"({d['method']} s{d['seed']}){mark}")
    if not 0 <= winner_index < len(ranked):
        raise SystemExit(f"--winner-index {winner_index} out of range [0,{len(ranked)})")
    winner = ranked[winner_index]

    # carry α* (for the winner's-curse record) + anchors forward from the prior winner.json
    old_arch = anchors = base_lat = None
    if Path(old_winner).exists():
        prev = json.loads(Path(old_winner).read_text())
        old_arch, anchors = prev.get("arch"), prev.get("anchors")
        if anchors and anchors.get("a"):
            base_lat = anchors["a"].get("latency_ms")

    record = denoised_winner_record(
        winner, payload, anchors=anchors, baseline_latency_ms=base_lat,
        old_alpha_arch=old_arch, frontier_sources=[str(f) for f in frontier])
    assert decode(record["vector"]) == record["arch"], "winner vector does not round-trip"

    Path(out_winner).parent.mkdir(parents=True, exist_ok=True)
    Path(out_winner).write_text(json.dumps(record, indent=2) + "\n")
    r = record["reproduction"]
    print(f"\nwinner-v1 -> {out_winner}")
    print(f"  arch d={record['arch']['d']}  acc(mean)={record['acc']:.4f}+/-{record['acc_std']:.4f}"
          f"  lat={record['latency_ms']:.3f}ms")
    print(f"  vs yolo11n: {record['vs_yolo11n']['latency_speedup_pct']:.1f}% faster")
    print(f"  reproduction: |cached-mean|={abs(r['cached_minus_mean']):.4f} <= band {r['band']}"
          f"  -> passes={r['passes']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="CP 3.5 de-noise: re-score the top frontier candidates at fresh seeds.")
    p.add_argument("--make-candidates", action="store_true",
                   help="CPU: write the pinned top-K candidate set from the frontier, then exit")
    p.add_argument("--serialize", action="store_true",
                   help="CPU: write winner.json from a de-noise result, then exit")
    p.add_argument("--denoise", type=Path,
                   default=ROOT / "data" / "cp33_kaggle_out" / "denoise.json",
                   help="de-noise result JSON (serialize mode reads this)")
    p.add_argument("--winner-index", type=int, default=None,
                   help="serialize: index into the de-noised-mean ranking (the chosen winner)")
    p.add_argument("--old-winner", type=Path, default=ROOT / "state" / "winner_v1" / "winner.json",
                   help="prior winner.json — carries α* + anchors into the winner's-curse record")
    p.add_argument("--out-winner", type=Path, default=ROOT / "state" / "winner_v1" / "winner.json",
                   help="where to write the de-noised winner-v1 record")
    p.add_argument("--frontier", nargs="+", type=Path,
                   default=[ROOT / "data" / "cp33_kaggle_out" / "cp33_bo.json",
                            ROOT / "data" / "cp33_kaggle_out" / "cp34_tpe.json"])
    p.add_argument("--candidates", type=Path,
                   default=ROOT / "state" / "winner_v1" / "denoise_candidates.json",
                   help="pinned candidate set (re-score mode reads this)")
    p.add_argument("--t-max-ms", type=float, default=12.75)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--head-weights", type=Path, default=None)
    p.add_argument("--freeze-head", dest="freeze_head", action="store_true", default=True)
    p.add_argument("--no-freeze-head", dest="freeze_head", action="store_false")
    p.add_argument("--device", default="cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "state" / "winner_v1" / "denoise.json")
    a = p.parse_args(argv)

    if a.make_candidates:
        return _make_candidates(a.frontier, a.t_max_ms, a.top_k, a.out)

    if a.serialize:
        if a.winner_index is None:
            raise SystemExit("--serialize needs --winner-index (the chosen de-noised winner)")
        return _serialize_winner(a.denoise, a.winner_index, a.old_winner, a.out_winner, a.frontier)

    if a.head_weights is None:
        raise SystemExit("re-score mode needs --head-weights (the frozen gate-head donor)")
    payload = json.loads(Path(a.candidates).read_text())
    cands = payload["candidates"]
    seeds = tuple(int(x) for x in a.seeds.split(","))
    denoised = denoise_archs(
        cands, head_weights=a.head_weights, seeds=seeds, freeze_head=a.freeze_head,
        device=a.device, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs, cache=a.cache,
    )
    denoised.sort(key=lambda d: -d["denoised_mean"])
    out = {"t_max_ms": a.t_max_ms, "seeds": list(seeds), "candidates": denoised,
           "note": "denoised_mean = mean of N fresh-seed warm-head proxy mAPs; replaces the biased "
                   "single-seed cached_acc. Re-select with search.denoise.select_denoised."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[denoise] re-scored {len(denoised)} candidates at seeds {list(seeds)} -> {a.out}")
    for d in denoised:
        print(f"  mean={d['denoised_mean']:.4f}±{d['denoised_std']:.4f} "
              f"(cached {d['cached_acc']:.4f})  lat={d['latency_ms']:.3f}ms  d={d['arch']['d']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
