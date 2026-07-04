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


# ---- CLI: --make-candidates (CPU) | re-score (GPU) ---------------------------

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


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="CP 3.5 de-noise: re-score the top frontier candidates at fresh seeds.")
    p.add_argument("--make-candidates", action="store_true",
                   help="CPU: write the pinned top-K candidate set from the frontier, then exit")
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
