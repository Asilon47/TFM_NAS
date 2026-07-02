"""CP 3.5 — winner-v1 selection: two-anchor iso-J λ over the BO∪TPE frontier.

The final step of Phase 3: from the search frontiers (``cp33_bo.json`` +
``cp34_tpe.json``) pick the single architecture α* to deploy, by the D4 scalar
``J(α) = acc_eff − λ·latency`` argmaxed under the hard latency ceiling. λ is *not* a
magic constant — it is calibrated from **two reference models on one iso-J contour**
(:func:`search.objective.lambda_from_anchors`): anchor A = the deployed yolo11n-pose
``(0.877 mAP, 12.755 ms)`` and anchor B = a bigger yolo11-pose measured @640 on the
same Orin Nano. A **λ sensitivity sweep** (never one value) accompanies the pick.

This module is deliberately pure: every point in a frontier JSON already carries its
``latency_ms`` and ``acc_eff`` (``search.bo``/``search.tpe`` wrote them), so selection
is an argmax over stored numbers — no LUT recompute, no GPU. It reuses exactly the
``search.objective`` functions the search itself optimised (``scalarize`` /
``within_ceiling`` / ``lambda_from_anchors``), so selection and search share one J(α).

**Scope caveat.** The frontier ``acc`` values are the **5-epoch warm-head proxy** mAPs
(the CP 2.4 ranking signal), not full-train deployable numbers — so α* is the search's
best *candidate*; the CP 3.5 DoD (reload in a clean session, reproduce the cached
accuracy within noise) and Phase 8 (distillation) turn it into the deployable model.
Because the frontier is proxy-scale while the anchors are full-train, the λ sweep is
the honest hedge against that scale gap.

Run once anchor B lands (or explore now with a hypothetical λ over the BO frontier)::

    python -m search.select_winner --dry-run --lambda 0.01 \
        --frontier data/cp33_kaggle_out/cp33_bo.json
    python -m search.select_winner \
        --frontier data/cp33_kaggle_out/cp33_bo.json data/cp33_kaggle_out/cp34_tpe.json \
        --anchor-latency data/anchor_yolo11m_pose_640.json \
        --anchor-map data/anchor_yolo11m_pose_640_map.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path

from search.objective import Anchor, fps_to_ms, lambda_from_anchors, scalarize, within_ceiling
from search.space import encode

ROOT = Path(__file__).resolve().parents[1]

# The two search-output frontiers carry per-seed non-dominated points under these keys
# (search.bo writes ``bo_frontier``; search.tpe writes ``tpe_frontier``).
_FRONTIER_KEYS = (("bo_frontier", "bo"), ("tpe_frontier", "tpe"))


# ---- anchors -----------------------------------------------------------------

def read_anchor(latency_path: Path, map_path: Path) -> Anchor:
    """Join a bench_model latency JSON + a pose_map accuracy JSON into one Anchor.

    The two halves live in separate files by construction — latency is Jetson-measured
    (``bench_model.py`` → ``{name}.json``), accuracy is GPU/CPU-measured
    (``detect.evaluate.pose_map`` → ``{name}_map.json``) — so the anchor is assembled
    here rather than read from one blob.
    """
    lat = json.loads(Path(latency_path).read_text())["latency_ms"]["mean"]
    acc = json.loads(Path(map_path).read_text())["map"]
    return Anchor(acc=float(acc), latency_ms=float(lat))


# ---- frontier loading --------------------------------------------------------

def load_frontier(paths: Sequence[Path]) -> list[dict]:
    """Union of the per-seed non-dominated frontiers from CP 3.3/3.4 output JSONs.

    Concatenates every ``runs[].{bo_frontier|tpe_frontier}`` point across all payloads,
    tagging each with the ``method`` (bo/tpe) and ``seed`` it came from for provenance.
    Points keep the search record shape ``{arch, acc, latency_ms, acc_eff}``. TPE is
    optional (a BO-only union is valid before CP 3.4 lands); duplicates across
    seeds/methods are harmless to the argmax and keep the provenance honest.
    """
    points: list[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for run in payload.get("runs", []):
            for key, method in _FRONTIER_KEYS:
                for pt in run.get(key, []):
                    points.append({**pt, "method": method, "seed": run.get("seed")})
    return points


# ---- selection: argmax J under the hard ceiling ------------------------------

def _J(point: dict, lam: float) -> float:
    """``J = acc_eff − λ·latency`` for one frontier point (μ folded into acc_eff already,
    so μ=0 here avoids double-counting the memory term)."""
    return scalarize(point["acc_eff"], point["latency_ms"], 0.0, lam=lam, mu=0.0)


def feasible_frontier(frontier: Sequence[dict], t_max: float) -> list[dict]:
    """The frontier points within the hard latency ceiling (the search already ceilinged;
    this re-guards against a mixed-source or hand-edited union)."""
    return [pt for pt in frontier if within_ceiling(pt["latency_ms"], t_max)]


def select_winner(frontier: Sequence[dict], *, lam: float, t_max: float) -> dict:
    """α* = the feasible frontier point maximising ``J = acc_eff − λ·latency``.

    Ties break toward higher accuracy (a deterministic, sensible preference when two
    archs sit on the same iso-J line). Raises if nothing clears the ceiling.
    """
    feasible = feasible_frontier(frontier, t_max)
    if not feasible:
        raise ValueError(f"no frontier point within the {t_max} ms latency ceiling")
    return max(feasible, key=lambda pt: (_J(pt, lam), pt["acc_eff"]))


# ---- λ sensitivity sweep -----------------------------------------------------

def lambda_grid(lam: float, *, n: int = 7, span: float = 2.0) -> list[float]:
    """A log-spaced λ grid ``[λ/span, λ·span]`` (n points) geometrically centred on λ.

    Log spacing because λ is an exchange *rate* (multiplicative moves matter, not
    additive); an odd ``n`` puts the committed λ exactly mid-sweep.
    """
    if lam <= 0.0:
        raise ValueError(f"lambda must be positive for a log sweep, got {lam}")
    if n < 2:
        raise ValueError(f"need at least 2 sweep points, got {n}")
    lo, hi = math.log(lam / span), math.log(lam * span)
    return [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def lambda_sensitivity(
    frontier: Sequence[dict], *, t_max: float, lambdas: Sequence[float]
) -> list[dict]:
    """The α* selected at each λ — the report that shows how stable the winner is to the
    exchange rate (``objective.py`` mandates a sweep, never one magic value)."""
    sweep: list[dict] = []
    for lam in lambdas:
        w = select_winner(frontier, lam=lam, t_max=t_max)
        sweep.append({"lambda": lam, "arch": w["arch"], "acc": w["acc"],
                      "latency_ms": w["latency_ms"], "J": _J(w, lam)})
    return sweep


# ---- serialisation: the winner-v1 export -------------------------------------

def winner_record(
    winner: dict,
    *,
    lam: float,
    anchor_a: Anchor,
    anchor_b: Anchor | None,
    t_max: float,
    sources: Sequence[str],
    sensitivity: Sequence[dict],
) -> dict:
    """The self-describing winner-v1 record: α* + its ``encode`` vector, the λ and both
    anchors it was picked under, J, the frontier provenance, the sweep, and the caveat."""
    return {
        "arch": winner["arch"],
        "vector": encode(winner["arch"]),
        "latency_ms": winner["latency_ms"],
        "acc": winner["acc"],
        "acc_eff": winner["acc_eff"],
        "J": _J(winner, lam),
        "lambda": lam,
        "method": winner.get("method"),
        "seed": winner.get("seed"),
        "t_max_ms": t_max,
        "anchors": {
            "a": {"acc": anchor_a.acc, "latency_ms": anchor_a.latency_ms},
            "b": (None if anchor_b is None
                  else {"acc": anchor_b.acc, "latency_ms": anchor_b.latency_ms}),
        },
        "frontier_sources": [str(s) for s in sources],
        "lambda_sensitivity": list(sensitivity),
        "note": ("acc is the 5-epoch warm-head PROXY mAP (CP 2.4 ranking signal), NOT a "
                 "full-train deployable number; the CP 3.5 DoD reproduces it in a clean "
                 "session and Phase 8 distills the deployable weights."),
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def serialize_winner(record: dict, out_dir: Path) -> Path:
    """Write ``winner.json`` under ``out_dir`` (``state/winner_v1/``); returns its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "winner.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    return out


def _t_max_from_frontier(paths: Sequence[Path], baseline: Anchor | None) -> float:
    """Default T_max = the ceiling the search itself used (first payload's ``t_max_ms``);
    fall back to ``min(baseline_latency, 60 FPS → 16.7 ms)`` if unrecorded."""
    for p in paths:
        tm = json.loads(Path(p).read_text()).get("t_max_ms")
        if tm is not None:
            return float(tm)
    if baseline is not None:
        return min(baseline.latency_ms, fps_to_ms(60))
    return fps_to_ms(60)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="CP 3.5 — winner-v1 selection (two-anchor iso-J λ)")
    p.add_argument("--frontier", nargs="+", type=Path,
                   default=[ROOT / "data" / "cp33_kaggle_out" / "cp33_bo.json"],
                   help="CP 3.3/3.4 output JSON(s); their frontiers are unioned")
    p.add_argument("--baseline-latency", type=Path, default=ROOT / "data" / "baseline_anchor.json",
                   help="anchor A latency (yolo11n-pose @640, bench_model.py)")
    p.add_argument("--baseline-map", type=Path, default=ROOT / "data" / "baseline_anchor_map.json",
                   help="anchor A accuracy (yolo11n-pose gate mAP, pose_map)")
    p.add_argument("--anchor-latency", type=Path, default=None,
                   help="anchor B latency (bigger yolo11-pose @640, bench_model.py)")
    p.add_argument("--anchor-map", type=Path, default=None,
                   help="anchor B accuracy (bigger yolo11-pose gate mAP, pose_map)")
    p.add_argument("--lambda", dest="lam", type=float, default=None,
                   help="override λ directly (skip anchor B; explore a hypothetical rate)")
    p.add_argument("--t-max-ms", type=float, default=None,
                   help="hard ceiling (default: the frontier's own t_max_ms)")
    p.add_argument("--sweep-n", type=int, default=7, help="λ sensitivity-sweep points")
    p.add_argument("--sweep-span", type=float, default=2.0, help="sweep half-width factor")
    p.add_argument("--out-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--dry-run", action="store_true", help="print α* but do not serialize")
    a = p.parse_args(argv)

    frontier = load_frontier(a.frontier)
    if not frontier:
        raise SystemExit(f"no frontier points found in {[str(f) for f in a.frontier]}")

    anchor_a = read_anchor(a.baseline_latency, a.baseline_map)
    anchor_b: Anchor | None = None
    if a.lam is not None:
        lam = a.lam
    else:
        if not (a.anchor_latency and a.anchor_map):
            raise SystemExit(
                "λ is undefined: pass --lambda, OR both --anchor-latency and --anchor-map "
                "(anchor B's @640 latency + gate mAP) to calibrate it from the two anchors.")
        anchor_b = read_anchor(a.anchor_latency, a.anchor_map)
        lam = lambda_from_anchors(anchor_a, anchor_b)
        if lam <= 0.0:
            raise SystemExit(
                f"λ={lam:.4g} ≤ 0: anchor B does not Pareto-trade against A (one dominates "
                "the other) — pick a bigger/more-accurate anchor B.")

    t_max = a.t_max_ms if a.t_max_ms is not None else _t_max_from_frontier(a.frontier, anchor_a)
    winner = select_winner(frontier, lam=lam, t_max=t_max)
    sweep = lambda_sensitivity(frontier, t_max=t_max,
                               lambdas=lambda_grid(lam, n=a.sweep_n, span=a.sweep_span))
    record = winner_record(winner, lam=lam, anchor_a=anchor_a, anchor_b=anchor_b, t_max=t_max,
                           sources=[str(f) for f in a.frontier], sensitivity=sweep)

    n_feasible = len(feasible_frontier(frontier, t_max))
    print(f"lambda = {lam:.6g} acc/ms  (T_max = {t_max:.4g} ms; "
          f"{n_feasible}/{len(frontier)} frontier points feasible)")
    print(f"winner alpha* [{winner.get('method')}, seed {winner.get('seed')}]: "
          f"acc={winner['acc']:.4f}  latency={winner['latency_ms']:.3f} ms  "
          f"J={record['J']:.4f}  d={winner['arch']['d']}")
    print("lambda sensitivity (lambda -> winner latency):")
    for row in sweep:
        print(f"  {row['lambda']:.5g}\t-> {row['latency_ms']:.3f} ms (acc {row['acc']:.4f})")

    if a.dry_run:
        print("[dry-run] not serialized.")
        return 0
    out = serialize_winner(record, a.out_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
