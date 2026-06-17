"""Wire CP 2.2's deferred additivity DoD against the finished LUT.

The composite cost (``search.cost.cost``) *sums* per-block LUT latencies; CP 2.2's
Definition of Done requires checking that sum against the *measured* whole-subnet
latency, **binned by depth** (``search.validate_additivity``), because TensorRT
fuses across block seams a per-block LUT can't see and that error grows with depth
(peer-review R4.2). Pass = no depth bin exceeds 15 %; a breach pre-registers CP 2.3.

The LUT gives the *summed* side for free; the *measured* side needs whole subnets
benchmarked on the Jetson. This module splits the DoD across that boundary:

* ``manifest`` — now, offline: pick subnets whose depth spans the whole range
  (11 -> 21 LUT blocks), compute each ``summed_ms`` from the LUT, and write
  ``data/additivity_subnets.json`` with ``measured_ms: null`` placeholders. This
  pins *exactly which subnets* the Jetson must measure, so summed and measured
  refer to the same archs.
* ``report`` — later, once the ``measured_ms`` fields are filled from on-device
  runs: build the depth-binned :class:`~search.validate_additivity.AdditivityReport`
  and state PASS / BREACH. One command closes (or escalates) the DoD.

Runs under ``.venv`` (CPU); no ``ofa``/torch needed.

    .venv/bin/python -m search.additivity_preview manifest
    .venv/bin/python -m search.additivity_preview report --manifest data/additivity_subnets.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from catalog.contracts import ArchDict, LutRow
from catalog.ofa_mbv3 import KS, MAX_DEPTH, D, E
from lut.loader import load_lut
from search.arch_to_blocks import arch_to_blocks, random_arch_dict
from search.cost import DEFAULT_CALIBRATION_PATH, cost
from search.validate_additivity import AdditivityPoint, AdditivityReport, validate_additivity

if TYPE_CHECKING:
    from search.predictor_stats import CalibrationFit, PredictorStats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUT = ROOT / "data" / "lut.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "additivity_subnets.json"
DEFAULT_PAIRS_CSV = ROOT / "data" / "additivity_pairs.csv"

# Block count = 1 fixed first block + sum(per-stage depth); per-stage depth ranges
# over ``D`` across 5 stages => 11..21 blocks. Frames "spans the depth range".
MIN_BLOCKS = 1 + 5 * min(D)
MAX_BLOCKS = 1 + 5 * max(D)


def _depth_corner_arch(stage_depth: int, rng: random.Random) -> ArchDict:
    """An arch with every stage at ``stage_depth`` and seeded ks/e (a depth extreme)."""
    n = 5 * MAX_DEPTH
    return {
        "ks": [rng.choice(KS) for _ in range(n)],
        "e": [rng.choice(E) for _ in range(n)],
        "d": [stage_depth] * 5,
    }


def _arch_id(arch: ArchDict) -> str:
    """Stable short id for an arch — a unique remote job dir and manifest key.

    Purely arch-derived (sha1 of the canonical-JSON arch), so the same subnet maps
    to the same id across regenerations, and writeback can match measured values to
    entries even when several entries share a depth.
    """
    blob = json.dumps(arch, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def select_subnets(
    lut: dict[str, LutRow], *, seed: int = 0, k: int = 1, max_tries: int = 500
) -> list[dict]:
    """Up to ``k`` distinct subnets per distinct depth, spanning ``MIN_BLOCKS``..``MAX_BLOCKS``.

    Bucketing by depth keeps the report's depth bins cleanly separated — the point of
    the binned DoD — and ``k > 1`` gives each bin a within-bin sample so one outlier
    arch can't false-trigger or false-pass it. The two depth extremes are seeded
    explicitly (uniform draws almost never hit depth ``MIN_BLOCKS``/``MAX_BLOCKS``);
    intermediate depths fill from random draws. Each entry's ``summed_ms`` is composed
    from the LUT now; ``measured_ms`` waits for the Jetson.
    """
    rng = random.Random(seed)
    by_depth: dict[int, list[ArchDict]] = {}

    def offer(arch: ArchDict) -> None:
        bucket = by_depth.setdefault(len(arch_to_blocks(arch)), [])
        if len(bucket) < k and arch not in bucket:  # dedup distinct archs per depth
            bucket.append(arch)

    for stage_depth in (min(D), max(D)):  # guarantee k subnets at each depth extreme
        depth = 1 + 5 * stage_depth
        for _ in range(max_tries):
            if len(by_depth.get(depth, [])) >= k:
                break
            offer(_depth_corner_arch(stage_depth, rng))
    for _ in range(max_tries * k):  # fill the intermediate depths
        offer(random_arch_dict(rng))

    short = {d: len(v) for d, v in by_depth.items() if len(v) < k}
    if short:
        print(f"WARNING: fewer than k={k} subnets for depth(s) {short} "
              "(raise max_tries or lower k).")

    return [
        {
            "id": _arch_id(arch),
            "arch_dict": arch,
            "depth": depth,
            "summed_ms": cost(arch, lut)["latency_ms"],
            "measured_ms": None,
        }
        for depth in sorted(by_depth)
        for arch in by_depth[depth]
    ]


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def write_calibration(path: Path, stats: PredictorStats, *, precision: str,
                      source: str = "additivity_subnets") -> None:
    """Persist the affine latency fit (+ provenance) that ``search.cost`` reads back.

    ``search.cost.load_latency_calibration`` consumes only ``slope``/``intercept``; the
    rest (R^2, the fusion-discount factor, rank correlations, the precision the fit was
    derived at) is provenance so the JSON is self-documenting for the thesis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slope": stats.fit.slope,
        "intercept": stats.fit.intercept,
        "r2": stats.fit.r2,
        "mult_factor": stats.fit.mult_factor,
        "n": stats.n,
        "spearman_rho": stats.spearman_rho,
        "kendall_tau": stats.kendall_tau,
        "pearson_r": stats.pearson_r,
        "mape": stats.mape,
        "mape_calibrated": stats.mape_calibrated,
        "rmse_ms": stats.rmse_ms,
        "precision": precision,
        "source": source,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_pairs_csv(path: Path, entries: list[dict], fit: CalibrationFit) -> None:
    """Write per-subnet (summed, measured) pairs + relative error & calibrated prediction.

    The raw material for the summed-vs-measured scatter / residual plots in the thesis
    (matplotlib isn't a dependency here; plot the CSV elsewhere).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["id,depth,summed_ms,measured_ms,rel_error,calibrated_ms"]
    for e in entries:
        summed, measured = e["summed_ms"], e["measured_ms"]
        rel = (summed - measured) / measured
        calibrated = fit.slope * summed + fit.intercept
        rows.append(f"{e['id']},{e['depth']},{summed:.6g},{measured:.6g},"
                    f"{rel:.6g},{calibrated:.6g}")
    path.write_text("\n".join(rows) + "\n")


def _require_measured(entries: list[dict]) -> None:
    """Raise if any manifest entry still lacks ``measured_ms`` (on-device half undone)."""
    missing = [e["depth"] for e in entries if e.get("measured_ms") is None]
    if missing:
        raise ValueError(
            f"measured_ms is null for depth bin(s) {missing}; benchmark those whole "
            "subnets on the Jetson and fill measured_ms before reporting."
        )


def report_from_manifest(path: Path, *, bar: float = 0.15) -> AdditivityReport:
    """Build the depth-binned additivity report from a manifest with measured values.

    Raises ``ValueError`` if any ``measured_ms`` is still null — the on-device half
    of the DoD has not been run for those subnets yet.
    """
    entries = load_manifest(path)
    _require_measured(entries)
    points = [
        AdditivityPoint(depth=e["depth"], measured_ms=e["measured_ms"],
                        summed_ms=e["summed_ms"])
        for e in entries
    ]
    return validate_additivity(points, bar=bar)


def predictor_stats_from_manifest(path: Path) -> PredictorStats:
    """Rank correlation + affine calibration of summed-vs-measured latency.

    Lazy-imports ``search.predictor_stats`` (scipy) so the ``manifest`` path and the
    module import stay scipy-free; only ``report`` pulls scipy in.
    """
    from search.predictor_stats import predictor_stats
    entries = load_manifest(path)
    _require_measured(entries)
    return predictor_stats([e["summed_ms"] for e in entries],
                           [e["measured_ms"] for e in entries])


def print_report(report: AdditivityReport) -> None:
    print("\nAdditivity DoD — summed (LUT) vs measured (whole-net) latency by depth")
    print("=" * 72)
    print(f"  bar = {report.bar:.0%} mean signed error per depth bin "
          "(positive = LUT over-predicts, the fusion signature)\n")
    for depth in sorted(report.by_depth):
        err = report.by_depth[depth]
        flag = "   <-- BREACH" if depth in report.breaching_depths else ""
        print(f"    depth {depth:2d}: {err:+7.1%}{flag}")
    print(f"\n  aggregate: {report.aggregate:+.1%}")
    if report.breached:
        print(f"  RESULT: BREACH at depth bin(s) {report.breaching_depths} "
              "-> trigger CP 2.3 (residual correction).")
    else:
        print("  RESULT: PASS — no depth bin exceeds the bar; CP 2.2 DoD satisfied.")


def print_predictor_stats(stats: PredictorStats, *, precision: str) -> None:
    """Print the rank-correlation + calibration summary beneath the depth-binned table."""
    print(f"\nPredictor fidelity — summed-LUT vs measured whole-net latency "
          f"(n={stats.n}, precision={precision})")
    print("=" * 72)
    print("  Ranking (what search relies on — are archs ordered like the device?):")
    print(f"    Spearman rho = {stats.spearman_rho:+.4f}  (p={stats.spearman_p:.1e})")
    print(f"    Kendall  tau = {stats.kendall_tau:+.4f}  (p={stats.kendall_p:.1e})")
    print(f"    Pearson    r = {stats.pearson_r:+.4f}  (p={stats.pearson_p:.1e})")
    f = stats.fit
    print("\n  Calibration (measured ~= slope*summed + intercept):")
    print(f"    slope = {f.slope:.4f} +/- {f.stderr:.4f}   "
          f"intercept = {f.intercept:+.4f} ms   R^2 = {f.r2:.4f}")
    print(f"    fusion discount (through-origin) = {f.mult_factor:.4f}  "
          f"=> device runs ~{(1 - f.mult_factor) * 100:.1f}% faster than the summed LUT")
    print("\n  Absolute error (raw -> calibrated):")
    print(f"    MAPE = {stats.mape:.2%} -> {stats.mape_calibrated:.2%}    "
          f"RMSE = {stats.rmse_ms:.4g} -> {stats.rmse_calibrated_ms:.4g} ms")
    print(f"    bias = {stats.bias:+.2%}  (signed; + = LUT over-predicts)")


def _cmd_manifest(args: argparse.Namespace) -> int:
    lut = load_lut(args.lut, precision=args.precision)
    print(f"Loaded {len(lut)} LUT rows from {args.lut} (precision={args.precision})")
    entries = select_subnets(lut, seed=args.seed, k=args.k)
    write_manifest(args.out, entries)
    depths = [e["depth"] for e in entries]
    print(f"Selected {len(entries)} subnets spanning depth {min(depths)}..{max(depths)} "
          f"blocks (k={args.k} per depth).")
    for e in entries:
        print(f"    depth {e['depth']:2d}: id={e['id']} summed_ms={e['summed_ms']:.4g}  "
              "measured_ms=null")
    print(f"\nManifest -> {args.out}")
    print("Next: benchmark each subnet whole on the Jetson, fill 'measured_ms', then "
          "run `report`.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = report_from_manifest(args.manifest, bar=args.bar)
    print_report(report)
    stats = predictor_stats_from_manifest(args.manifest)
    print_predictor_stats(stats, precision=args.precision)
    if args.write_calibration:
        write_calibration(args.calibration_out, stats, precision=args.precision)
        print(f"\nCalibration -> {args.calibration_out}  "
              f"(slope={stats.fit.slope:.4f}, intercept={stats.fit.intercept:+.4f} ms)")
    if args.csv:
        write_pairs_csv(args.csv_out, load_manifest(args.manifest), stats.fit)
        print(f"Pairs CSV   -> {args.csv_out}  ({stats.n} rows)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="select depth-spanning subnets + summed_ms")
    m.add_argument("--lut", type=Path, default=DEFAULT_LUT)
    m.add_argument("--precision", default="fp32")
    m.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--k", type=int, default=1,
                   help="subnets per depth (3 gives each depth bin a within-bin sample)")

    r = sub.add_parser("report",
                       help="depth-binned report + predictor stats from a filled manifest")
    r.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    r.add_argument("--bar", type=float, default=0.15)
    r.add_argument("--precision", default="fp32",
                   help="precision label recorded in the calibration JSON / stats header")
    r.add_argument("--write-calibration", action="store_true",
                   help="write the affine fit to --calibration-out (search.cost reads it)")
    r.add_argument("--calibration-out", type=Path, default=DEFAULT_CALIBRATION_PATH)
    r.add_argument("--csv", action="store_true",
                   help="write the (summed, measured) pairs to --csv-out for plotting")
    r.add_argument("--csv-out", type=Path, default=DEFAULT_PAIRS_CSV)

    args = p.parse_args(argv)
    if args.cmd == "manifest":
        return _cmd_manifest(args)
    return _cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
