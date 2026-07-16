"""Stage 0 — additively stamp the measured end-to-end truth into ``winner.json``.

The pivot's owed correction (procedure.md "Plan pivot"): ``vs_yolo11n.latency_speedup_pct``
compares a backbone-blocks-only LUT sum against a full-network baseline measurement. This tool
adds an ``"e2e"`` block with the honest, same-session numbers — winner measured end-to-end vs
the baseline **re-checked in the same session** — plus the fallbacks' measurements so a
re-pick (if the margin collapsed) is data-ready.

**Additive only.** Every pre-existing key of ``winner.json`` is left byte-identical
(``eval/verify_winner.py`` and the CP 3.5 tests read them); the merge asserts it. Re-stamping
requires ``--force`` (and still only replaces the ``e2e`` block).

Same-session guard: winner/baseline/backbone rows must agree on precision + power_mode and be
clocks-locked — a speedup across clock regimes is the exact trap this tool exists to kill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean(row: dict) -> float:
    return float(row["latency_ms"]["mean"])


def _check_regime(rows: dict[str, dict]) -> dict:
    """All rows share precision/power_mode and are clocks-locked; returns the common stamps."""
    ref_name = next(iter(rows))
    ref = rows[ref_name]
    for name, row in rows.items():
        for key in ("precision", "power_mode"):
            if row.get(key) != ref.get(key):
                raise ValueError(f"{name} row {key}={row.get(key)!r} != {ref_name}'s "
                                 f"{ref.get(key)!r} — one session, one regime, or no stamp")
        if not row.get("clocks_locked"):
            raise ValueError(f"{name} row is not clocks-locked — no stamp")
    return {"precision": ref.get("precision"), "power_mode": ref.get("power_mode"),
            "trt_version": ref.get("trt_version"), "clocks_locked": True}


def e2e_block(
    winner: dict,
    e2e_row: dict,
    baseline_row: dict,
    *,
    backbone_row: dict | None = None,
    fallback_rows: dict[str, dict] | None = None,
) -> dict:
    """Build the honest e2e block (pure). ``fallback_rows`` maps label → bench row."""
    rows = {"winner_e2e": e2e_row, "baseline_recheck": baseline_row}
    if backbone_row is not None:
        rows["winner_backbone"] = backbone_row
    for label, row in (fallback_rows or {}).items():
        rows[f"fallback:{label}"] = row
    regime = _check_regime(rows)

    winner_ms, base_ms = _mean(e2e_row), _mean(baseline_row)
    speedup = 100.0 * (base_ms - winner_ms) / base_ms
    block: dict = {
        "latency_ms_measured": winner_ms,
        "baseline_recheck_ms": base_ms,
        "speedup_pct_e2e": speedup,
        "winner_beats_baseline_e2e": winner_ms < base_ms,
        "lut_sum_ms": float(winner["latency_ms"]),
        "baseline_anchor_ms": float(winner["vs_yolo11n"]["baseline_latency_ms"]),
        "regime": regime,
        "note": ("measured end-to-end, one session, both sides full networks — supersedes the "
                 "headline vs_yolo11n.latency_speedup_pct, which compared the backbone-blocks-"
                 "only LUT sum against the full-network baseline (see procedure.md 'Plan "
                 "pivot'). Existing keys are untouched by design."),
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if backbone_row is not None:
        block["backbone_measured_ms"] = _mean(backbone_row)
        block["offset_ms_derived"] = winner_ms - _mean(backbone_row)
    if fallback_rows:
        block["fallbacks"] = {
            label: {"latency_ms_measured": _mean(row),
                    "speedup_pct_e2e": 100.0 * (base_ms - _mean(row)) / base_ms}
            for label, row in fallback_rows.items()
        }
    return block


def stamped(winner: dict, block: dict, *, force: bool = False) -> dict:
    """The winner record + the ``e2e`` block, additive-only (pure; asserts no key mutated)."""
    if "e2e" in winner and not force:
        raise ValueError("winner.json already has an 'e2e' block — pass --force to re-stamp")
    out = {**winner, "e2e": block}
    for key, value in winner.items():
        if key != "e2e" and out[key] != value:
            raise AssertionError(f"additive-only violated: key {key!r} changed")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--winner", type=Path, default=ROOT / "state" / "winner_v1" / "winner.json")
    ap.add_argument("--e2e", type=Path, required=True, help="winner e2e bench row JSON")
    ap.add_argument("--baseline", type=Path, required=True,
                    help="same-session baseline re-check row JSON (NOT the old anchor)")
    ap.add_argument("--backbone", type=Path, default=None,
                    help="winner backbone-only bench row JSON (adds the derived offset)")
    ap.add_argument("--fallback", action="append", default=[], metavar="LABEL=PATH",
                    help="fallback bench row, repeatable (e.g. idx11_d22432=data/e2e/f11.json)")
    ap.add_argument("--force", action="store_true", help="replace an existing e2e block")
    args = ap.parse_args(argv)

    winner = json.loads(args.winner.read_text())
    fallbacks: dict[str, dict] = {}
    for spec in args.fallback:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--fallback needs LABEL=PATH, got {spec!r}")
        fallbacks[label] = json.loads(Path(path).read_text())

    block = e2e_block(
        winner, json.loads(args.e2e.read_text()), json.loads(args.baseline.read_text()),
        backbone_row=json.loads(args.backbone.read_text()) if args.backbone else None,
        fallback_rows=fallbacks or None)
    record = stamped(winner, block, force=args.force)
    args.winner.write_text(json.dumps(record, indent=2) + "\n")

    print(f"e2e stamp -> {args.winner}")
    print(f"  winner e2e {block['latency_ms_measured']:.4g} ms  vs baseline re-check "
          f"{block['baseline_recheck_ms']:.4g} ms  => honest speedup "
          f"{block['speedup_pct_e2e']:+.1f}%  (LUT sum was {block['lut_sum_ms']:.4g} ms)")
    for label, f in block.get("fallbacks", {}).items():
        print(f"  fallback {label}: {f['latency_ms_measured']:.4g} ms "
              f"({f['speedup_pct_e2e']:+.1f}%)")
    if not block["winner_beats_baseline_e2e"]:
        print("  ⚠ winner does NOT beat the baseline end-to-end — ceiling-first re-pick from "
              "the benched fallbacks is the documented next step (user decision).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
