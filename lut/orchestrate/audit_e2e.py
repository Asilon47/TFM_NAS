"""Audit measured e2e rows for the contention signature — which numbers are safe to believe.

The 2026-07-08 cross-family bench saw three anomalous fp16 numbers and concluded "fp16 carries
±~20 % TRT-build variance (autotiler kernel selection) — indicative only", demoting the whole
fp16 column of models/README.md. fp16 is the deploy precision, so that caveat suppressed the one
axis every deployment claim rests on.

It was the wrong diagnosis, and the evidence was already in the rows. **Build variance and
contention leave different fingerprints:**

  * a different *build* is a stable-but-different number  -> mean shifts, std stays tight
  * *contention* (or throttling) is instability WITHIN one run -> std explodes, p95 detaches
    from p50, and mean drifts off p50 because a fat tail drags it

``prune_base_r45_640_fp16`` was recorded with std 1.0582 on mean 7.185 (14.7 %) and p95 3.38 ms
above p50, while every clean row in data/e2e sits at std ~0.2 % and p95-p50 ~0.015 ms. That is a
contention fingerprint wearing a build-variance label. The 2026-07-08 write-up even records the
session's own contention incident two paragraphs later.

This makes the check mechanical instead of a thing someone has to notice. Run it before trusting
any measured row, and after any bench session::

    python -m lut.orchestrate.audit_e2e                 # .venv; exits 1 if any row is suspect
    python -m lut.orchestrate.audit_e2e --dir data/e2e --rel-std 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Clean rows measure ~0.1-0.7 % relative std (67/70 of the 2026-07 corpus). 1 % is a generous
# floor: it flags nothing clean and catches all three known-bad rows.
REL_STD_PCT = 1.0
# A fat tail is the tell contention leaves and a build choice does not. Clean rows: ~0.015 ms.
TAIL_MS = 0.10


def audit_row(path: Path, *, rel_std_pct: float = REL_STD_PCT,
              tail_ms: float = TAIL_MS) -> dict | None:
    """Return a verdict dict for one e2e/anchor JSON, or None if it carries no distribution."""
    try:
        d = json.loads(path.read_text())
        lat = d["latency_ms"]
        mean, std, p50, p95 = (float(lat[k]) for k in ("mean", "std", "p50", "p95"))
    except Exception:
        return None
    rel = 100.0 * std / mean if mean else 0.0
    tail = p95 - p50
    reasons = []
    if rel > rel_std_pct:
        reasons.append(f"std {rel:.1f}% of mean (clean rows ~0.2%)")
    if tail > tail_ms:
        reasons.append(f"p95-p50 = {tail:.3f} ms (clean rows ~0.015)")
    return {
        "row": path.stem, "precision": d.get("precision"), "mean": mean, "std": std,
        "p50": p50, "p95": p95, "rel_std_pct": rel, "tail_ms": tail,
        "suspect": bool(reasons), "reasons": reasons,
        # p50 resists a fat tail, so it is the salvageable number when a row is suspect.
        "p50_vs_mean_pct": (100.0 * (mean - p50) / p50) if p50 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "e2e")
    ap.add_argument("--rel-std", type=float, default=REL_STD_PCT,
                    help=f"flag rows whose std exceeds this %% of mean (default {REL_STD_PCT})")
    ap.add_argument("--json", action="store_true", help="emit the verdicts as JSON")
    args = ap.parse_args(argv)

    rows = [r for r in (audit_row(p, rel_std_pct=args.rel_std)
                        for p in sorted(args.dir.glob("*.json"))) if r]
    if not rows:
        raise SystemExit(f"no rows with a latency distribution under {args.dir}")
    suspect = [r for r in rows if r["suspect"]]

    if args.json:
        print(json.dumps({"n": len(rows), "n_suspect": len(suspect), "rows": rows}, indent=2))
    else:
        print(f"{len(rows)} rows audited, {len(suspect)} suspect "
              f"(threshold: std > {args.rel_std}% of mean, or p95-p50 > {TAIL_MS} ms)\n")
        for r in sorted(suspect, key=lambda r: -r["rel_std_pct"]):
            print(f"  SUSPECT {r['row']} ({r['precision']})")
            print(f"          mean {r['mean']:.3f}  p50 {r['p50']:.3f}  std {r['std']:.4f}  "
                  f"p95 {r['p95']:.3f}")
            for why in r["reasons"]:
                print(f"          - {why}")
            print(f"          -> mean is {r['p50_vs_mean_pct']:+.1f}% off p50; re-measure on an "
                  f"idle board (verify `docker ps` is empty first) before trusting it")
        if not suspect:
            print("  all clean — no contention fingerprint")
    return 1 if suspect else 0


if __name__ == "__main__":
    raise SystemExit(main())
