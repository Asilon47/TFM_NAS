"""Winner-v2-OFA Track 4 — stamp the deliverable's own record + an additive v1 pointer.

Unlike ``stamp_winner_e2e`` (which adds an ``e2e`` block to the winner-v1 arch record), the
v2 deliverable is a *different* artifact — a pruned/KD graft with its own arch, compression
spec, de-noised accuracy, and a median-of-N fp16 latency (the TRT-variance de-risk). It gets
its own ``state/winner_v2_ofa/winner.json``. ``--link-v1`` then adds ONE pointer key to the
v1 record, asserting every pre-existing key stays byte-identical (the additive-only contract
``eval/verify_winner`` + the CP 3.5 tests depend on).

The pick rule is the user's latency-first mandate (2026-07-13): **max de-noised mAP subject
to median fp16 < baseline recheck AND fp32 < baseline recheck** — strict on BOTH axes, vs the
SAME-session baseline (never the stale anchor). The stamp refuses to certify a winner that
misses either bar, and requires ≥3 fp16 rebuild rows (median computed here, not supplied).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

from search.stamp_winner_e2e import _check_regime, _mean

ROOT = Path(__file__).resolve().parents[1]
MIN_FP16_BUILDS = 3      # the variance-de-risk protocol (fresh-timing-cache rebuilds)


def median_fp16_ms(build_rows: list[dict]) -> float:
    """Median of the fresh-cache fp16 rebuild rows (≥3; computed, never supplied)."""
    if len(build_rows) < MIN_FP16_BUILDS:
        raise ValueError(f"need >={MIN_FP16_BUILDS} fp16 rebuild rows for the median "
                         f"(got {len(build_rows)}) — the TRT ±20% de-risk protocol")
    for r in build_rows:
        if r.get("precision") != "fp16":
            raise ValueError(f"fp16 build row has precision={r.get('precision')!r}")
        if not r.get("fresh_timing_cache"):
            raise ValueError("fp16 rebuild rows must carry fresh_timing_cache=true — a shared "
                             "TRT timing cache makes rebuilds identical (no variance sampled)")
    return float(statistics.median(_mean(r) for r in build_rows))


def verdict(fp32_ms: float, fp16_median_ms: float, base_fp32_row: dict,
            base_fp16_row: dict) -> dict:
    """Strict both-axes vs the same-session baseline recheck (the user's latency mandate)."""
    b32, b16 = _mean(base_fp32_row), _mean(base_fp16_row)
    return {
        "baseline_fp32_ms": b32, "baseline_fp16_ms": b16,
        "fp32_ms": fp32_ms, "fp16_median_ms": fp16_median_ms,
        "beats_fp32": fp32_ms < b32, "beats_fp16": fp16_median_ms < b16,
        "beats_both_axes": (fp32_ms < b32) and (fp16_median_ms < b16),
        "margin_fp32_pct": 100.0 * (b32 - fp32_ms) / b32,
        "margin_fp16_pct": 100.0 * (b16 - fp16_median_ms) / b16,
    }


def winner_v2_record(
    *,
    arch: dict,
    compression: dict,
    accuracy: dict,
    fp32_row: dict,
    fp16_build_rows: list[dict],
    base_fp32_row: dict,
    base_fp16_row: dict,
    physical_fit: str = "search/graft_latency_fit.json",
    provenance: str = "",
) -> dict:
    """The full winner-v2-OFA record (pure). Regime-checked across fp32 + all fp16 builds."""
    # fp16 builds and fp16 baseline share their own regime; fp32 rows share theirs. Precision
    # differs across the two groups by design, so regime-check each precision group separately.
    _check_regime({"v2_fp32": fp32_row, "baseline_fp32": base_fp32_row})
    fp16_rows = {f"v2_fp16_{i}": r for i, r in enumerate(fp16_build_rows)}
    fp16_rows["baseline_fp16"] = base_fp16_row
    regime16 = _check_regime(fp16_rows)

    fp32_ms = _mean(fp32_row)
    fp16_med = median_fp16_ms(fp16_build_rows)
    v = verdict(fp32_ms, fp16_med, base_fp32_row, base_fp16_row)
    return {
        "family": "ofa_graft",
        "deliverable": "winner-v2-OFA",
        "arch": arch,
        "compression": compression,   # {technique, spec, kd, seed, params, sparsity, ...}
        "accuracy": accuracy,         # {map_seed0, denoised: {seeds, maps, mean, std}}
        "latency": {
            "fp32_ms": fp32_ms,
            "fp16_ms_builds": [_mean(r) for r in fp16_build_rows],
            "fp16_ms_median": fp16_med,
            "fp16_builds_n": len(fp16_build_rows),
            "regime_fp16": regime16,
            "fresh_timing_cache": True,
        },
        "verdict": v,
        "physical_fit": physical_fit,
        "provenance": provenance or ("winner-v2-OFA: OFA graft, hardware-honest compression "
                                     "(allocate_v2 act-fence + global_taylor + KD); latency = "
                                     "median of fresh-cache fp16 rebuilds; both-axes strict vs "
                                     "same-session baseline"),
        "rank_rule": "max de-noised mAP s.t. median fp16 < baseline AND fp32 < baseline",
        "certified": v["beats_both_axes"],
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def link_v1(winner_v1: dict, *, path: str, summary: str, force: bool = False) -> dict:
    """Add ONE additive pointer key to the v1 record; assert nothing else changed."""
    key = "winner_v2_ofa"
    if key in winner_v1 and not force:
        raise ValueError(f"winner_v1 already has {key!r} — pass --force to replace")
    out = {**winner_v1, key: {"path": path, "summary": summary,
                              "timestamp": dt.datetime.now(dt.timezone.utc).strftime(
                                  "%Y-%m-%dT%H:%M:%SZ")}}
    for k, val in winner_v1.items():
        if k != key and out[k] != val:
            raise AssertionError(f"additive-only violated: key {k!r} changed")
    return out


def _load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "state" / "winner_v2_ofa" / "winner.json")
    ap.add_argument("--meta", type=Path, required=True,
                    help="recover_graft <tag>.meta.json of the chosen champion (arch/spec/kd/"
                         "params/map)")
    ap.add_argument("--denoise", type=Path, default=None,
                    help="optional de-noise JSON {seeds, maps} for the accuracy block")
    ap.add_argument("--fp32", type=Path, required=True, help="champion fp32 bench row")
    ap.add_argument("--fp16", action="append", required=True, metavar="PATH",
                    help="champion fp16 rebuild row (repeat >=3 times, fresh_timing_cache)")
    ap.add_argument("--base-fp32", type=Path, required=True, help="same-session baseline fp32")
    ap.add_argument("--base-fp16", type=Path, required=True, help="same-session baseline fp16")
    ap.add_argument("--link-v1", type=Path, default=None,
                    help="winner_v1/winner.json to add the additive pointer key to")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    meta = _load(a.meta)
    arch = meta.get("arch") or meta.get("spec", {}).get("arch") or {}
    accuracy: dict = {"map_seed0": meta.get("map")}
    if a.denoise:
        d = _load(a.denoise)
        maps = d["maps"]
        accuracy["denoised"] = {"seeds": d.get("seeds"), "maps": maps,
                                "mean": float(statistics.mean(maps)),
                                "std": float(statistics.pstdev(maps)) if len(maps) > 1 else 0.0}
    compression = {k: meta.get(k) for k in ("technique", "spec", "kd", "seed", "params",
                                            "params_sparsity", "ratio", "arch_tag")}

    record = winner_v2_record(
        arch=arch, compression=compression, accuracy=accuracy,
        fp32_row=_load(a.fp32), fp16_build_rows=[_load(p) for p in a.fp16],
        base_fp32_row=_load(a.base_fp32), base_fp16_row=_load(a.base_fp16))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2) + "\n")

    v = record["verdict"]
    print(f"winner-v2-OFA -> {a.out}")
    print(f"  fp32 {v['fp32_ms']:.3f} ms ({v['margin_fp32_pct']:+.1f}% vs "
          f"{v['baseline_fp32_ms']:.3f}) {'PASS' if v['beats_fp32'] else 'FAIL'}")
    print(f"  fp16 median {v['fp16_median_ms']:.3f} ms ({v['margin_fp16_pct']:+.1f}% vs "
          f"{v['baseline_fp16_ms']:.3f}) {'PASS' if v['beats_fp16'] else 'FAIL'}")
    print(f"  CERTIFIED: {record['certified']}  (mAP seed0 {accuracy['map_seed0']}"
          + (f", de-noised {accuracy['denoised']['mean']:.4f}" if 'denoised' in accuracy else "")
          + ")")
    if not record["certified"]:
        print("  ⚠ does NOT beat the baseline on both axes — not a valid winner-v2 (pick a "
              "lighter rung or a bigger-margin spec).")

    if a.link_v1:
        summary = (f"pruned OFA graft, fp16 {v['fp16_median_ms']:.2f} ms "
                   f"({v['margin_fp16_pct']:+.0f}%), mAP "
                   f"{accuracy.get('denoised', {}).get('mean', accuracy['map_seed0'])}")
        linked = link_v1(_load(a.link_v1), path=str(a.out), summary=summary, force=a.force)
        a.link_v1.write_text(json.dumps(linked, indent=2) + "\n")
        print(f"  linked pointer into {a.link_v1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
