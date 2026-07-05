"""CP 3.5 sensitivity companion — which winner each principled tie-band picks.

The CP 3.5 close (procedure.md "CP 3.5 CLOSED") documented that the de-noised winner is a
*researcher-degrees-of-freedom* point: `search.denoise.select_denoised` breaks the saturated
top cluster with a tie-band, and the three principled band choices (the top arch's own σ /
the typical σ / the max σ) pick three different archs — accuracy-first, knee, latency-first.
The committed knee (``state/winner_v1/winner.json``) corresponds to ``tie_band ~ 0.015``.

This module makes that sensitivity inspectable on demand — the honest companion to the single
committed pick (and the homed replacement of the root-level ``evaluate_denoised.py`` scratch
script, which printed only the max-σ and argmax regimes)::

    python -m search.denoise_report                    # the four principled regimes
    python -m search.denoise_report --tie-band 0.015   # + the committed knee's band

Pure logic (:func:`band_regimes` / :func:`winners_by_band`) is ``.venv``/CI-tested; the CLI
reads ``data/cp33_kaggle_out/denoise.json``.
"""
from __future__ import annotations

import json
import statistics as st
from collections.abc import Mapping, Sequence
from pathlib import Path

from search.denoise import select_denoised

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENOISE = ROOT / "data" / "cp33_kaggle_out" / "denoise.json"


def band_regimes(cands: Sequence[dict]) -> dict[str, float]:
    """The principled tie-band choices from the CP 3.5 close, band-ascending.

    * ``argmax``  — 0: a plain de-noised argmax (no tie at all).
    * ``strict``  — the best-mean arch's *own* σ: only archs inside the top's noise.
    * ``typical`` — the median σ across candidates: the representative noise floor.
    * ``loose``   — the max σ (CP 3.5: α*'s outlier spread): the widest defensible tie.
    """
    if not cands:
        raise ValueError("no candidates")
    best = max(cands, key=lambda d: float(d["denoised_mean"]))
    stds = [float(d["denoised_std"]) for d in cands]
    return {
        "argmax": 0.0,
        "strict (top arch's own σ)": float(best["denoised_std"]),
        "typical (median σ)": float(st.median(stds)),
        "loose (max σ)": max(stds),
    }


def winners_by_band(
    cands: Sequence[dict], *, t_max: float, bands: Mapping[str, float] | None = None
) -> dict[str, dict]:
    """``select_denoised`` winner per tie-band regime (default: :func:`band_regimes`)."""
    regimes = dict(bands) if bands is not None else band_regimes(cands)
    return {name: select_denoised(cands, t_max=t_max, tie_band=b) for name, b in regimes.items()}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="CP 3.5 tie-band sensitivity: the de-noised winner per principled band.")
    p.add_argument("--denoise-json", type=Path, default=DEFAULT_DENOISE,
                   help="de-noise result (search.denoise re-score output)")
    p.add_argument("--t-max-ms", type=float, default=None,
                   help="latency ceiling; default = the file's t_max_ms")
    p.add_argument("--tie-band", type=float, default=None,
                   help="report one extra custom band (e.g. 0.015 = the committed knee's)")
    a = p.parse_args(argv)

    payload = json.loads(Path(a.denoise_json).read_text())
    cands = payload["candidates"]
    t_max = float(payload["t_max_ms"]) if a.t_max_ms is None else a.t_max_ms

    print(f"de-noised candidates (t_max={t_max} ms, {a.denoise_json}):")
    for c in sorted(cands, key=lambda d: -float(d["denoised_mean"])):
        print(f"  mean={c['denoised_mean']:.4f}±{c['denoised_std']:.4f}  "
              f"lat={c['latency_ms']:.3f}ms  d={c['arch']['d']}")

    regimes = band_regimes(cands)
    if a.tie_band is not None:
        regimes[f"custom ({a.tie_band:g})"] = a.tie_band
    print("\nwinner per tie-band (select_denoised: fastest within band of the best mean):")
    for name, w in winners_by_band(cands, t_max=t_max, bands=regimes).items():
        print(f"  {name:26s} band={regimes[name]:.4f} -> "
              f"mean={w['denoised_mean']:.4f}±{w['denoised_std']:.4f}  "
              f"lat={w['latency_ms']:.3f}ms  d={w['arch']['d']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
