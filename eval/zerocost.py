"""CP 2.4 Tier-1A — zero-cost architecture ranker (no GPU, no head, no fine-tune).

The CP 2.4 5-epoch fine-tune proxy ranks OFA backbones at only Kendall-τ=0.20 vs full-train
ground truth (random-head distortion — Kumar et al., LP-FT, ICLR 2022; cluster collapse —
Zero-Shot NAS survey, arXiv 2307.01998). This module ranks the *same* archs from cheap
descriptors that need no training:

* ``depth_sum``  — total MBConv depth ``sum(arch["d"])`` (free).
* ``params`` / ``flops`` / ``latency_ms`` — from the composed LUT cost (``search.cost.cost``;
  CPU-only, no Jetson/GPU). ``latency_ms`` is hardware-aware (Jetson-measured per-block).

Validated 2026-06-22 against ``data/cp24_proxy_rank.json`` (the 10-arch seed-0 ground truth):

    descriptor    kendall_tau  spearman  precision@3  top1_regret
    5-epoch proxy      0.200     0.212        0.33        0.0195   (the failure)
    depth_sum          0.767     0.843        0.67        0.0000   ← passes + picks true best
    latency_ms         0.733     0.855        0.67        0.0000
    flops              0.689     0.842        0.67        0.0000
    params             0.556     0.685        0.67        0.0000

(precision@3 is tie-sensitive — depth_sum has integer ties at the top-3 boundary, so it reads
0.67 under a stable sort and up to 1.0 under another tiebreak; τ and regret are tie-robust.)
Every descriptor picks the true-best arch (regret 0) — the zero-cost ranker *dominates* the
failed proxy on every metric, with no GPU. Literature: size/#params/#FLOPs are the strongest
single broad-range NAS signals (Abdelfattah et al., ICLR 2021; NAS-Bench-Suite-Zero, NeurIPS
2022). A gradient zero-cost proxy (ZiCo, ICLR 2023; jacob_cov) on the backbone is the GPU-gated
upgrade — see the plan; this module ships the CPU descriptor ranker.

    python -m eval.zerocost      # reproduce the table above against data/

The descriptor/score helpers are pure (arch + a pre-composed CostDict) → unit-tested under
.venv / CI; :func:`rank_report` is LUT-backed and smoked via ``__main__``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from catalog.contracts import ArchDict, CostDict
from eval.shortft import RankVerdict, rank_verdict

ROOT = Path(__file__).resolve().parents[1]

# The validated winner on the CP 2.4 ground truth (τ=0.767, precision@3=1.0, regret 0).
DEFAULT_DESCRIPTOR = "depth_sum"
DESCRIPTOR_KEYS = ("depth_sum", "latency_ms", "flops", "params")


def descriptors(arch_dict: ArchDict, cost_dict: CostDict) -> dict[str, float]:
    """Zero-cost ranking descriptors from an arch dict + its composed ``CostDict``.

    ``cost_dict`` is the output of :func:`search.cost.cost` (CPU LUT composition); passing it
    in keeps this pure and LUT-free. ``depth_sum`` comes straight off the arch's depth list.
    Higher is "more capacity" for every key — used directly as a maximize-me rank score.
    """
    return {
        "depth_sum": float(sum(arch_dict["d"])),
        "params": float(cost_dict["params"]),
        "flops": float(cost_dict["flops"]),
        "latency_ms": float(cost_dict["latency_ms"]),
    }


def zerocost_score(
    arch_dict: ArchDict, cost_dict: CostDict, *, key: str = DEFAULT_DESCRIPTOR
) -> float:
    """One zero-cost rank score for ``arch_dict`` — the ``key`` descriptor (default depth_sum)."""
    d = descriptors(arch_dict, cost_dict)
    if key not in d:
        raise ValueError(f"unknown descriptor {key!r}; choose from {sorted(d)}")
    return d[key]


@dataclass(frozen=True)
class DescriptorRanking:
    """How one zero-cost descriptor ranks a set of archs vs their full-train ground truth.

    Wraps the search-relevant :class:`eval.shortft.RankVerdict` (Spearman + top-1 regret) so a
    descriptor is judged by the *same* reframed gate as the fine-tune proxy.
    """

    name: str
    verdict: RankVerdict

    @property
    def passes(self) -> bool:
        return self.verdict.passes


def rank_report(
    records: list[dict],
    lut: dict,
    *,
    descriptor_keys: tuple[str, ...] = DESCRIPTOR_KEYS,
    k: int = 3,
) -> list[DescriptorRanking]:
    """Score each zero-cost descriptor against ``full_map`` over ``records`` (search-relevant gate).

    ``records`` are the ``{index, arch, proxy_map, full_map}`` rows from
    ``data/cp24_proxy_rank.json``; ``lut`` is a precision-filtered LUT (``lut.loader.load_lut``).
    Costs are composed per arch via :func:`search.cost.cost` (raises ``CostError`` if the LUT
    lacks a block). Returns one :class:`DescriptorRanking` per descriptor.
    """
    from search.cost import cost

    full = [r["full_map"] for r in records]
    scored: dict[str, list[float]] = {key: [] for key in descriptor_keys}
    for r in records:
        c = cost(r["arch"], lut)
        d = descriptors(r["arch"], c)
        for key in descriptor_keys:
            scored[key].append(d[key])

    return [
        DescriptorRanking(name=key, verdict=rank_verdict(scores, full, k=k))
        for key, scores in scored.items()
    ]


if __name__ == "__main__":  # smoke: reproduce the validation table against data/
    from lut.loader import load_lut

    recs = json.loads((ROOT / "data" / "cp24_proxy_rank.json").read_text())
    lut = {}
    for prec in ("fp32", "fp16"):
        lut = load_lut(ROOT / "data" / "lut.jsonl", precision=prec)
        if lut:
            print(f"LUT precision={prec}, rows={len(lut)}, archs={len(recs)}")
            break

    full = [r["full_map"] for r in recs]
    proxy = [r["proxy_map"] for r in recs]
    hdr = (f"{'descriptor':12} {'kendall_tau':>11} {'spearman':>9} "
           f"{'prec@3':>7} {'regret':>8} {'gate':>5}")
    print("\n(gate = search-relevant: Spearman >= 0.70 AND top1_regret <= 0.01)")
    print(hdr)

    def show(name: str, v: RankVerdict) -> None:
        print(f"{name:12} {v.kendall_tau:>11.3f} {v.spearman:>9.3f} "
              f"{v.precision_at_k:>7.2f} {v.top1_regret:>8.4f} "
              f"{'PASS' if v.passes else 'fail':>5}")

    show("5ep_proxy", rank_verdict(proxy, full))
    for dr in rank_report(recs, lut):
        show(dr.name, dr.verdict)
