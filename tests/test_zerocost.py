"""Zero-cost ranker descriptors (CP 2.4 Tier-1A).

The pure descriptor/score logic takes an arch dict + a pre-composed ``CostDict`` (from
``search.cost.cost``), so it's testable with a fake cost — no LUT / torch / ofa. The
LUT-backed end-to-end ranking is smoked via the module ``__main__``.
"""

import pytest

from eval.zerocost import DEFAULT_DESCRIPTOR, descriptors, zerocost_score

FAKE_COST = {
    "latency_ms": 3.2, "peak_mem_mib": 10.0, "params": 5_000_000, "flops": 1_200_000_000,
}
ARCH = {"d": [2, 3, 4, 2, 3], "e": [3] * 20, "ks": [3] * 20}


def test_descriptors_extracts_depth_sum_and_cost_fields():
    d = descriptors(ARCH, FAKE_COST)
    assert d["depth_sum"] == 14.0  # 2+3+4+2+3
    assert d["params"] == 5_000_000.0
    assert d["flops"] == 1_200_000_000.0
    assert d["latency_ms"] == 3.2


def test_zerocost_score_defaults_to_depth_sum():
    assert DEFAULT_DESCRIPTOR == "depth_sum"
    assert zerocost_score(ARCH, FAKE_COST) == 14.0


def test_zerocost_score_selects_named_descriptor():
    assert zerocost_score(ARCH, FAKE_COST, key="latency_ms") == 3.2


def test_zerocost_score_rejects_unknown_key():
    with pytest.raises(ValueError):
        zerocost_score(ARCH, FAKE_COST, key="bogus")
