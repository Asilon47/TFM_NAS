"""Resolution-aware catalog: the @640 deploy sweep is an append-only re-key.

D1's pose pivot runs the backbone at 640, not the OFA ImageNet 224. The per-block
input resolutions therefore re-key (stem 320; stage res_in [320,160,80,40,40] →
taps 80/40/20, confirmed by supernet/pose_backbone.py). These tests pin that the
new resolution helpers (1) reproduce the legacy @224 tables EXACTLY (so every
measured row + golden hash is untouched) and (2) produce a disjoint @640 grid.
"""
import random

from catalog.ofa_mbv3 import (
    FIRST_BLOCK,
    STAGES,
    first_block_for,
    reachable_mbconv_configs,
    stages_for_resolution,
    stem_res_for,
)
from catalog.sweep import iter_sweep
from search.arch_to_blocks import _random_arch_dict, arch_to_keys


def test_stem_res_halves_input():
    assert stem_res_for(224) == 112
    assert stem_res_for(640) == 320


def test_stages_for_224_reproduce_the_legacy_table():
    """The append-only guarantee: @224 derivation == the hardcoded constant."""
    assert stages_for_resolution(224) == STAGES


def test_first_block_for_224_reproduces_the_legacy_constant():
    assert first_block_for(224) == FIRST_BLOCK


def test_stages_for_640_scale_the_resolutions():
    stages = stages_for_resolution(640)
    assert [s["res_in"] for s in stages] == [320, 160, 80, 40, 40]
    # out_c / stride / se are resolution-invariant — only res_in moves.
    for legacy, scaled in zip(STAGES, stages, strict=True):
        assert (scaled["out_c"], scaled["stride"], scaled["se"]) == (
            legacy["out_c"], legacy["stride"], legacy["se"])


def test_first_block_for_640_sits_at_the_post_stem_resolution():
    assert first_block_for(640)["res"] == 320
    # everything but res is identical to the @224 first block
    assert {k: v for k, v in first_block_for(640).items() if k != "res"} == {
        k: v for k, v in FIRST_BLOCK.items() if k != "res"}


def test_reachable_640_has_the_same_count_as_224():
    """Same topology (channels, KS×E, entry/repeat dedup) → same config count."""
    assert len(reachable_mbconv_configs(640)) == len(reachable_mbconv_configs(224))


def test_reachable_640_is_disjoint_from_224():
    """@640 rows are genuinely new — they never collide with a measured @224 row."""
    configs_224 = reachable_mbconv_configs(224)
    configs_640 = reachable_mbconv_configs(640)
    assert all(c not in configs_224 for c in configs_640)
    assert {c["res"] for c in configs_640} == {320, 160, 80, 40, 20}


def test_640_arch_keys_are_all_in_the_catalog():
    """A pose subnet costed @640 finds every block row in the unioned catalog."""
    mbconv_keys = {k for *_, k in iter_sweep(["mbconv"])}
    rng = random.Random(0)
    for _ in range(20):
        arch = _random_arch_dict(rng)
        missing = [k for k in arch_to_keys(arch, res=640) if k not in mbconv_keys]
        assert not missing, f"{len(missing)} @640 keys absent from the catalog"
