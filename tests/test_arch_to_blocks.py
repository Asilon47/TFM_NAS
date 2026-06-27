"""CP 2.1 contract: arch_to_blocks output is LUT-covered and structurally sound.

The DoD smoke test (``python -m search.arch_to_blocks``) checks emitted keys
against the generated data/lut.jsonl; these tests check the stronger
in-memory invariant — emitted keys ⊆ catalog sweep keys — with no file
dependency, plus the structural properties (channel chaining, stride and
resolution propagation, depth truncation) the smoke test cannot see.
"""
import random

import pytest

from catalog.ofa_mbv3 import FIRST_BLOCK, MAX_DEPTH, STAGES, stage_in_c
from catalog.sweep import iter_sweep, row_key
from search.arch_to_blocks import _random_arch_dict, arch_to_blocks, arch_to_keys

N_ARCHS = 25


@pytest.fixture(scope="module")
def mbconv_keys():
    return {k for *_, k in iter_sweep(["mbconv"])}


def test_emitted_keys_are_lut_covered(mbconv_keys):
    rng = random.Random(0)
    for i in range(N_ARCHS):
        arch = _random_arch_dict(rng)
        missing = [k for k in arch_to_keys(arch) if k not in mbconv_keys]
        assert not missing, f"arch {i}: {len(missing)} keys missing from catalog"


def test_first_block_is_fixed():
    arch = _random_arch_dict(random.Random(1))
    name, cfg, shape = arch_to_blocks(arch)[0]
    assert name == "mbconv"
    assert cfg == FIRST_BLOCK
    assert row_key(name, cfg, shape) == "97b3989502a54710"


def test_block_count_is_one_plus_sum_d():
    rng = random.Random(2)
    for _ in range(N_ARCHS):
        arch = _random_arch_dict(rng)
        assert len(arch_to_blocks(arch)) == 1 + sum(arch["d"])


def test_channels_chain_across_all_blocks():
    rng = random.Random(3)
    for _ in range(N_ARCHS):
        blocks = arch_to_blocks(_random_arch_dict(rng))
        for prev, cur in zip(blocks, blocks[1:], strict=False):
            assert cur[1]["in_c"] == prev[1]["out_c"]


def test_strides_and_resolutions_follow_stage_table():
    arch = _random_arch_dict(random.Random(4))
    blocks = arch_to_blocks(arch)
    i = 1  # blocks[0] is the fixed first block
    for s, stage in enumerate(STAGES):
        for j in range(arch["d"][s]):
            _, cfg, shape = blocks[i]
            if j == 0:  # stage entry: prev_w -> out_w at the stage stride
                assert cfg["stride"] == stage["stride"]
                assert cfg["res"] == stage["res_in"]
                assert cfg["in_c"] == stage_in_c(s)
            else:  # repeats: out_w -> out_w, stride 1, post-stride res
                assert cfg["stride"] == 1
                assert cfg["res"] == stage["res_in"] // stage["stride"]
                assert cfg["in_c"] == stage["out_c"]
            assert cfg["out_c"] == stage["out_c"]
            assert shape == (1, cfg["in_c"], cfg["res"], cfg["res"])
            i += 1
    assert i == len(blocks)


def test_res_640_scales_every_block_resolution():
    """res=640 emits the pose deploy grid: same structure, scaled resolutions."""
    arch = _random_arch_dict(random.Random(8))
    blocks_224 = arch_to_blocks(arch, res=224)
    blocks_640 = arch_to_blocks(arch, res=640)
    assert len(blocks_640) == len(blocks_224)  # identical structure
    # every @640 cfg differs from its @224 twin only in res (channels/k/stride same)
    for (_, c224, _), (_, c640, _) in zip(blocks_224, blocks_640, strict=True):
        assert c640["res"] != c224["res"]
        assert {k: v for k, v in c640.items() if k != "res"} == {
            k: v for k, v in c224.items() if k != "res"}
    # the first block sits at the post-stem 320, stage entries start there too
    assert blocks_640[0][1]["res"] == 320


def test_depth_truncation_ignores_inactive_slots():
    """With d=2 everywhere, slots 2..3 of each stage must not affect output."""
    n = 5 * MAX_DEPTH
    base = {"ks": [3] * n, "e": [3] * n, "d": [2] * 5}
    poked = {"ks": list(base["ks"]), "e": list(base["e"]), "d": [2] * 5}
    for s in range(5):
        for j in range(2, MAX_DEPTH):
            poked["ks"][MAX_DEPTH * s + j] = 7
            poked["e"][MAX_DEPTH * s + j] = 6
    assert arch_to_blocks(base) == arch_to_blocks(poked)
