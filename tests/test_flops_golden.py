"""Hand-computed FLOPs goldens for the hook-based counter.

Every expected value below derives from the Conv2d multiply-add formula
``2 * out_c * (in_c / groups) * kh * kw * oh * ow`` worked out by hand — NOT
from running the code — so a silent change to the counter's arithmetic fails
here. Targets gen_dummy_lut.measure_block, which delegates to the shared
counter in catalog/flops.py.
"""
from catalog.blocks import build_block
from catalog.flops import count_flops
from lut.orchestrate.gen_dummy_lut import measure_block


def test_conv3x3_flops():
    m = measure_block("conv3x3", {"in_c": 16, "out_c": 16, "stride": 1, "res": 112},
                      (1, 16, 112, 112))
    assert m["flops"] == 2 * 16 * 16 * 3 * 3 * 112 * 112 == 57_802_752


def test_dwconv_flops_respect_groups():
    m = measure_block("dwconv", {"in_c": 16, "kernel": 3, "stride": 1, "res": 56},
                      (1, 16, 56, 56))
    assert m["flops"] == 2 * 16 * 1 * 3 * 3 * 56 * 56 == 903_168


def test_se_flops_on_pooled_spatial():
    # SEBlock(32): pool to 1x1, then 1x1 convs 32 -> 8 -> 32 at 1x1 spatial.
    m = measure_block("se", {"in_c": 32, "res": 28}, (1, 32, 28, 28))
    assert m["flops"] == 2 * 8 * 32 + 2 * 32 * 8 == 1_024


def test_io_bytes_are_fp16_sized():
    # skip is identity: io = input numel + output numel, 2 bytes each in FP16.
    m = measure_block("skip", {"in_c": 16, "res": 28}, (1, 16, 28, 28))
    assert m["io_bytes"] == 2 * (1 * 16 * 28 * 28) * 2


def test_measure_block_delegates_to_shared_counter():
    """run_sweep and gen_dummy_lut must count FLOPs identically by construction."""
    cfg = {"in_c": 16, "out_c": 16, "stride": 1, "res": 112}
    shape = (1, 16, 112, 112)
    direct = count_flops(build_block("conv3x3", cfg).eval(), shape)
    assert measure_block("conv3x3", cfg, shape)["flops"] == direct
