"""Freeze the LUT row_key contract (catalog/sweep.py).

The append-only LUT (data/lut.jsonl) is keyed by
``sha1(json.dumps({"b": block, "c": cfg, "s": list(shape)}, sort_keys=True))[:16]``.
The golden values below pin today's hashes: if any of them ever changes,
every measured Jetson row is orphaned.

NEVER update a golden hash here without an explicit decision recorded in
procedure.md — a failing golden means the change re-keys the LUT.
"""
import pytest

from catalog.blocks import input_shape_for
from catalog.ofa_mbv3 import FIRST_BLOCK
from catalog.sweep import row_key

GOLDEN = [
    ("mbconv", dict(FIRST_BLOCK), "97b3989502a54710"),
    ("mbconv", {"in_c": 16, "out_c": 24, "kernel": 3, "stride": 2,
                "expand": 3, "se": False, "res": 112}, "52e3f3125087c80a"),
    ("mbconv", {"in_c": 16, "out_c": 16, "kernel": 3, "stride": 1,
                "expand": 3, "se": False, "res": 56}, "2c80f6675e07f839"),
    ("conv3x3", {"in_c": 16, "out_c": 16, "stride": 1, "res": 112},
     "6716c9188f3c29a4"),
    ("dethead", {"in_c": 128, "num_classes": 80, "num_anchors": 3, "res": 40},
     "c88eee8a84672882"),
]


@pytest.mark.parametrize("block,cfg,expected", GOLDEN,
                         ids=[f"{b}-{k[:6]}" for b, _, k in GOLDEN])
def test_golden_hashes(block, cfg, expected):
    shape = input_shape_for(block, cfg)
    assert row_key(block, cfg, shape) == expected


def test_cfg_insertion_order_is_irrelevant():
    """json.dumps(sort_keys=True) makes the hash independent of dict order."""
    cfg = {"in_c": 16, "out_c": 24, "kernel": 3, "stride": 2,
           "expand": 3, "se": False, "res": 112}
    reordered = dict(reversed(list(cfg.items())))
    shape = (1, 16, 112, 112)
    assert row_key("mbconv", cfg, shape) == row_key("mbconv", reordered, shape)


def test_tuple_and_list_shapes_are_equivalent():
    """row_key coerces input_shape via list(); tuple/list callers must agree."""
    cfg = {"in_c": 16, "out_c": 16, "stride": 1, "res": 112}
    assert (row_key("conv3x3", cfg, (1, 16, 112, 112))
            == row_key("conv3x3", cfg, [1, 16, 112, 112]))


def test_bool_vs_int_changes_the_hash():
    """TRIPWIRE: cfg value *types* are load-bearing.

    JSON serializes ``True`` as ``true`` but ``1`` as ``1``, so coercing
    ``se`` (or any bool) to an int silently re-keys the row. Code that
    constructs cfgs destined for row_key must preserve exact Python types
    (see catalog/contracts.py).
    """
    base = {"in_c": 16, "out_c": 24, "kernel": 3, "stride": 2,
            "expand": 3, "se": False, "res": 112}
    as_int = dict(base, se=0)
    shape = (1, 16, 112, 112)
    assert row_key("mbconv", base, shape) == "52e3f3125087c80a"
    assert row_key("mbconv", as_int, shape) == "f160aea663964354"


def test_key_format_is_16_hex_chars():
    key = row_key("skip", {"in_c": 16, "res": 28}, (1, 16, 28, 28))
    assert len(key) == 16
    assert set(key) <= set("0123456789abcdef")
