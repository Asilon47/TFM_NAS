"""Catalog grid invariants.

The grids in catalog/blocks.py ARE the cfg schema: every cfg in a block's
grid carries exactly the keys its builder and input_shape_fn read (the sweep
would crash otherwise), so schema uniformity within a grid is what keeps
build/export/key-derivation in lockstep.

Counts are pinned exactly on purpose: widening a grid is fine (the LUT is
append-only) but must be a conscious act — update the pin in the same commit
and regenerate/extend the LUT.
"""
import pytest
import torch

from catalog.blocks import BLOCK_REGISTRY, build_block, input_shape_for
from catalog.ofa_mbv3 import reachable_mbconv_configs
from catalog.sweep import iter_sweep, sweep_size


# Pins bumped 2026-06-28 (CP 3.3 prep): the @640 pose deploy grid was unioned in
# (+91 MBConv rows, res {320,160,80,40,20}, disjoint from @224 → append-only). The
# golden row_key hashes (test_row_key.py) are unchanged; only these counts move.
# Decision recorded in procedure.md "CP 3.3 — @640 LUT re-key".
def test_sweep_size_pinned():
    assert sweep_size() == 2801   # was 2710 @224-only; +91 @640


def test_mbconv_grid_size_pinned():
    assert len(BLOCK_REGISTRY["mbconv"]["grid"]) == 2198   # was 2107; +91 @640


def test_reachable_ofa_configs_pinned():
    assert len(reachable_mbconv_configs()) == 91          # per single resolution
    assert len(reachable_mbconv_configs(640)) == 91       # @640 mirrors @224


@pytest.mark.parametrize("name", sorted(BLOCK_REGISTRY))
def test_grid_cfgs_share_one_schema(name):
    grid = BLOCK_REGISTRY[name]["grid"]
    assert grid, f"{name}: empty grid"
    keys = set(grid[0])
    for cfg in grid:
        assert set(cfg) == keys, f"{name}: cfg {cfg} deviates from schema {keys}"


def test_ofa_reachable_configs_are_in_mbconv_grid():
    """CP 2.1 invariant: every block the OFA space can reach has a LUT row."""
    grid = BLOCK_REGISTRY["mbconv"]["grid"]
    missing = [cfg for cfg in reachable_mbconv_configs() if cfg not in grid]
    assert not missing, f"{len(missing)} OFA-reachable cfgs absent from grid"


def test_iter_sweep_keys_globally_unique():
    keys = [k for *_, k in iter_sweep()]
    assert len(keys) == len(set(keys)) == 2801   # @224 (2710) + @640 (91)


@pytest.mark.parametrize("name", sorted(BLOCK_REGISTRY))
def test_first_and_last_cfg_build_and_forward(name):
    """Smoke: the grid's corner cfgs construct a module that forwards."""
    grid = BLOCK_REGISTRY[name]["grid"]
    for cfg in (grid[0], grid[-1]):
        shape = input_shape_for(name, cfg)
        module = build_block(name, cfg).eval()
        with torch.no_grad():
            out = module(torch.zeros(*shape))
        assert out.shape[0] == shape[0]
