"""Iterate the full sweep as (block_name, cfg, input_shape, row_key) tuples."""
import hashlib
import json

from .blocks import BLOCK_REGISTRY, input_shape_for


def row_key(block: str, cfg: dict, input_shape) -> str:
    """Stable ID for a LUT row — used for resumability and engine filenames."""
    payload = json.dumps({"b": block, "c": cfg, "s": list(input_shape)},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def iter_sweep(only_blocks=None):
    for name, spec in BLOCK_REGISTRY.items():
        if only_blocks and name not in only_blocks:
            continue
        for cfg in spec["grid"]:
            shape = input_shape_for(name, cfg)
            yield name, cfg, shape, row_key(name, cfg, shape)


def sweep_size(only_blocks=None) -> int:
    return sum(1 for _ in iter_sweep(only_blocks))
