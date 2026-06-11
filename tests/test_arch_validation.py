"""validate_arch_dict: the boundary where Phase-3 search output enters.

Beyond shape/membership errors, the type checks are a row_key guard:
``True`` or ``np.int64(3)`` in an arch dict would respectively change or
crash the JSON hash downstream (see tests/test_row_key.py's tripwire).
"""
import random

import pytest

from catalog.ofa_mbv3 import MAX_DEPTH
from search.arch_to_blocks import (_random_arch_dict, arch_to_blocks,
                                   validate_arch_dict)

N = 5 * MAX_DEPTH


def _good():
    return {"ks": [3] * N, "e": [4] * N, "d": [2, 3, 4, 2, 3]}


def test_random_archs_validate():
    rng = random.Random(0)
    for _ in range(25):
        validate_arch_dict(_random_arch_dict(rng))


def test_good_arch_passes():
    validate_arch_dict(_good())


def test_extra_keys_tolerated():
    arch = _good() | {"image_size": 224}
    validate_arch_dict(arch)


@pytest.mark.parametrize("field", ["ks", "e", "d"])
def test_missing_key_named(field):
    arch = _good()
    del arch[field]
    with pytest.raises(ValueError, match=f"missing key '{field}'"):
        validate_arch_dict(arch)


def test_wrong_length_named():
    arch = _good()
    arch["ks"] = arch["ks"][:-1]
    with pytest.raises(ValueError, match=f"'ks' must be a list of length {N}"):
        validate_arch_dict(arch)


def test_out_of_set_value_named():
    arch = _good()
    arch["e"][3] = 5  # not in E = [3, 4, 6]
    with pytest.raises(ValueError, match="'e' has values outside"):
        validate_arch_dict(arch)


def test_bool_rejected_even_though_true_equals_one():
    arch = _good()
    arch["d"][0] = True  # bool is an int subclass; must still be rejected
    with pytest.raises(ValueError, match="'d' has values outside"):
        validate_arch_dict(arch)


def test_numpy_ints_rejected():
    np = pytest.importorskip("numpy")
    arch = _good()
    arch["ks"][0] = np.int64(3)
    with pytest.raises(ValueError, match="'ks' has values outside"):
        validate_arch_dict(arch)


def test_arch_to_blocks_validates_at_entry():
    with pytest.raises(ValueError, match="invalid arch_dict"):
        arch_to_blocks({"ks": [3], "e": [3], "d": [2]})


def test_multiple_problems_reported_together():
    with pytest.raises(ValueError) as exc:
        validate_arch_dict({"ks": [3] * N})
    msg = str(exc.value)
    assert "missing key 'e'" in msg
    assert "missing key 'd'" in msg
