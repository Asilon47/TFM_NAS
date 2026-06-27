"""CP 3.1 contract: search/space.py encode/decode round-trip + surrogate surface.

DoD (PROJECT_PLAN.md:213): ``decode(encode(arch)) == arch`` for 100 random archs.

Beyond the DoD, these pin the two properties that keep the encoder honest:

- The round-trip is **lossless over all 45 slots**, including the depth-inactive
  don't-cares that OFA's sampler fills randomly (see
  ``test_arch_to_blocks.test_depth_truncation_ignores_inactive_slots`` for the
  proof those slots don't affect the network). A masking "optimization" in
  encode/decode would silently break the bijection for any ``d < 4`` arch.
- ``decode`` emits **plain ``int``** (``validate_arch_dict`` rejects ``np.int64``
  / ``bool``), and ``canonical()`` collapses functionally-identical archs to one
  vector — the input surface the CP 3.3 GP surrogate consumes.
"""
import random

from catalog.ofa_mbv3 import KS, MAX_DEPTH, STAGES, D, E
from search.arch_to_blocks import random_arch_dict, validate_arch_dict
from search.space import (
    AXIS_CARDINALITIES,
    AXIS_TYPES,
    INACTIVE,
    VECTOR_LEN,
    canonical,
    decode,
    encode,
)

N_ARCHS = 100
N_SLOTS = len(STAGES) * MAX_DEPTH


def test_decode_encode_roundtrip_is_identity():
    """The DoD: decode(encode(arch)) == arch for 100 random archs."""
    rng = random.Random(0)
    for i in range(N_ARCHS):
        arch = random_arch_dict(rng)
        assert decode(encode(arch)) == arch, f"arch {i} did not round-trip"


def test_encode_length_and_index_range():
    """Every encoded value is a category index in its axis's range."""
    rng = random.Random(1)
    for _ in range(N_ARCHS):
        vec = encode(random_arch_dict(rng))
        assert len(vec) == VECTOR_LEN
        for x, card in zip(vec, AXIS_CARDINALITIES, strict=True):
            assert 0 <= x < card


def test_decode_emits_plain_ints_passing_the_contract():
    """decode output must satisfy validate_arch_dict (plain int, not np.int64)."""
    rng = random.Random(2)
    for _ in range(N_ARCHS):
        arch = decode(encode(random_arch_dict(rng)))
        for field in ("ks", "e", "d"):
            assert all(type(x) is int for x in arch[field])
        validate_arch_dict(arch)  # raises ValueError on any violation


def test_canonical_collapses_depth_inactive_slots():
    """Two archs differing ONLY in inactive slots map to one canonical vector."""
    base = {"ks": [KS[0]] * N_SLOTS, "e": [E[0]] * N_SLOTS, "d": [D[0]] * len(STAGES)}
    poked = {"ks": list(base["ks"]), "e": list(base["e"]), "d": list(base["d"])}
    for s in range(len(STAGES)):
        for j in range(D[0], MAX_DEPTH):  # inactive: j >= d[s]
            poked["ks"][MAX_DEPTH * s + j] = KS[-1]
            poked["e"][MAX_DEPTH * s + j] = E[-1]
    assert poked != base  # the raw dicts differ...
    assert canonical(encode(base)) == canonical(encode(poked))  # ...but collapse


def test_canonical_distinguishes_active_slots():
    """A change in an ACTIVE slot must change the canonical vector."""
    base = {"ks": [KS[0]] * N_SLOTS, "e": [E[0]] * N_SLOTS, "d": [D[-1]] * len(STAGES)}
    poked = {"ks": list(base["ks"]), "e": list(base["e"]), "d": list(base["d"])}
    poked["ks"][0] = KS[-1]  # stage 0, block 0 — always active (d>=2)
    assert canonical(encode(base)) != canonical(encode(poked))


def test_canonical_marks_inactive_ks_and_e_with_sentinel():
    """Inactive ks AND e slots become INACTIVE; active slots keep a real index."""
    arch = {"ks": [KS[0]] * N_SLOTS, "e": [E[0]] * N_SLOTS, "d": [D[0]] * len(STAGES)}
    vec = canonical(encode(arch))
    for s in range(len(STAGES)):
        for j in range(MAX_DEPTH):
            ks_slot = MAX_DEPTH * s + j
            e_slot = N_SLOTS + MAX_DEPTH * s + j
            if j < D[0]:  # active
                assert vec[ks_slot] != INACTIVE
                assert vec[e_slot] != INACTIVE
            else:  # inactive
                assert vec[ks_slot] == INACTIVE
                assert vec[e_slot] == INACTIVE


def test_axis_metadata_shape_and_types():
    """Metadata is per-dim: ks/e categorical, d ordinal, all cardinality 3."""
    assert len(AXIS_TYPES) == VECTOR_LEN
    assert len(AXIS_CARDINALITIES) == VECTOR_LEN
    assert all(t == "categorical" for t in AXIS_TYPES[: 2 * N_SLOTS])
    assert all(t == "ordinal" for t in AXIS_TYPES[2 * N_SLOTS :])
    assert all(c == 3 for c in AXIS_CARDINALITIES)


def test_corner_archs_roundtrip():
    """The all-min and all-max corners round-trip exactly."""
    lo = {"ks": [KS[0]] * N_SLOTS, "e": [E[0]] * N_SLOTS, "d": [D[0]] * len(STAGES)}
    hi = {"ks": [KS[-1]] * N_SLOTS, "e": [E[-1]] * N_SLOTS, "d": [D[-1]] * len(STAGES)}
    assert decode(encode(lo)) == lo
    assert decode(encode(hi)) == hi
