"""CP 3.1 — OFA arch_dict <-> flat surrogate vector (search-space encoder).

The Phase-3 surrogate (CP 3.2 NSGA-II / CP 3.3 BO) searches over a fixed-length
numeric vector, not the nested OFA ``{"ks":[20], "e":[20], "d":[5]}`` dict. This
module is the bijection between the two, plus the per-axis metadata the GP kernel
needs and a canonical form that quotients out OFA's depth-inactive don't-cares.

Layout (length ``VECTOR_LEN`` = 45): ``[ ks(20) | e(20) | d(5) ]``. Each slot is
stored as a **category index** into its choice set (``KS``/``E``/``D`` from
:mod:`catalog.ofa_mbv3`), so categorical axes (ks, e) feed a Hamming kernel as
0/1 mismatches while the ordinal depth axis (d) keeps its order for a Matern
kernel. Lengths derive from the topology, never hardcoded — CP 7.1 extends this
same file for new op choices.

Two distinct jobs, deliberately separated:

- ``encode`` / ``decode`` are a **lossless** bijection over all 45 slots. OFA's
  ``sample_active_subnet`` fills every ks/e slot — including the trailing slots a
  stage's depth switches off — with random values, so the DoD
  (``decode(encode(arch)) == arch``) requires preserving the don't-cares verbatim.
  These functions never mask.
- ``canonical`` masks those inactive slots to ``INACTIVE`` so two archs that
  differ only in don't-care slots (the *same* network) collapse to one point. The
  surrogate's distance metric consumes ``canonical``; masking here, not in
  ``encode``, is what lets the bijection and the surrogate coexist.

Run the DoD smoke test::

    python -m search.space
"""
from __future__ import annotations

from collections.abc import Sequence

from catalog.contracts import ArchDict
from catalog.ofa_mbv3 import KS, MAX_DEPTH, STAGES, D, E
from search.arch_to_blocks import validate_arch_dict

N_SLOTS = len(STAGES) * MAX_DEPTH       # ks/e length: one slot per block position
N_STAGES = len(STAGES)                   # d length: one active-depth per stage
VECTOR_LEN = 2 * N_SLOTS + N_STAGES      # 45

INACTIVE = -1                            # canonical sentinel for a depth-off slot

# Per-dimension surrogate metadata (data-driven; consumed by the CP 3.3 kernel).
AXIS_TYPES: list[str] = ["categorical"] * (2 * N_SLOTS) + ["ordinal"] * N_STAGES
AXIS_CARDINALITIES: list[int] = (
    [len(KS)] * N_SLOTS + [len(E)] * N_SLOTS + [len(D)] * N_STAGES
)


def encode(arch_dict: ArchDict) -> list[int]:
    """OFA ``arch_dict`` -> flat list of category indices (length ``VECTOR_LEN``).

    Lossless over every slot, inactive ones included — see the module docstring.
    """
    return (
        [KS.index(v) for v in arch_dict["ks"]]
        + [E.index(v) for v in arch_dict["e"]]
        + [D.index(v) for v in arch_dict["d"]]
    )


def decode(vec: Sequence[int]) -> ArchDict:
    """Inverse of :func:`encode`. Emits plain ``int`` and validates the result.

    The index->value maps (``KS``/``E``/``D``) return Python ``int``s, so the
    arch_dict passes ``validate_arch_dict`` (which rejects ``np.int64`` / ``bool``)
    and serializes cleanly into the LUT ``row_key`` downstream.
    """
    ks = [KS[i] for i in vec[:N_SLOTS]]
    e = [E[i] for i in vec[N_SLOTS : 2 * N_SLOTS]]
    d = [D[i] for i in vec[2 * N_SLOTS : 2 * N_SLOTS + N_STAGES]]
    arch: ArchDict = {"ks": ks, "e": e, "d": d}
    validate_arch_dict(arch)
    return arch


def canonical(vec: Sequence[int]) -> list[int]:
    """Mask depth-inactive ks/e slots to ``INACTIVE`` for the surrogate.

    For stage ``s`` with active depth ``D[d_index]``, every slot ``j >= depth``
    (ks at ``MAX_DEPTH*s+j``, e at ``N_SLOTS + MAX_DEPTH*s+j``) becomes
    ``INACTIVE``, so archs identical in their active structure share one canonical
    vector. The depth block is left intact.
    """
    out = list(vec)
    d_indices = vec[2 * N_SLOTS : 2 * N_SLOTS + N_STAGES]
    for s, d_idx in enumerate(d_indices):
        depth = D[d_idx]
        for j in range(depth, MAX_DEPTH):
            out[MAX_DEPTH * s + j] = INACTIVE            # ks slot
            out[N_SLOTS + MAX_DEPTH * s + j] = INACTIVE  # e slot
    return out


def _dod_smoke_test(n_archs: int = 100, seed: int = 0) -> None:
    """CP 3.1 DoD: decode(encode(arch)) == arch for N random archs."""
    import random

    from search.arch_to_blocks import random_arch_dict

    rng = random.Random(seed)
    ok = 0
    for i in range(n_archs):
        arch = random_arch_dict(rng)
        if decode(encode(arch)) == arch:
            ok += 1
        else:
            print(f"  arch {i}: round-trip MISMATCH  d={arch['d']}")
    print(f"round-trip: {ok}/{n_archs} archs")
    print("DoD PASS" if ok == n_archs else "DoD FAIL")
    if ok != n_archs:
        raise SystemExit(1)


if __name__ == "__main__":
    _dod_smoke_test()
