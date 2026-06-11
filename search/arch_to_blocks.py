"""CP 2.1 — translate an OFA arch_dict into an ordered MBConv block list.

Given a canonical OFA arch spec ``{"ks": [20], "e": [20], "d": [5]}``, emit the
ordered ``("mbconv", cfg, input_shape)`` tuples the latency LUT is keyed on.
This is the bridge from a sampled subnet (``supernet.sampler``) to LUT-aware
cost (CP 2.2): summing ``LUT[row_key]`` over this list yields a candidate's
predicted latency / memory / params without re-measuring on the Jetson.

Only the searchable MBConv backbone is emitted. The stem (3->16), the
final-expand and feature-mix convs, and the classifier are fixed across every
arch — a constant offset that CP 2.2 adds separately, not a per-arch lookup.

The topology lives in :mod:`catalog.ofa_mbv3` (shared with the LUT grid so the
two never drift). Run as a script for the CP 2.1 DoD smoke test::

    python -m search.arch_to_blocks
"""

from __future__ import annotations

import random

from catalog.blocks import input_shape_for
from catalog.contracts import ArchDict, Block
from catalog.ofa_mbv3 import FIRST_BLOCK, KS, MAX_DEPTH, STAGES, D, E, stage_in_c
from catalog.sweep import row_key


def validate_arch_dict(arch_dict: ArchDict) -> None:
    """Reject malformed arch dicts at the boundary, naming every violation.

    Phase-3 search (BO / evolution) will generate arch dicts
    programmatically; a malformed one must fail here with a precise message,
    not as an IndexError deep inside the stage walk. ``type(x) is int``
    deliberately rejects ``bool`` and ``np.int64``: a non-plain int would
    change (bool) or crash (np.int64) the row_key JSON serialization
    downstream. Extra keys are tolerated for forward compatibility.
    """
    problems: list[str] = []
    n_slots = len(STAGES) * MAX_DEPTH
    checks = (("ks", n_slots, KS), ("e", n_slots, E), ("d", len(STAGES), D))
    for field, expected_len, allowed in checks:
        if field not in arch_dict:
            problems.append(f"missing key {field!r}")
            continue
        seq = arch_dict[field]  # type: ignore[literal-required]
        if not isinstance(seq, list) or len(seq) != expected_len:
            problems.append(f"{field!r} must be a list of length {expected_len}")
            continue
        bad = [x for x in seq if type(x) is not int or x not in allowed]
        if bad:
            problems.append(f"{field!r} has values outside {allowed}: {bad[:5]!r}")
    if problems:
        raise ValueError("invalid arch_dict: " + "; ".join(problems))


def arch_to_blocks(arch_dict: ArchDict) -> list[Block]:
    """OFA ``arch_dict`` -> ordered list of ``("mbconv", cfg, input_shape)``.

    ``ks`` / ``e`` are length ``5 * MAX_DEPTH`` (one slot per block position);
    ``d`` is length 5 (per-stage active depth). For stage ``s``, active block
    ``j in range(d[s])`` reads slot ``MAX_DEPTH * s + j``. Block 0 is the entry
    (prev_w -> out_w at the stage stride, at ``res_in``); blocks 1+ are repeats
    (out_w -> out_w, stride 1, at ``res_in // stride``).

    Raises ``ValueError`` on a malformed ``arch_dict`` (see
    :func:`validate_arch_dict`).
    """
    validate_arch_dict(arch_dict)
    ks, e, d = arch_dict["ks"], arch_dict["e"], arch_dict["d"]
    blocks: list[Block] = []

    def emit(cfg: dict) -> None:
        blocks.append(("mbconv", cfg, input_shape_for("mbconv", cfg)))

    emit(dict(FIRST_BLOCK))  # fixed, non-elastic

    for s, stage in enumerate(STAGES):
        out_c, se = stage["out_c"], stage["se"]
        res_in = stage["res_in"]
        res_out = res_in // stage["stride"]
        for j in range(d[s]):
            slot = MAX_DEPTH * s + j
            if j == 0:
                emit({"in_c": stage_in_c(s), "out_c": out_c, "kernel": ks[slot],
                      "stride": stage["stride"], "expand": e[slot], "se": se,
                      "res": res_in})
            else:
                emit({"in_c": out_c, "out_c": out_c, "kernel": ks[slot],
                      "stride": 1, "expand": e[slot], "se": se, "res": res_out})
    return blocks


def arch_to_keys(arch_dict: ArchDict) -> list[str]:
    """The LUT ``row_key`` for each block in the arch's ordered block list."""
    return [row_key(b, cfg, shape) for b, cfg, shape in arch_to_blocks(arch_dict)]


def _random_arch_dict(rng: random.Random) -> ArchDict:
    """A random arch in OFA's format (matches ``supernet.sampler.random_arch``)."""
    n = 5 * MAX_DEPTH
    return {
        "ks": [rng.choice(KS) for _ in range(n)],
        "e": [rng.choice(E) for _ in range(n)],
        "d": [rng.choice(D) for _ in range(5)],
    }


def _dod_smoke_test(n_archs: int = 10, seed: int = 0) -> None:
    """CP 2.1 DoD: every emitted tuple of N random archs matches a LUT row_key."""
    from pathlib import Path

    from lut.orchestrate.resume import completed_keys

    lut_path = Path(__file__).resolve().parents[1] / "data" / "lut.jsonl"
    known = completed_keys(lut_path)
    print(f"LUT: {len(known)} row_keys in {lut_path}")

    rng = random.Random(seed)
    all_ok = True
    for i in range(n_archs):
        arch = _random_arch_dict(rng)
        blocks = arch_to_blocks(arch)
        missing = [(cfg, tuple(shape)) for b, cfg, shape in blocks
                   if row_key(b, cfg, shape) not in known]
        tag = "OK" if not missing else f"MISSING {len(missing)}/{len(blocks)}"
        print(f"  arch {i}: d={arch['d']} -> {len(blocks):2d} blocks  [{tag}]")
        for cfg, shape in missing[:3]:
            print(f"      no LUT row for {cfg} shape={shape}")
        all_ok &= not missing

    print("DoD PASS" if all_ok else "DoD FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _dod_smoke_test()
