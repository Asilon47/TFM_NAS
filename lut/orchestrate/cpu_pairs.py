"""The declared Jetson-row <-> local-ONNX pairing for the CPU rank check.

Jetson row names in ``data/e2e/`` do not match the ONNX filenames under ``models/`` and cannot
be derived from them: ``baseline_recheck_640`` is ``yolo11n_pose_640.onnx``, ``dense_ctrl_n_640``
is ``dense_w25_640.onnx``, ``winner_v1_e2e_640`` is ``winner_v1_noneck_e2e_640.onnx``. Any
auto-matching heuristic would mis-pair models and still produce a plausible-looking correlation,
so the map is explicit, reviewed data.

Exclusions are deliberate (see spec §4): ``*_backbone_640`` rows are a different network scope;
``fallback_idx{3,11}`` have no local ONNX; ``recover_graft_r{40,60}_640.onnx`` are left out
because ``screen_prune_graft/graft_prune_r{40,60}_e2e_640.onnx`` are the artifacts whose Jetson
latencies (11.81 / 9.00 ms) match the CP 6.2-G rungs recorded in ``models/README.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANARY = "baseline_recheck_640"

#: Families whose latency/params relationship the project already trusts -- the OLS reference.
#: The anchor (yolo11s, 9.7M params) is excluded: at 2.5x the next-largest reference model it
#: would dominate the fit as a leverage point. dense_nas is excluded as a distinct design process.
REFERENCE_FAMILIES = frozenset({"dense", "prune", "baseline"})

#: The families under test.
GRAFT_FAMILIES = frozenset({"graft", "graft_pruned"})


@dataclass(frozen=True)
class Pair:
    """One measured Jetson row and its local ONNX counterpart."""

    jetson_name: str
    onnx: str
    family: str


PAIRS: tuple[Pair, ...] = (
    Pair("baseline_recheck_640", "models/baseline/yolo11n_pose_640.onnx", "baseline"),
    Pair("yolo11s_pose_640", "models/anchor/yolo11s_pose_640.onnx", "anchor"),
    Pair("dense_ctrl_n_640", "models/dense_scaled/dense_w25_640.onnx", "dense"),
    Pair("dense_d33_w20_640", "models/dense_scaled/dense_w20_640.onnx", "dense"),
    Pair("dense_d50_w15_640", "models/dense_scaled/dense_w15_640.onnx", "dense"),
    Pair("dense_w13_640", "models/dense_scaled/dense_w13_640.onnx", "dense"),
    Pair("dense_w18_640", "models/dense_scaled/dense_w18_640.onnx", "dense"),
    Pair("dense_w22_640", "models/dense_scaled/dense_w22_640.onnx", "dense"),
    Pair("dense_w30_640", "models/dense_scaled/dense_w30_640.onnx", "dense"),
    Pair("densenas_s31_640", "models/dense_nas/dense_s31-40-40-40-13_o100_640.onnx", "dense_nas"),
    Pair("densenas_s39_640", "models/dense_nas/dense_s39-40-38-38-14_o100_640.onnx", "dense_nas"),
    Pair("densenas_s40_640", "models/dense_nas/dense_s40-38-39-36-13_o100_640.onnx", "dense_nas"),
    Pair("prune_base_r10_640", "models/pruned_baseline/prune_r10_640.onnx", "prune"),
    Pair("prune_base_r15_640", "models/pruned_baseline/prune_r15_640.onnx", "prune"),
    Pair("prune_base_r20_640", "models/pruned_baseline/prune_r20_640.onnx", "prune"),
    Pair("prune_base_r30_640", "models/pruned_baseline/prune_r30_640.onnx", "prune"),
    Pair("prune_base_r35_640", "models/pruned_baseline/prune_r35_640.onnx", "prune"),
    Pair("prune_base_r45_640", "models/pruned_baseline/prune_r45_640.onnx", "prune"),
    Pair("prune_base_r55_640", "models/pruned_baseline/prune_r55_640.onnx", "prune"),
    Pair("winner_v1_e2e_640", "models/graft/winner_v1_noneck_e2e_640.onnx", "graft"),
    Pair("winner_v1_v2topdown_e2e_640", "models/graft/winner_v1_v2topdown_e2e_640.onnx", "graft"),
    Pair("winner_v1_v3pan_e2e_640", "models/graft/winner_v1_v3pan_e2e_640.onnx", "graft"),
    Pair(
        "graft_prune_r20_e2e_640",
        "models/screen_prune_graft/graft_prune_r20_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_prune_r40_e2e_640",
        "models/screen_prune_graft/graft_prune_r40_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_prune_r60_e2e_640",
        "models/screen_prune_graft/graft_prune_r60_e2e_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_r50_gtay_640",
        "models/graft_pruned/recover_graft_r50_gtay_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_r60_gtay_640",
        "models/graft_pruned/recover_graft_r60_gtay_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_halp_9p0_640",
        "models/graft_pruned/recover_graft_halp_fp32_9p0_640.onnx",
        "graft_pruned",
    ),
    Pair(
        "graft_halp_10p4_640",
        "models/graft_pruned/recover_graft_halp_fp32_10p4_640.onnx",
        "graft_pruned",
    ),
)


def resolve_pairs(root: Path) -> list[tuple[Pair, Path]]:
    """Resolve every pair's ONNX against ``root``; hard-fail listing ALL misses.

    Fails before any timing rather than benching a silently-short set: a missing model would
    weaken the correlation without announcing itself.
    """
    resolved: list[tuple[Pair, Path]] = []
    missing: list[str] = []
    for pair in PAIRS:
        path = root / pair.onnx
        if path.is_file():
            resolved.append((pair, path))
        else:
            missing.append(pair.onnx)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(PAIRS)} ONNX files missing under {root}:\n  "
            + "\n  ".join(missing)
        )
    return resolved
