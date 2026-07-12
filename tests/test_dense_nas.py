"""Stage-3 dense-space NAS — search/dense_nas.py pure parts (yaml surgery + tags + guards)."""
import pytest

from search.dense_nas import (
    BACKBONE_STAGE,
    DIVISOR,
    HEAD_STAGE,
    N_STAGES,
    SCALE_HI,
    STAGE5_LO,
    candidate_tag,
    stagewise_yaml,
)

_BASE = {
    "nc": 1,
    "kpt_shape": [8, 3],
    "scales": {"n": [0.5, 0.25, 1024]},
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],          # B0  stage 1
        [-1, 1, "Conv", [128, 3, 2]],         # B1  stage 2
        [-1, 2, "C3k2", [256, False, 0.25]],  # B2  stage 2
        [-1, 1, "Conv", [256, 3, 2]],         # B3  stage 3
        [-1, 2, "C3k2", [512, False, 0.25]],  # B4  stage 3
        [-1, 1, "Conv", [512, 3, 2]],         # B5  stage 4
        [-1, 2, "C3k2", [512, True]],         # B6  stage 4
        [-1, 1, "Conv", [1024, 3, 2]],        # B7  stage 5
        [-1, 2, "C3k2", [1024, True]],        # B8  stage 5
        [-1, 1, "SPPF", [1024, 5]],           # B9  stage 5
        [-1, 2, "C2PSA", [1024]],             # B10 stage 5
    ],
    "head": [
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],        # H2  stage 4
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],        # H5  stage 3
        [-1, 1, "Conv", [256, 3, 2]],         # H6  stage 3
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],        # H8  stage 4
        [-1, 1, "Conv", [512, 3, 2]],         # H9  stage 4
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],        # H11 stage 5
        [[16, 19, 22], 1, "Pose", ["nc", "kpt_shape"]],
    ],
}


def test_stage_maps_cover_exactly_the_channel_layers():
    assert set(BACKBONE_STAGE) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert set(HEAD_STAGE) == {2, 5, 6, 8, 9, 11}
    assert set(BACKBONE_STAGE.values()) | set(HEAD_STAGE.values()) == set(range(1, N_STAGES + 1))


def test_stagewise_yaml_scales_by_stage_and_aligns():
    scales = [0.20, 0.25, 0.30, 0.15, 0.20]
    out = stagewise_yaml(_BASE, scales)
    assert out["scales"] == {"n": [0.5, 1.0, 4096]}          # widths become absolute
    assert out["backbone"][0][3][0] == 16                    # 64×0.20=12.8 → floor 16
    assert out["backbone"][2][3][0] == 64                    # 256×0.25
    assert out["backbone"][4][3][0] == 160                   # 512×0.30 → 153.6 → 160
    assert out["backbone"][10][3][0] == 208                  # C2PSA: 1024×0.20 → 204.8 → 208
    assert out["head"][5][3][0] == 80                        # H5 stage3: 256×0.30 → 76.8 → 80
    assert out["head"][11][3][0] == 208                      # H11 stage5
    for sec, stage_of in (("backbone", BACKBONE_STAGE), ("head", HEAD_STAGE)):
        for i in stage_of:
            assert out[sec][i][3][0] % DIVISOR == 0
    # non-channel layers untouched; source dict not mutated
    assert out["head"][0][3] == ["None", 2, "nearest"]
    assert _BASE["backbone"][0][3][0] == 64


def test_stagewise_yaml_guards():
    with pytest.raises(ValueError, match="stage scales"):
        stagewise_yaml(_BASE, [0.25] * 4)
    with pytest.raises(ValueError, match="outside"):
        stagewise_yaml(_BASE, [0.25, 0.25, 0.25, 0.25, SCALE_HI + 0.05])
    with pytest.raises(ValueError, match="outside"):
        stagewise_yaml(_BASE, [0.25, 0.25, 0.25, 0.25, STAGE5_LO - 0.02])  # C2PSA floor


def test_candidate_tag_roundtrips_the_grid():
    assert candidate_tag([0.25] * 5) == "s25-25-25-25-25"    # the yolo11n point
    assert candidate_tag([0.1, 0.2, 0.3, 0.4, 0.13]) == "s10-20-30-40-13"
