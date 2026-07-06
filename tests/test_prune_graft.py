"""CP 6.1 — prune/prune_graft.py: the CPU-smoke DoD (prune 20% → forward OK, shapes
unchanged, params reduced, changed convs %16-aligned) plus the two hard guards."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")     # the Pose head → .venv-nas only
pytest.importorskip("torch_pruning")   # the CP 6.1 dependency

from torch import nn  # noqa: E402

from catalog.ofa_mbv3 import STAGES, stage_in_c  # noqa: E402
from detect.adapter import ChannelAdapter  # noqa: E402
from detect.pose_model import YOLO11N_HEAD_CHANNELS, GraftedPoseModel  # noqa: E402
from prune.prune_graft import head_ignored_layers, prune_graft  # noqa: E402
from supernet.pose_backbone import FEATURE_TAP_CHANNELS, PoseBackbone  # noqa: E402

IMGSZ = 128


class _StubBackbone(nn.Module):
    """Torch-only OFA stand-in (mirrors tests/test_grafted_pose_model.py)."""

    def __init__(self, depths: list[int]) -> None:
        super().__init__()
        self.first_conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        blocks: list[nn.Module] = [nn.Conv2d(16, 16, 1)]
        for s, stage in enumerate(STAGES):
            out_c = stage["out_c"]
            for j in range(depths[s]):
                in_c = stage_in_c(s) if j == 0 else out_c
                stride = stage["stride"] if j == 0 else 1
                blocks.append(nn.Conv2d(in_c, out_c, 1, stride=stride))
        self.blocks = nn.ModuleList(blocks)


def _build_graft() -> GraftedPoseModel:
    from ultralytics.nn.modules.head import Pose

    depths = [2, 3, 4, 2, 3]
    backbone = PoseBackbone(_StubBackbone(depths), depths)
    adapter = ChannelAdapter(FEATURE_TAP_CHANNELS, YOLO11N_HEAD_CHANNELS)
    head = Pose(nc=1, kpt_shape=(8, 3), ch=YOLO11N_HEAD_CHANNELS)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    return GraftedPoseModel(backbone, adapter, head)


def _decoded(model: GraftedPoseModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model.predict(x)
    return out[0] if isinstance(out, tuple) else out


def test_dod_prune_20pct_smoke() -> None:
    model = _build_graft()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    shape_before = tuple(_decoded(model, x).shape)

    report = prune_graft(model, x, ratio=0.2, round_to=16)

    assert report["params_after"] < report["params_before"]      # params reduced
    assert report["params_sparsity"] > 0.05
    assert report["n_convs_changed"] > 0
    assert report["all_rounded"], f"misaligned convs: {report['misaligned']}"
    decoded = _decoded(model, x)
    assert tuple(decoded.shape) == shape_before                  # output format untouched
    assert bool(torch.isfinite(decoded).all())
    # the semantic output convs kept their channel counts
    head = model.model[-1]
    assert head.cv2[0][-1].out_channels == 64                    # 4 * reg_max
    assert head.cv3[0][-1].out_channels == 1                     # nc
    assert head.cv4[0][-1].out_channels == 24                    # 8 kpts * 3


def test_frozen_parameters_refused() -> None:
    from detect.pose_model import freeze_module

    model = _build_graft()
    freeze_module(model.model[-1])
    with pytest.raises(ValueError, match="frozen parameter"):
        prune_graft(model, torch.rand(1, 3, IMGSZ, IMGSZ), ratio=0.2)


def test_bad_ratio_and_ignored_collection() -> None:
    model = _build_graft()
    ignored = head_ignored_layers(model)
    # 3 scales × (cv2, cv3, cv4) + dfl.conv
    assert len(ignored) == 10
    with pytest.raises(ValueError, match="ratio"):
        prune_graft(model, torch.rand(1, 3, IMGSZ, IMGSZ), ratio=1.5)


def test_necked_graft_prunes_too() -> None:
    """The CP 6.2 input may be winner-v1.5 WITH a neck — DepGraph must cope with the scalar
    gates (0-dim params) and the fusion adds."""
    from ultralytics.nn.modules.head import Pose

    from detect.neck import build_neck

    depths = [2, 3, 4, 2, 3]
    backbone = PoseBackbone(_StubBackbone(depths), depths)
    adapter = ChannelAdapter(FEATURE_TAP_CHANNELS, YOLO11N_HEAD_CHANNELS)
    head = Pose(nc=1, kpt_shape=(8, 3), ch=YOLO11N_HEAD_CHANNELS)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    model = GraftedPoseModel(backbone, adapter, head,
                             neck=build_neck("topdown", YOLO11N_HEAD_CHANNELS))
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    shape_before = tuple(_decoded(model, x).shape)
    report = prune_graft(model, x, ratio=0.2, round_to=16)
    assert report["params_after"] < report["params_before"]
    assert tuple(_decoded(model, x).shape) == shape_before
