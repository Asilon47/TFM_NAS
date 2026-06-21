"""Integration tests for detect.pose_model.GraftedPoseModel — the CP 2.4 trainable graft.

These prove the *loss/grad path*: a backbone grafted under a real Ultralytics Pose head trains
end-to-end (forward → v8PoseLoss → backward reaches the backbone). They need ``ultralytics`` (the
Pose head) so they ``importorskip`` and run only under ``.venv-nas``; they use a torch-only
**stub** backbone (right channels/strides, cheap 1x1 convs) so the 31 MB OFA checkpoint is not
required. The real OFA wiring is ``detect/pose_model.py``'s ``__main__`` smoke.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")  # the Pose head → .venv-nas only

from torch import nn  # noqa: E402

from catalog.ofa_mbv3 import STAGES, stage_in_c  # noqa: E402
from detect.adapter import ChannelAdapter  # noqa: E402
from detect.pose_model import (  # noqa: E402
    YOLO11N_HEAD_CHANNELS,
    GraftedPoseModel,
    default_pose_args,
)
from supernet.pose_backbone import FEATURE_TAP_CHANNELS, PoseBackbone  # noqa: E402

IMGSZ = 128  # small → P3/P4/P5 grids 16/8/4; keeps the CPU loss/grad smoke fast


class _StubBackbone(nn.Module):
    """Torch-only OFA stand-in: real first_conv (3->16, /2) + per-stage 1x1 convs at the true
    channels/strides from ``catalog.ofa_mbv3`` (mirrors tests/test_pose_backbone.py)."""

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


def _build_stub_grafted() -> GraftedPoseModel:
    """A GraftedPoseModel over the stub backbone — exercises the wiring without OFA."""
    from ultralytics.nn.modules.head import Pose

    depths = [2, 3, 4, 2, 3]
    backbone = PoseBackbone(_StubBackbone(depths), depths)
    adapter = ChannelAdapter(FEATURE_TAP_CHANNELS, YOLO11N_HEAD_CHANNELS)
    head = Pose(nc=1, kpt_shape=(8, 3), ch=YOLO11N_HEAD_CHANNELS)
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    return GraftedPoseModel(backbone, adapter, head)


def _synthetic_pose_batch(imgsz: int = IMGSZ, nkpt: int = 8) -> dict:
    """One gate (box + 8 visible keypoints) in a single image — a valid v8PoseLoss batch."""
    img = torch.rand(1, 3, imgsz, imgsz)
    kpts = torch.rand(1, nkpt, 3)
    kpts[..., 2] = 1.0  # all keypoints visible
    return {
        "img": img,
        "batch_idx": torch.zeros(1),            # the 1 object belongs to image 0
        "cls": torch.zeros(1, 1),               # class 0 == gate
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # normalized xywh, centered
        "keypoints": kpts,
    }


# --- the Ultralytics contract the wrapper must expose ------------------------

def test_grafted_exposes_head_as_last_model_layer():
    # init_criterion/v8PoseLoss/_apply all reach the head via self.model[-1] — it MUST be the head.
    model = _build_stub_grafted()
    from ultralytics.nn.modules.head import Pose

    assert isinstance(model.model[-1], Pose)


def test_grafted_exposes_loss_gains_and_metadata():
    model = _build_stub_grafted()
    for gain in ("box", "cls", "dfl", "pose", "kobj"):  # v8PoseLoss reads model.args.<gain>
        assert hasattr(model.args, gain)
    assert model.names == {0: "gate"}
    assert model.nc == 1
    assert list(model.kpt_shape) == [8, 3]
    assert model.task == "pose"
    assert torch.equal(model.stride, model.model[-1].stride)


def test_default_pose_args_has_all_loss_gains():
    args = default_pose_args()
    for gain in ("box", "cls", "dfl", "pose", "kobj"):
        assert isinstance(getattr(args, gain), (int, float))


# --- the predict path: bypasses Ultralytics .f/.i routing --------------------

def test_grafted_predict_returns_pose_head_dict_in_train_mode():
    model = _build_stub_grafted().train()
    out = model.predict(torch.rand(1, 3, IMGSZ, IMGSZ))
    # Pose head in train mode returns the raw-output dict v8PoseLoss consumes.
    assert isinstance(out, dict)
    assert {"feats", "kpts"} <= set(out)


# --- the money test: loss is finite and grad reaches the backbone ------------

def test_grafted_loss_flows_grad_to_backbone():
    model = _build_stub_grafted().train()
    loss, items = model(_synthetic_pose_batch())
    assert loss.shape == (5,)  # box, pose, kobj, cls, dfl
    assert torch.isfinite(loss).all()

    loss.sum().backward()
    stem_grad = model.model[0].first_conv.weight.grad  # gradient must reach the backbone stem
    assert stem_grad is not None
    assert torch.isfinite(stem_grad).all()
    assert stem_grad.abs().sum() > 0  # cls-BCE flows on every anchor → non-zero


@pytest.mark.slow
def test_grafted_overfits_single_image():
    # A few steps on ONE batch must drive the loss down — proves the optimization path, not just
    # one backward. This is the CP 2.4 "1-image overfit" CPU smoke.
    model = _build_stub_grafted().train()
    batch = _synthetic_pose_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def step() -> float:
        loss, _ = model(batch)
        total = loss.sum()
        opt.zero_grad()
        total.backward()
        opt.step()
        return float(total.detach())

    first = step()
    for _ in range(25):
        last = step()
    assert last < first  # loss decreased
