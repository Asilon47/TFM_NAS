"""Graft a YOLO11-pose head onto a sampled OFA backbone (D1 gate-pose prototype).

``build_pose_model`` wires ``PoseBackbone -> ChannelAdapter -> Ultralytics Pose head`` so a
sampled OFA subnet drives gate detection + 8-keypoint pose. The backbone emits the fixed
``(40, 112, 160)`` channels; the adapter remaps them to the YOLO11n-pose head's neck-output
channels (``(64, 128, 256)`` at nano width); the head is the real
``ultralytics.nn.modules.head.Pose`` so the output format matches the deployed model.

The head is freshly initialized here — what this prototype proves is the *wiring* (an OFA
subnet forwards through a real YOLO pose head at 640x640). Cloning ``yolo11n-pose.pt``'s trained
head weights is a same-shape ``state_dict`` copy once the channels line up, and wrapping the
whole thing as an Ultralytics model for end-to-end pose training/val is the GPU-gated follow-up.

``ultralytics`` is imported lazily, so ``import detect.pose_model`` and the adapter tests stay
ultralytics-free under ``.venv``. The real forward is the ``__main__`` smoke, run under
``.venv-nas`` (mirrors ``supernet/sampler.py``)::

    python -m detect.pose_model
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from detect.adapter import ChannelAdapter
from detect.neck import build_neck
from net2net.graft_init import apply_adapter_init
from supernet.pose_backbone import PoseBackbone

# YOLO11n-pose neck-output channels feeding the Pose head (nano width_multiple=0.25).
YOLO11N_HEAD_CHANNELS: tuple[int, int, int] = (64, 128, 256)
# P3/P4/P5 strides — set on the head so its eval-mode decode has the anchor grid scale.
HEAD_STRIDES: tuple[int, int, int] = (8, 16, 32)


class PoseModel(nn.Module):
    """OFA backbone -> 1x1 channel adapter -> YOLO Pose head."""

    def __init__(self, backbone: PoseBackbone, adapter: ChannelAdapter, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: Tensor) -> Any:
        feats = self.adapter(self.backbone(x))
        return self.head(list(feats))


def build_pose_model(
    arch_dict: dict,
    *,
    nc: int = 1,
    kpt_shape: tuple[int, int] = (8, 3),
    head_channels: Sequence[int] = YOLO11N_HEAD_CHANNELS,
    supernet: Any = None,
) -> PoseModel:
    """Sample the OFA subnet for ``arch_dict`` and graft a fresh YOLO Pose head onto it.

    ``nc=1`` (gate), ``kpt_shape=(8, 3)`` (8 keypoints, x/y/visibility) match ``dataset.yaml``.
    """
    from ultralytics.nn.modules.head import Pose  # lazy: needs .venv-nas

    from supernet.sampler import sample

    backbone = PoseBackbone(sample(arch_dict, supernet), arch_dict["d"])
    adapter = ChannelAdapter(backbone.out_channels, head_channels)
    head = Pose(nc=nc, kpt_shape=kpt_shape, ch=tuple(head_channels))
    head.stride = torch.tensor([float(s) for s in HEAD_STRIDES])
    return PoseModel(backbone, adapter, head)


# --------------------------------------------------------------------------------------------
# CP 2.4 proxy repair: warm-start (+ optionally freeze) the Pose head. The original proxy
# fine-tuned a *randomly-initialized* head for 5 epochs, so its mAP measured head-init luck, not
# backbone quality (rank Kendall-τ=0.20; idx8 = best backbone but worst proxy). Cloning a trained
# gate head removes that variance; freezing it turns the proxy into "how well does this backbone
# feed a fixed, competent head" — i.e. a backbone-quality probe. These two helpers are generic
# torch state_dict logic (no ultralytics) so they unit-test under .venv; ``_donor_head_state`` is
# the only ultralytics-touching piece (lazy import, GPU/.venv-nas-gated).
# --------------------------------------------------------------------------------------------

def freeze_module(module: nn.Module) -> nn.Module:
    """Set ``requires_grad=False`` on every parameter of ``module``; return it for chaining."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def warm_start_head(head: nn.Module, donor_state: Mapping[str, Tensor]) -> dict[str, list[str]]:
    """Copy ``donor_state`` into ``head`` where key **and** shape match; leave the rest at init.

    Shape-aware so one path serves both donors: the deployed **gate** ``yolo11n-pose`` (nc=1,
    8-kpt → every tensor matches, whole head transfers) and the generic **COCO** ``yolo11n-pose``
    (17-kpt → the keypoint branch ``cv4`` mismatches and stays at init, while box/cls transfer).

    Returns ``{"copied": [...], "skipped": [...]}`` (state-dict keys). Raises ``ValueError`` if
    *nothing* matches — a sign the donor is not a compatible pose head (wrong checkpoint), which
    must fail loudly rather than silently warm-start nothing.
    """
    target = head.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    for key, tensor in target.items():
        src = donor_state.get(key)
        if src is not None and src.shape == tensor.shape:
            target[key] = src
            copied.append(key)
        else:
            skipped.append(key)
    if not copied:
        raise ValueError(
            f"no head tensors matched the donor checkpoint (head has {len(target)} tensors, none "
            "shape-matched) — is this an Ultralytics pose model?"
        )
    head.load_state_dict(target)
    return {"copied": copied, "skipped": skipped}


def _donor_head_state(weights_path: Any) -> dict[str, Any]:
    """The Pose head ``state_dict`` of an Ultralytics pose checkpoint (e.g. ``yolo11n-pose.pt``).

    The head is the last module of the wrapped task model (``model.model[-1]``). Lazy ultralytics
    import — only the warm-start path (GPU/.venv-nas) reaches it.
    """
    from ultralytics import YOLO

    donor = YOLO(str(weights_path))
    return donor.model.model[-1].state_dict()


# --------------------------------------------------------------------------------------------
# CP 2.4: the *trainable* graft. ``GraftedPoseModel`` subclasses Ultralytics' ``PoseModel`` so
# the inherited ``loss()`` / ``init_criterion()`` / ``v8PoseLoss`` / ``_apply`` work unchanged —
# they only ever reach the head via ``self.model[-1]``. We override just construction (skip the
# yaml-based build) and ``_predict_once`` (the OFA backbone + adapter lack Ultralytics' per-layer
# ``.f``/``.i`` routing attributes, so we run backbone→adapter→head directly). The subclass is
# built lazily (via module ``__getattr__``) to keep ``import detect.pose_model`` ultralytics-free
# under ``.venv``; only touching ``detect.pose_model.GraftedPoseModel`` pulls ultralytics in.
# --------------------------------------------------------------------------------------------

_grafted_cls: type | None = None


def default_pose_args() -> Any:
    """Fresh Ultralytics hyperparameter namespace — supplies ``v8PoseLoss`` its loss gains
    (``box``/``cls``/``dfl``/``pose``/``kobj``); the grafted model exposes it as ``.args``."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    return get_cfg(DEFAULT_CFG)


def _grafted_pose_model_cls() -> type:
    """Define (once) and return ``GraftedPoseModel``, subclass of Ultralytics' ``PoseModel``."""
    global _grafted_cls
    if _grafted_cls is not None:
        return _grafted_cls

    from ultralytics.nn.tasks import PoseModel as _UltralyticsPoseModel

    class GraftedPoseModel(_UltralyticsPoseModel):  # type: ignore[misc, valid-type]
        """OFA backbone → adapter → YOLO Pose head, wrapped as a trainable Ultralytics PoseModel.

        Bypasses ``DetectionModel.__init__`` (which builds ``self.model`` from a yaml + runs a
        stride-inference forward): we already hold the assembled parts. ``self.model`` is a
        ``Sequential(backbone, adapter, head)`` so ``self.model[-1]`` is the Pose head — the only
        thing the inherited loss/criterion/``_apply`` paths require.
        """

        def __init__(
            self,
            backbone: nn.Module,
            adapter: nn.Module,
            head: nn.Module,
            *,
            neck: nn.Module | None = None,
            nc: int = 1,
            names: dict[int, str] | None = None,
            args: Any = None,
        ) -> None:
            nn.Module.__init__(self)  # NOT super().__init__ — skip the yaml-based build
            # model[-1] == head is THE contract (loss/criterion/_apply reach it there). With
            # neck=None the layout stays the original 3-module Sequential, so pre-CP-5.1
            # state_dicts (e.g. full_finetune_weights.pt) keep loading unchanged.
            stages = (backbone, adapter, head) if neck is None else (backbone, adapter, neck, head)
            self.model = nn.Sequential(*stages)
            self.save: list[int] = []
            self.nc = nc
            self.names = names if names is not None else {0: "gate"}
            self.kpt_shape = list(head.kpt_shape)  # type: ignore[attr-defined]
            self.stride = head.stride  # type: ignore[attr-defined]
            self.args = args if args is not None else default_pose_args()
            self.yaml = {"nc": nc, "kpt_shape": self.kpt_shape, "task": "pose"}
            self.task = "pose"
            self.inplace = True

        def _predict_once(self, x: Tensor, profile: bool = False, visualize: bool = False,
                          embed: Any = None) -> Any:
            *body, head = self.model  # 3 modules (no neck) or 4 (with neck) — head stays last
            feats: Any = x
            for module in body:
                feats = module(feats)
            return head(list(feats))

    _grafted_cls = GraftedPoseModel
    return _grafted_cls


def __getattr__(name: str) -> Any:  # PEP 562: lazy ultralytics-backed symbols
    if name == "GraftedPoseModel":
        return _grafted_pose_model_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_grafted_pose_model(
    arch_dict: dict,
    *,
    nc: int = 1,
    kpt_shape: tuple[int, int] = (8, 3),
    head_channels: Sequence[int] = YOLO11N_HEAD_CHANNELS,
    supernet: Any = None,
    head_weights: Any = None,
    freeze_head: bool = False,
    adapter_init: str | None = None,
    neck: str | None = None,
) -> Any:
    """Sample the OFA subnet for ``arch_dict`` and graft a trainable YOLO Pose head onto it.

    Like :func:`build_pose_model` but returns a :class:`GraftedPoseModel` (Ultralytics
    ``PoseModel`` subclass) wired for end-to-end pose training/eval. ``head.bias_init()`` seeds the
    head's cls/box biases the way Ultralytics does (needs ``stride`` set first).

    ``head_weights`` (a pose ``.pt`` path) warm-starts the head from a trained donor via
    :func:`warm_start_head` (CP 2.4 proxy repair); ``freeze_head`` then locks it so the short
    fine-tune adapts only the backbone+adapter to a fixed, competent head. Both default off (the
    original fresh-random behaviour).

    ``adapter_init`` (CP 4.4, e.g. ``"net2wider"``) replaces the adapter's random init with the
    identity-embedding prior (``net2net.graft_init``) so the head sees real backbone features
    from step 0; ``None`` keeps the original random 1×1s (the CP 5.2 V0 control).

    ``neck`` (CP 5.1): ``None`` keeps the neck-less 3-module graft (old state_dicts load
    unchanged); ``"topdown"`` (V2) / ``"pan"`` (V3) insert a ``detect.neck``
    ``ZeroGatedTopDownNeck`` between adapter and head — identity at init (zero gates), so the
    frozen donor head sees unchanged inputs on day 0.
    """
    from ultralytics.nn.modules.head import Pose

    from supernet.sampler import sample

    backbone = PoseBackbone(sample(arch_dict, supernet), arch_dict["d"])
    adapter = ChannelAdapter(backbone.out_channels, head_channels)
    if adapter_init is not None:
        apply_adapter_init(adapter, adapter_init)
    head = Pose(nc=nc, kpt_shape=kpt_shape, ch=tuple(head_channels))
    head.stride = torch.tensor([float(s) for s in HEAD_STRIDES])
    head.bias_init()
    if head_weights is not None:
        report = warm_start_head(head, _donor_head_state(head_weights))
        print(f"[warm-start] head: copied {len(report['copied'])}, "
              f"skipped {len(report['skipped'])} (donor: {head_weights})")
    if freeze_head:
        freeze_module(head)
    neck_module = build_neck(neck, head_channels)
    return _grafted_pose_model_cls()(backbone, adapter, head, neck=neck_module, nc=nc)


if __name__ == "__main__":  # real OFA + YOLO-head integration smoke — run under .venv-nas
    from supernet.sampler import load_supernet, random_arch

    sn = load_supernet()
    arch = random_arch(sn)
    model = build_pose_model(arch, supernet=sn).train()  # train mode -> raw head outputs
    with torch.no_grad():
        out = model(torch.randn(1, 3, 640, 640))
    # Ultralytics Pose returns a dict in train mode, and (decoded, dict) in eval mode.
    res = out if isinstance(out, dict) else out[1]
    nk = 8 * 3  # keypoints * (x, y, visibility)
    n_kpt, n_score = res["kpts"].shape[1], res["scores"].shape[1]
    print(f"arch d={arch['d']}  head_channels={YOLO11N_HEAD_CHANNELS}")
    print(f"  boxes {tuple(res['boxes'].shape)}  scores {tuple(res['scores'].shape)}  "
          f"kpts {tuple(res['kpts'].shape)}")
    assert n_kpt == nk, f"expected {nk} keypoint channels, got {n_kpt}"
    assert n_score == 1, f"expected nc=1 (gate) score channel, got {n_score}"
    print("PoseModel forward OK: (1,3,640,640) -> YOLO pose head (boxes/scores/kpts)")

    # GraftedPoseModel: the trainable graft (CP 2.4). One centered gate → loss → overfit a few
    # steps → loss must fall (proves the end-to-end loss/grad/optimizer path on CPU).
    grafted = build_grafted_pose_model(arch, supernet=sn).train()
    kpts = torch.rand(1, 8, 3)
    kpts[..., 2] = 1.0
    batch = {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        "keypoints": kpts,
    }
    opt = torch.optim.AdamW(grafted.parameters(), lr=1e-2)
    first = last = 0.0
    for i in range(15):
        loss, _ = grafted(batch)
        total = loss.sum()
        opt.zero_grad()
        total.backward()
        opt.step()
        last = float(total.detach())
        if i == 0:
            first = last
    print(f"GraftedPoseModel overfit OK: loss {first:.2f} -> {last:.2f} over 15 steps")
    assert last < first, "expected the 1-image overfit loss to decrease"
