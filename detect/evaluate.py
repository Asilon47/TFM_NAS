"""Pose-mAP (OKS) evaluation, reusing Ultralytics' validator instead of re-implementing OKS.

CP 2.4's accuracy signal becomes pose mAP, not top-1. Ultralytics already computes it: a YOLO
pose model's ``.val(data=...)`` returns ``metrics.pose.map`` (mAP50-95) and ``.map50``, on the
exact OKS definition the deployed model is scored with — so we reuse it.

Two wrinkles this module owns:

1. ``dataset.yaml`` ships a stale absolute ``path:`` (``/root/workspace/...``) from the box it
   was authored on. Ultralytics resolves ``train``/``val`` relative to ``path``, so
   :func:`resolve_data_yaml` rewrites it to the local ``dataset/`` into a temp copy — the user's
   file is never touched.
2. ``pose_map`` evaluates an Ultralytics YOLO pose model directly (the immediate use: anchor the
   **baseline** yolo11n-pose's mAP + the teacher's). Scoring a *grafted* backbone
   (:mod:`detect.pose_model`) means wrapping it as an Ultralytics model (its ``.args``/``.names``/
   loss + a short fine-tune so the fresh head is meaningful) — the GPU-gated follow-up.

GPU-gated: a live run needs ``.venv-nas`` + the dataset. ``ultralytics``/``yaml`` import lazily,
so ``import detect.evaluate`` stays dependency-free under ``.venv``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = REPO_ROOT / "dataset" / "dataset.yaml"


def resolve_data_yaml(
    data_yaml: Path = DEFAULT_DATA_YAML, dataset_root: Path | None = None
) -> Path:
    """Copy ``data_yaml`` with its ``path:`` rewritten to the local dataset dir; return the copy.

    ``dataset_root`` defaults to the yaml's own directory (i.e. ``dataset/``). The original file
    is left unmodified — Ultralytics reads the returned temp copy.
    """
    import yaml

    spec = yaml.safe_load(Path(data_yaml).read_text())
    spec["path"] = str(dataset_root if dataset_root is not None else Path(data_yaml).parent)
    fd, tmp = tempfile.mkstemp(suffix="_dataset.yaml", text=True)
    out = Path(tmp)
    with open(fd, "w") as fh:
        yaml.safe_dump(spec, fh)
    return out


def pose_map(
    model: Any,
    *,
    data_yaml: Path = DEFAULT_DATA_YAML,
    imgsz: int = 640,
    device: str = "cpu",
    **val_kwargs: Any,
) -> dict[str, float]:
    """Validate an Ultralytics YOLO pose model; return ``{'map', 'map50'}`` (pose/OKS mAP).

    ``model`` is an Ultralytics ``YOLO`` instance or a path/str to pose weights (e.g.
    ``yolo11n-pose.pt``). Reuses Ultralytics' pose validator so the metric matches the deployed
    model's.
    """
    from ultralytics import YOLO

    yolo = model if hasattr(model, "val") else YOLO(model)
    metrics = yolo.val(data=str(resolve_data_yaml(data_yaml)), imgsz=imgsz, device=device,
                       **val_kwargs)
    return {"map": float(metrics.pose.map), "map50": float(metrics.pose.map50)}


def pose_map_model(
    model: Any,
    *,
    data_yaml: Path = DEFAULT_DATA_YAML,
    imgsz: int = 640,
    device: str = "cpu",
    batch: int = 16,
    **val_kwargs: Any,
) -> dict[str, float]:
    """Pose mAP (OKS) for a *grafted* model — the CP 2.4 fine-tune's accuracy proxy.

    :func:`pose_map` drives the high-level ``YOLO`` wrapper (baseline / teacher anchoring);
    a :class:`detect.pose_model.GraftedPoseModel` has no ``.val`` of its own, so we run
    Ultralytics' ``PoseValidator`` directly against it. The validator wraps the model in
    ``AutoBackend`` (reads ``.stride``/``.names``/``.kpt_shape``, which the graft exposes), builds
    the pose dataloader from the (path-rewritten) ``data_yaml``, and computes the same OKS mAP.

    Returns ``{'map', 'map50'}``. GPU/dataset-gated to run; ``ultralytics`` imports lazily.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.utils import DEFAULT_CFG

    args = get_cfg(DEFAULT_CFG)
    args.data = str(resolve_data_yaml(data_yaml))
    args.imgsz, args.batch, args.device = imgsz, batch, device
    args.task, args.mode = "pose", "val"
    args.save_json = args.plots = args.verbose = False
    for key, value in val_kwargs.items():
        setattr(args, key, value)

    was_training = model.training
    validator = PoseValidator(args=args)
    validator(model=model.eval())
    if was_training:
        model.train()
    metrics = validator.metrics
    return {"map": float(metrics.pose.map), "map50": float(metrics.pose.map50)}
