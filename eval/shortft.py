"""CP 2.4 — short (~5-epoch) pose fine-tune harness + the two DoD gates.

The accuracy signal that drives the search. For a sampled OFA arch we graft it under a
YOLO11-pose head (:func:`detect.pose_model.build_grafted_pose_model`), fine-tune ~5 epochs on
the gate-pose target (D1 = ``dataset/``), and score **pose mAP (OKS)** with Ultralytics' own
validator (:mod:`detect.evaluate`). Two DoDs gate it:

- **reproducibility** (:func:`reproducible`) — the same arch twice within 0.5 mAP points. This
  is *precision*, not rank correctness.
- **proxy-rank fidelity** (:func:`rank_fidelity`) — the panel's #1 gate (peer-review R2.1):
  the 5-epoch proxy ranking of ~8-12 archs must agree with their full-train ranking at
  **Kendall-τ ≥ 0.7**. If it doesn't, BO would climb the wrong surface — repair the proxy
  (epochs / LR / resolution) *before* spending search compute.

Split by where it runs:

- The two DoD helpers are pure (scipy-only) → unit-tested under ``.venv`` / CI
  (``tests/test_shortft.py``).
- :func:`short_finetune` imports torch/ultralytics/ofa lazily and needs a GPU + the dataset, so
  it is integration-smoked, not unit-tested. The CPU loss/grad path of the grafted model is
  proven by ``tests/test_grafted_pose_model.py``; the real 5-epoch fine-tune + both DoDs are the
  GPU-gated run (Kaggle / Jetson). See CLAUDE.md "Known blockers".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# CP 2.4 DoD thresholds (PROJECT_PLAN.md CP 2.4).
KENDALL_TAU_GATE = 0.7
# "twice within 0.5 %": pose mAP is a [0, 1] fraction, so 0.5 % == 0.005 mAP points (absolute).
REPRODUCIBILITY_ABS_TOL = 0.005


@dataclass(frozen=True)
class RankFidelity:
    """Agreement between a proxy (5-epoch) ranking and the full-train ranking of the same archs."""

    kendall_tau: float
    spearman: float
    n: int

    @property
    def passes(self) -> bool:
        """CP 2.4 gate: the proxy is trustworthy enough to drive search iff τ ≥ 0.7."""
        return self.kendall_tau >= KENDALL_TAU_GATE


def rank_fidelity(
    proxy_scores: Sequence[float], full_scores: Sequence[float]
) -> RankFidelity:
    """Kendall-τ (the gate) + Spearman ρ between the proxy and full-train rankings.

    ``proxy_scores[i]`` and ``full_scores[i]`` are the 5-epoch and full-train pose mAPs of the
    same architecture ``i``. Only the *order* matters — absolute offset between the two regimes
    is irrelevant to the search.
    """
    if len(proxy_scores) != len(full_scores):
        raise ValueError(
            f"proxy_scores and full_scores must have the same length: "
            f"{len(proxy_scores)} != {len(full_scores)}"
        )
    if len(proxy_scores) < 2:
        raise ValueError("need at least 2 architectures to compute a ranking correlation")

    from scipy.stats import kendalltau, spearmanr  # lazy: keeps import light where unused

    tau = float(kendalltau(proxy_scores, full_scores).statistic)
    rho = float(spearmanr(proxy_scores, full_scores).statistic)
    return RankFidelity(kendall_tau=tau, spearman=rho, n=len(proxy_scores))


def reproducible(
    run_a: float, run_b: float, *, abs_tol: float = REPRODUCIBILITY_ABS_TOL
) -> bool:
    """True iff two pose-mAP readings of one arch agree within ``abs_tol`` (default 0.5 pts)."""
    return abs(run_a - run_b) <= abs_tol


# --------------------------------------------------------------------------------------------
# Integration: the actual short fine-tune. Heavy imports are lazy so importing this module (for
# the DoD helpers above) stays torch/ultralytics/ofa-free under .venv / CI. GPU-gated to run.
# --------------------------------------------------------------------------------------------

def short_finetune(
    arch_dict: dict,
    *,
    epochs: int = 5,
    seed: int = 0,
    lr: float = 1e-3,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    data_yaml: Any = None,
    supernet: Any = None,
    max_steps: int | None = None,
    head_weights: Any = None,
    freeze_head: bool = False,
) -> dict[str, float]:
    """Fine-tune a sampled OFA-backbone pose model ~``epochs`` epochs; return its pose mAP.

    Returns ``{'map', 'map50'}`` (OKS pose mAP), the CP 2.4 accuracy proxy. ``max_steps`` caps
    optimizer steps (CPU smoke / debugging). Deterministic given ``seed`` (the reproducibility
    DoD). CUDA + dataset dependent — see the module docstring.

    ``head_weights`` warm-starts the Pose head from a trained gate ``.pt`` and ``freeze_head``
    locks it (CP 2.4 proxy repair): with a frozen, competent head the optimizer trains only the
    backbone+adapter, so the score reflects backbone quality not head-init luck. Both default off
    (the original fresh-random-head proxy).
    """
    import torch

    from detect.evaluate import DEFAULT_DATA_YAML, pose_map_model
    from detect.pose_model import build_grafted_pose_model

    _seed_everything(seed)
    data_yaml = DEFAULT_DATA_YAML if data_yaml is None else data_yaml

    model = build_grafted_pose_model(
        arch_dict, supernet=supernet, head_weights=head_weights, freeze_head=freeze_head,
    ).to(device).train()
    loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="train")
    # Only trainable params: a frozen head must stay out of the optimizer (a no-op otherwise).
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    step = 0
    for _ in range(epochs):
        for raw in loader:
            batch_dict = _to_device(_preprocess_batch(raw), device)
            loss, _items = model(batch_dict)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    return pose_map_model(model, data_yaml=data_yaml, imgsz=imgsz, device=device)


def _seed_everything(seed: int) -> None:
    """Pin Python/NumPy/torch RNGs + deterministic cuDNN so the reproducibility DoD is real."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_pose_loader(data_yaml: Any, *, imgsz: int, batch: int, mode: str) -> Any:
    """Ultralytics pose dataloader over the gate dataset (GPU-gated; scaffolds the real run)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    from detect.evaluate import resolve_data_yaml

    spec = _load_data_dict(resolve_data_yaml(data_yaml))
    cfg = get_cfg(DEFAULT_CFG)
    cfg.task = "pose"  # else the dataset omits keypoints and v8PoseLoss has nothing to fit
    cfg.imgsz = imgsz
    split = "train" if mode == "train" else "val"
    dataset = build_yolo_dataset(cfg, spec[split], batch, spec, mode=mode, stride=32)
    return build_dataloader(dataset, batch=batch, workers=0, shuffle=(mode == "train"))


def _load_data_dict(data_yaml: Any) -> dict:
    from pathlib import Path

    import yaml

    spec = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(spec["path"])
    for k in ("train", "val"):
        if spec.get(k):
            spec[k] = str(base / spec[k])
    return spec


def _preprocess_batch(raw: dict) -> dict:
    """Normalize an Ultralytics dataloader batch (uint8 img → float/255)."""
    out = dict(raw)
    out["img"] = raw["img"].float() / 255.0
    return out


def _to_device(batch: dict, device: str) -> dict:
    import torch

    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


if __name__ == "__main__":  # integration smoke — run under .venv-nas (CPU ok; capped steps)
    from supernet.sampler import load_supernet, random_arch

    sn = load_supernet()
    arch = random_arch(sn)
    print(f"short_finetune smoke: arch d={arch['d']} (2 steps, CPU)")
    metrics = short_finetune(arch, epochs=1, batch=2, device="cpu", supernet=sn, max_steps=2)
    print(f"  pose mAP={metrics['map']:.4f}  mAP50={metrics['map50']:.4f}")
    print("short_finetune OK: graft → fine-tune step → pose-mAP eval ran end-to-end")
