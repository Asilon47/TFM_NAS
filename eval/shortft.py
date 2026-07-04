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


def precision_at_k(
    proxy_scores: Sequence[float], full_scores: Sequence[float], k: int
) -> float:
    """Fraction of the true top-``k`` architectures recovered in the proxy's top-``k``.

    The search-relevant success metric Kendall-τ misses (CP 2.4 Tier-1B): ``1.0`` means
    the ranker surfaced exactly the top-``k`` archs (order within the ``k`` ignored), so a
    BO loop seeded from the proxy's leaders would start on genuinely good architectures.
    ``proxy_scores[i]`` / ``full_scores[i]`` are scores of the same arch ``i``; higher is
    better. The CP 2.4 zero-cost descriptors hit precision@3 = 1.0 while their τ "failed".
    """
    if len(proxy_scores) != len(full_scores):
        raise ValueError(
            f"proxy_scores and full_scores must have the same length: "
            f"{len(proxy_scores)} != {len(full_scores)}"
        )
    n = len(full_scores)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    def topk(scores: Sequence[float]) -> set[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return set(order[:k])

    return len(topk(proxy_scores) & topk(full_scores)) / k


def top1_regret(proxy_scores: Sequence[float], full_scores: Sequence[float]) -> float:
    """Full-train mAP gap between the true-best arch and the one the proxy ranks #1.

    ``0.0`` iff the proxy's argmax is a true-best arch — the headline cost of trusting the
    ranker to pick a single winner. In CP 2.4 the zero-cost descriptors had regret 0.0
    (picked the true best) while the 5-epoch proxy had regret ~0.02.
    """
    if len(proxy_scores) != len(full_scores):
        raise ValueError(
            f"proxy_scores and full_scores must have the same length: "
            f"{len(proxy_scores)} != {len(full_scores)}"
        )
    if not proxy_scores:
        raise ValueError("need at least 1 architecture to compute top-1 regret")
    pick = max(range(len(proxy_scores)), key=lambda i: proxy_scores[i])
    return max(full_scores) - full_scores[pick]


# CP 2.4 search-relevant gate (the reframe — supersedes the τ-on-10 DoD, which the data showed
# mis-measures: size descriptors "fail" τ yet pick the true-best arch). Thresholds are proposed
# defaults, tunable per the user's risk appetite (D4-adjacent).
SPEARMAN_GATE = 0.70  # conventional "strong" monotonic correlation
TOP1_REGRET_TOL = 0.01  # the proxy's #1 pick must be within 1 mAP point of the true best


@dataclass(frozen=True)
class RankVerdict:
    """Search-relevant rank-fidelity verdict for a proxy/zero-cost ranker.

    Passes iff the ranker is both monotonically informative (Spearman ρ ≥ ``spearman_gate``)
    **and** its #1 pick is near-best (``top1_regret`` ≤ ``regret_tol``) — what a BO loop seeded
    from the ranker actually needs. Replaces the Kendall-τ-on-10 gate, whose CIs at n=10 are very
    wide and which punishes mid-rank disagreements the search ignores (CP 2.4: depth_sum/latency
    pass here, the 5-epoch proxy fails). ``precision_at_k`` and ``kendall_tau`` ride along as
    diagnostics; the stored thresholds keep ``passes`` self-contained.
    """

    spearman: float
    kendall_tau: float
    precision_at_k: float
    top1_regret: float
    k: int
    n: int
    spearman_gate: float
    regret_tol: float

    @property
    def passes(self) -> bool:
        return self.spearman >= self.spearman_gate and self.top1_regret <= self.regret_tol


def rank_verdict(
    proxy_scores: Sequence[float],
    full_scores: Sequence[float],
    *,
    k: int = 3,
    spearman_gate: float = SPEARMAN_GATE,
    regret_tol: float = TOP1_REGRET_TOL,
) -> RankVerdict:
    """Search-relevant verdict combining Spearman ρ + top-1 regret (+ precision@k / τ diagnostics).

    The CP 2.4 reframe's success criterion. ``proxy_scores[i]`` / ``full_scores[i]`` are the
    ranker and full-train scores of arch ``i``; higher is better. ``rank_fidelity`` is called
    first so a length mismatch raises before any metric is computed.
    """
    rf = rank_fidelity(proxy_scores, full_scores)  # validates lengths; gives τ + ρ
    return RankVerdict(
        spearman=rf.spearman,
        kendall_tau=rf.kendall_tau,
        precision_at_k=precision_at_k(proxy_scores, full_scores, k),
        top1_regret=top1_regret(proxy_scores, full_scores),
        k=k,
        n=rf.n,
        spearman_gate=spearman_gate,
        regret_tol=regret_tol,
    )


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
    save_to: Any = None,
) -> dict[str, float]:
    """Fine-tune a sampled OFA-backbone pose model ~``epochs`` epochs; return its pose mAP.

    Returns ``{'map', 'map50'}`` (OKS pose mAP), the CP 2.4 accuracy proxy. ``max_steps`` caps
    optimizer steps (CPU smoke / debugging). Deterministic given ``seed`` (the reproducibility
    DoD). CUDA + dataset dependent — see the module docstring.

    ``head_weights`` warm-starts the Pose head from a trained gate ``.pt`` and ``freeze_head``
    locks it (CP 2.4 proxy repair): with a frozen, competent head the optimizer trains only the
    backbone+adapter, so the score reflects backbone quality not head-init luck. Both default off
    (the original fresh-random-head proxy).

    ``save_to`` (CP 3.5 winner export) persists the fine-tuned model's ``state_dict`` to that path
    after training — the "weights" half of the winner-v1 artifact. The whole grafted model is saved
    (backbone + adapter + head) so a reload is self-contained: pair it with the arch to rebuild.
    No-op when ``None``.
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

    if save_to is not None:  # CP 3.5: persist the fine-tuned weights (the winner-v1 artifact)
        from pathlib import Path
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_to))

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
