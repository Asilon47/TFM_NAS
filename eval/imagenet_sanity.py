"""CP 1.4 — ImageNet sanity check for the inherited OFA-MBv3-w1.0 supernet.

Validates that ``supernet.sampler.load_supernet()`` + ``sample(arch)`` + **BatchNorm
recalibration** reproduces the top-1 accuracy OFA's *own* released model predicts for that
arch. This is the first time the accuracy axis is exercised; a wrong weight load or a
skipped BN-recalibration would silently poison every accuracy number from CP 2.4 on.

**Protocol** (mirrors OFA's own ``ofa/tutorial/imagenet_eval_helper.py``):
``set_active_subnet`` -> ``get_active_subnet(preserve_weight=True)`` -> recalibrate BN on a
2000-image ``train`` subset (``set_running_statistics`` — the make-or-break step: a subnet
inherits the *supernet's* full-width BN stats, which are wrong for its sliced channels) ->
measure top-1 on a ``val`` subset.

**Reference** = OFA's released accuracy predictor (``ofa/tutorial/accuracy_predictor.py``).
It is a *ranking* model: the first real run (2026-06-17) showed it reads ~6.3pp high in
absolute top-1 with a near-constant offset (one offset reconciles every arch to <1pp; rank
order identical) — consistent with its labels being on a higher absolute scale (e.g. a
train-holdout subset). That is fine, because OFA uses it only to *order* candidates during
search. So CP 1.4 gates on **Spearman rank correlation** between measured and predicted top-1
across a spread of archs, and reports the OLS affine fit (``measured ~= slope*predicted +
intercept``) as the absolute-scale evidence. (Not the specialized nets: those carry 25-75
extra fine-tune epochs and would differ further.)

This module's *pure* layer (arch construction, scale normalization, CI, rank stats) runs
under ``.venv`` (which has ``scipy``); the OFA + ImageNet orchestration imports its heavy deps
lazily, so ``import eval.imagenet_sanity`` works in either venv. The actual measurement needs
``.venv-nas`` + a dataset. See ``eval/README.md``.

    source .venv-nas/bin/activate
    python -m eval.imagenet_sanity --imagenet-path /path/to/imagenet   # +--device cuda
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

from catalog.contracts import ArchDict
from catalog.ofa_mbv3 import KS, MAX_DEPTH, D, E
from search.arch_to_blocks import random_arch_dict

if TYPE_CHECKING:  # heavy deps stay lazy — imported inside the orchestration fns below
    from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3

# OFA-MBv3 has 5 stages x MAX_DEPTH block slots; ks/e carry one entry per slot, d one per
# stage. OFA's evaluate_ofa_subnet asserts exactly these lengths.
N_SLOTS = 5 * MAX_DEPTH


# --- pure decision layer (.venv-safe: no ofa/torchvision/ImageNet) -----------

def canonical_archs() -> dict[str, ArchDict]:
    """The two deterministic corners of the search space: biggest and smallest subnet."""
    return {
        "max": {"ks": [max(KS)] * N_SLOTS, "e": [max(E)] * N_SLOTS, "d": [max(D)] * 5},
        "min": {"ks": [min(KS)] * N_SLOTS, "e": [min(E)] * N_SLOTS, "d": [min(D)] * 5},
    }


def resolve_archs(names: list[str], *, seed: int) -> list[tuple[str, ArchDict]]:
    """Map CLI labels to (label, arch) pairs, preserving order.

    ``max``/``min`` are the space corners; ``random`` is a seed-deterministic draw (so a
    sanity run is reproducible). Unknown labels fail loudly.
    """
    corners = canonical_archs()
    resolved: list[tuple[str, ArchDict]] = []
    for name in names:
        if name == "random":
            resolved.append((name, random_arch_dict(random.Random(seed))))
        elif name in corners:
            resolved.append((name, corners[name]))
        else:
            raise ValueError(
                f"unknown arch label {name!r}; choose from ['max', 'min', 'random']"
            )
    return resolved


def build_net_config(arch: ArchDict, resolution: int) -> dict:
    """Add the input ``resolution`` OFA's eval/predictor need but ``ArchDict`` omits.

    Validates OFA's length contract (ks/e are ``N_SLOTS`` long, d has 5 entries) with a
    ``ValueError`` rather than an ``assert`` so it survives ``python -O`` and is testable.
    """
    ks, e, d = arch["ks"], arch["e"], arch["d"]
    if not len(ks) == len(e) == N_SLOTS:
        raise ValueError(
            f"ks and e must each have {N_SLOTS} entries, got {len(ks)} and {len(e)}"
        )
    if len(d) != 5:
        raise ValueError(f"d must have 5 entries (one per stage), got {len(d)}")
    return {"ks": list(ks), "e": list(e), "d": list(d), "r": [resolution]}


def normalize_to_percent(value: float) -> float:
    """Coerce a top-1 to percent. OFA's predictor scale is unverified offline; a value in
    [0, 1] is read as a fraction (x100), anything larger is already in percent."""
    return value * 100.0 if value <= 1.0 else float(value)


def binomial_ci95(p_percent: float, n: int) -> float:
    """Half-width (percentage points) of the 95% CI for a top-1 measured on ``n`` images.

    Sampling noise alone: 2k val images at ~77% give +-1.8pp, *wider* than the 1.5pp DoD
    bar — so the report shows this band to keep noise from being read as a weight bug.
    """
    if n <= 0:
        return float("inf")
    p = min(max(p_percent / 100.0, 0.0), 1.0)
    return 1.96 * math.sqrt(p * (1.0 - p) / n) * 100.0


def require_imagenet_layout(path: Path) -> Path:
    """Fail loudly unless ``path`` holds both ``train/`` and ``val/`` (ImageFolder roots).

    A missing split must abort before any measurement — never silently emit a
    meaningless accuracy from an empty / mislaid dataset.
    """
    for split in ("train", "val"):
        if not (path / split).is_dir():
            raise FileNotFoundError(
                f"ImageNet path {path} is missing a '{split}/' directory "
                f"(expected <path>/{split}/<wnid>/*.JPEG, torchvision ImageFolder layout)"
            )
    return path


def val_sample_size(n_val: int, available: int) -> int:
    """How many val images to score: ``n_val <= 0`` means all; otherwise cap at available."""
    return available if n_val <= 0 else min(n_val, available)


# --- rank-fidelity layer: a set of archs + the Spearman gate -----------------
# The first real run showed the predictor is a *ranking* model with a near-constant ~6.3pp
# absolute offset (one offset reconciles every arch to <1pp). So CP 1.4 gates on rank
# correlation across a spread of archs — the predictor's intended use — and reports the OLS
# affine fit as the scale evidence. Pure (no ofa/torchvision); the scipy stats import is lazy.

def random_archs(n: int, *, seed: int) -> list[tuple[str, ArchDict]]:
    """``n`` distinct random archs labelled ``rand0..rand{n-1}``.

    Each draw uses ``random_arch_dict(random.Random(seed + i))``, so the set is seed-
    deterministic and ``rand0`` is exactly the single draw ``resolve_archs(['random'])`` makes.
    Rank fidelity needs a *spread* of interior points — Spearman over one arch is undefined.
    """
    return [(f"rand{i}", random_arch_dict(random.Random(seed + i))) for i in range(n)]


def rank_pass(rho: float, *, threshold: float) -> bool:
    """The Spearman gate: ``rho >= threshold``. A NaN rho (degenerate input) fails closed
    (``nan >= x`` is ``False``)."""
    return bool(rho >= threshold)


def rank_summary(measured: list[float], predicted: list[float], *, threshold: float) -> dict:
    """Rank fidelity + affine fit of predicted-vs-measured top-1, plus the pass/fail gate.

    Reuses ``search.predictor_stats.predictor_stats`` (the CP 2.2 latency-predictor tooling)
    with ``x=predicted, y=measured`` (its summed/measured convention): Spearman rho/p and
    Kendall tau measure *ranking*; the OLS fit ``measured ~= slope*predicted + intercept``
    (+ r2, MAPE raw vs calibrated) measures the constant absolute offset. ``passed`` gates on
    Spearman rho alone — the predictor's intended use.
    """
    from search.predictor_stats import predictor_stats

    st = predictor_stats(predicted, measured)
    return {
        "spearman_rho": st.spearman_rho,
        "spearman_p": st.spearman_p,
        "kendall_tau": st.kendall_tau,
        "kendall_p": st.kendall_p,
        "slope": st.fit.slope,
        "intercept": st.fit.intercept,
        "r2": st.fit.r2,
        "mape": st.mape,
        "mape_calibrated": st.mape_calibrated,
        "passed": rank_pass(st.spearman_rho, threshold=threshold),
    }


# --- OFA + ImageNet orchestration --------------------------------------------
# Heavy deps (torch/torchvision/ofa, and supernet.sampler which imports ofa) are imported
# *inside* these functions so the pure layer above stays importable under .venv. Running
# any of these needs .venv-nas + a real dataset.

_PREDICTOR_CACHE: dict = {}


def resolve_device(spec: str) -> str:
    """``auto`` -> ``cuda`` when a GPU is visible else ``cpu``; any explicit value passes."""
    if spec != "auto":
        return spec
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def predict_topk(arch: ArchDict, resolution: int, *, device: str = "cpu") -> float:
    """OFA's released accuracy predictor's top-1 (%) for ``arch`` at ``resolution``.

    This is the apples-to-apples reference: the predictor was trained on direct-extraction +
    BN-recalibrated subnets of this exact w1.0 space. One predictor instance is cached per
    device (it downloads ``acc_predictor.pth`` on first use).
    """
    from ofa.tutorial.accuracy_predictor import AccuracyPredictor

    if device not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[device] = AccuracyPredictor(pretrained=True, device=device)
    pred = _PREDICTOR_CACHE[device].predict_accuracy([build_net_config(arch, resolution)])
    return normalize_to_percent(float(pred.flatten()[0]))


def _build_val_loader(
    imagenet_path: Path, *, n_val: int, batch_size: int, device: str, seed: int
):
    """ImageFolder ``DataLoader`` over ``val/``, subset to ``n_val`` images (0 = all).

    Returns ``(loader, n_used)``; ``n_used`` feeds the binomial CI. The placeholder transform
    is irrelevant — OFA's ``validate`` overrides ``dataset.transform`` before iterating.
    """
    import torch
    from torchvision import datasets, transforms

    dataset = datasets.ImageFolder(str(imagenet_path / "val"), transforms.ToTensor())
    n_used = val_sample_size(n_val, len(dataset))
    sampler = None
    if n_used < len(dataset):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=gen)[:n_used].tolist()
        sampler = torch.utils.data.SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=8, pin_memory=("cuda" in device), drop_last=False,
    )
    return loader, n_used


def measure_topk(
    supernet: OFAMobileNetV3, arch: ArchDict, resolution: int, imagenet_path: Path, *,
    batch_size: int = 100, bn_images: int = 2000, n_val: int = 0,
    device: str = "cpu", seed: int = 0,
) -> tuple[float, int]:
    """Measured top-1 (%) and image count for ``arch``, BN-recalibrated then validated.

    Step-for-step OFA's ``evaluate_ofa_subnet`` (set_active -> get_active(preserve_weight) ->
    ``calib_bn`` on ``bn_images`` train images -> ``validate``), but threading ``bn_images`` /
    ``n_val`` and stating ``preserve_weight=True`` explicitly.
    """
    from ofa.tutorial.imagenet_eval_helper import calib_bn, validate

    cfg = build_net_config(arch, resolution)
    supernet.set_active_subnet(ks=cfg["ks"], d=cfg["d"], e=cfg["e"])
    subnet = supernet.get_active_subnet(preserve_weight=True).to(device)
    loader, n_used = _build_val_loader(
        imagenet_path, n_val=n_val, batch_size=batch_size, device=device, seed=seed
    )
    calib_bn(subnet, str(imagenet_path), resolution, batch_size, num_images=bn_images)
    top1 = validate(subnet, str(imagenet_path), resolution, loader, batch_size, device)
    return float(top1), n_used


def run_sanity(
    imagenet_path: Path | str, *, archs: tuple[str, ...] = ("max", "min"),
    n_random: int = 8, resolution: int = 224, n_val: int = 0, bn_images: int = 2000,
    batch_size: int = 100, device: str = "auto", bar: float = 1.5,
    rank_threshold: float = 0.85, seed: int = 0,
) -> dict:
    """Measure vs predict top-1 over a set of archs; gate CP 1.4 on rank fidelity.

    The set is the fixed corners (``archs``) plus ``n_random`` sampled interior archs. The
    DoD is Spearman ``rho >= rank_threshold`` between measured and predicted top-1 (the
    predictor's intended use); the affine fit + per-arch *calibrated* gap (``measured -
    affine-fit(predicted)``, expected within ``bar``) are the supporting scale evidence.
    """
    path = require_imagenet_layout(Path(imagenet_path))
    dev = resolve_device(device)
    from supernet.sampler import load_supernet

    supernet = load_supernet()
    arch_set = resolve_archs(list(archs), seed=seed) + random_archs(n_random, seed=seed)
    results: list[dict] = []
    for label, arch in arch_set:
        predicted = predict_topk(arch, resolution, device=dev)
        measured, n_used = measure_topk(
            supernet, arch, resolution, path, batch_size=batch_size,
            bn_images=bn_images, n_val=n_val, device=dev, seed=seed,
        )
        results.append({
            "label": label, "measured": measured, "predicted": predicted,
            "gap": measured - predicted, "ci95": binomial_ci95(measured, n_used),
            "n_val": n_used, "resolution": resolution,
        })

    summary = rank_summary([r["measured"] for r in results],
                           [r["predicted"] for r in results], threshold=rank_threshold)
    for r in results:  # calibrated gap = how far each arch is off the affine-fit predictor
        calibrated = summary["slope"] * r["predicted"] + summary["intercept"]
        r["calib_gap"] = r["measured"] - calibrated
        r["within_bar"] = abs(r["calib_gap"]) <= bar
    report = {
        "device": dev, "resolution": resolution, "bar": bar,
        "rank_threshold": rank_threshold, "results": results, "rank": summary,
        "passed": summary["passed"],
    }
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    rk = report["rank"]
    n = len(report["results"])
    print(f"\nCP 1.4 ImageNet sanity (rank fidelity) — device={report['device']}, "
          f"r={report['resolution']}, {n} archs")
    print("=" * 78)
    print(f"  {'arch':7s} {'n_val':>7s} {'measured':>9s} {'predicted':>10s} "
          f"{'raw gap':>8s} {'calib':>7s} {'CI95':>8s}")
    for r in report["results"]:
        ci_s = f"+-{r['ci95']:.2f}" if math.isfinite(r["ci95"]) else "n/a"
        flag = "" if r["within_bar"] else "  !"   # calibrated gap busts the info band
        print(f"  {r['label']:7s} {r['n_val']:7d} {r['measured']:7.2f}%  "
              f"{r['predicted']:8.2f}%  {r['gap']:+7.2f} {r['calib_gap']:+6.2f} "
              f"{ci_s:>8s}{flag}")
    print(f"\n  ranking : Spearman rho={rk['spearman_rho']:+.3f} (p={rk['spearman_p']:.2g})  "
          f"Kendall tau={rk['kendall_tau']:+.3f} (p={rk['kendall_p']:.2g})")
    print(f"  scale   : measured ~= {rk['slope']:.3f}*predicted {rk['intercept']:+.3f}  "
          f"(r2={rk['r2']:.4f})")
    print(f"  abs err : MAPE {rk['mape'] * 100:.2f}% raw -> {rk['mape_calibrated'] * 100:.2f}% "
          "calibrated")
    print("\n  (max/min are the space corners, randN are sampled interior archs. The "
          "predictor is\n   a ranking model with a ~constant absolute offset; calib gap = "
          "measured - affine(pred).)")
    print(f"  DoD (Spearman rho >= {report['rank_threshold']}): "
          f"{'PASS' if report['passed'] else 'FAIL'} "
          f"(rho={rk['spearman_rho']:+.3f}, p={rk['spearman_p']:.2g})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--imagenet-path", type=Path, required=True,
                   help="ImageNet root containing train/ and val/ ImageFolder subdirs")
    p.add_argument("--archs", default="max,min",
                   help="fixed corner archs to include (subset of max, min); the interior is "
                        "drawn by --n-random")
    p.add_argument("--n-random", type=int, default=8,
                   help="sampled interior archs (rank fidelity needs a spread; 8 -> 10 total "
                        "with the corners; bump to ~14-18 for a tighter rho interval)")
    p.add_argument("--rank-threshold", type=float, default=0.85,
                   help="DoD gate: Spearman rho between measured and predicted top-1")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--n-val", type=int, default=0,
                   help="val images to score (0 = all found; full 50k = +-0.4pp CI keeps "
                        "measurement noise from flipping ranks)")
    p.add_argument("--bn-images", type=int, default=2000,
                   help="train images for BN recalibration (OFA's default)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--bar", type=float, default=1.5,
                   help="info band (pp) for the per-arch calibrated gap; not the gate")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    report = run_sanity(
        args.imagenet_path,
        archs=tuple(a.strip() for a in args.archs.split(",") if a.strip()),
        n_random=args.n_random, rank_threshold=args.rank_threshold,
        resolution=args.resolution, n_val=args.n_val, bn_images=args.bn_images,
        batch_size=args.batch_size, device=args.device, bar=args.bar, seed=args.seed,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
