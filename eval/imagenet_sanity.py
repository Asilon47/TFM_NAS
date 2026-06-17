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

**Reference** = OFA's released accuracy predictor (``ofa/tutorial/accuracy_predictor.py``),
trained on direct-extraction + BN-recalibrated subnets of *this exact* w1.0 space — so
predicted-vs-measured is apples-to-apples. (Not the specialized nets: those carry 25-75
extra fine-tune epochs and would legitimately differ by several points.)

This module's *pure* layer (arch construction, scale normalization, CI, verdict) runs under
``.venv`` with no ``ofa``/``torchvision``/ImageNet. The OFA + ImageNet orchestration imports
its heavy deps lazily, so ``import eval.imagenet_sanity`` works in either venv; the actual
measurement needs ``.venv-nas`` + a dataset. See ``eval/README.md``.

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


def verdict(
    measured_pct: float, predicted_pct: float, *, bar: float = 1.5,
    ci: float | None = None,
) -> dict:
    """Compare a measured top-1 to the predictor's, against the DoD bar (and noise band).

    ``within_bar`` is the literal DoD (|gap| <= bar). ``within_noise`` (when a measurement
    CI is supplied) is the honest statistical read: a gap that busts the bar but sits inside
    the sampling CI is consistent, not a defect.
    """
    gap = measured_pct - predicted_pct
    abs_gap = abs(gap)
    return {
        "measured": measured_pct,
        "predicted": predicted_pct,
        "gap": gap,
        "abs_gap": abs_gap,
        "bar": bar,
        "within_bar": abs_gap <= bar,
        "ci95": ci,
        "within_noise": (abs_gap <= ci) if ci is not None else None,
    }


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


def is_diagnostic(label: str) -> bool:
    """The ``max``/``min`` corners are diagnostic, not gating.

    The accuracy predictor is trained on interior (sampled) archs and extrapolates
    *optimistically* at the space corners, so a corner exceeding the bar reflects predictor
    error, not a weight-loading defect. The DoD speaks of "a sampled subnet" — an interior
    point — so the corners inform ranking/range but do not decide pass/fail.
    """
    return label in ("max", "min")


def overall_pass(results: list[dict]) -> bool:
    """DoD verdict: every *interior* arch within the bar (corners excluded when present)."""
    gated = [r for r in results if not r["diagnostic"]] or results
    return all(r["within_bar"] for r in gated)


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
    imagenet_path: Path | str, *, archs: tuple[str, ...] = ("max", "min", "random"),
    resolution: int = 224, n_val: int = 0, bn_images: int = 2000, batch_size: int = 100,
    device: str = "auto", bar: float = 1.5, seed: int = 0,
) -> dict:
    """Measure vs predict top-1 for each arch and assemble the CP 1.4 report dict."""
    path = require_imagenet_layout(Path(imagenet_path))
    dev = resolve_device(device)
    from supernet.sampler import load_supernet

    supernet = load_supernet()
    results = []
    for label, arch in resolve_archs(list(archs), seed=seed):
        predicted = predict_topk(arch, resolution, device=dev)
        measured, n_used = measure_topk(
            supernet, arch, resolution, path, batch_size=batch_size,
            bn_images=bn_images, n_val=n_val, device=dev, seed=seed,
        )
        row = verdict(measured, predicted, bar=bar, ci=binomial_ci95(measured, n_used))
        row.update(label=label, n_val=n_used, resolution=resolution,
                   diagnostic=is_diagnostic(label))
        results.append(row)
    report = {
        "device": dev, "resolution": resolution, "bar": bar, "results": results,
        "passed": overall_pass(results),
    }
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print(f"\nCP 1.4 ImageNet sanity — device={report['device']}, "
          f"r={report['resolution']}, bar=+-{report['bar']}pp")
    print("=" * 76)
    print(f"  {'arch':7s} {'n_val':>7s} {'measured':>9s} {'predicted':>10s} "
          f"{'gap':>7s} {'CI95':>8s}  verdict")
    for r in report["results"]:
        tag = "PASS" if r["within_bar"] else ("noise" if r["within_noise"] else "FAIL")
        if r["diagnostic"]:
            tag += " (diag)"     # corner: predictor extrapolates here; not gated
        ci_s = f"+-{r['ci95']:.2f}" if math.isfinite(r["ci95"]) else "n/a"
        print(f"  {r['label']:7s} {r['n_val']:7d} {r['measured']:7.2f}%  "
              f"{r['predicted']:8.2f}%  {r['gap']:+6.2f} {ci_s:>8s}  {tag}")
    results = report["results"]
    if len(results) >= 4:
        from search.cost_preview import spearman

        rho = spearman([r["measured"] for r in results],
                       [r["predicted"] for r in results])
        print(f"\n  measured~predicted Spearman rho = {rho:+.3f} "
              f"(rank fidelity across {len(results)} archs)")
    print("\n  (max/min are diagnostic — the predictor extrapolates at the space corners; "
          "the\n   DoD gates on the interior 'sampled' arch(s).)")
    print(f"  DoD (interior arch within +-{report['bar']}pp of predictor): "
          f"{'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--imagenet-path", type=Path, required=True,
                   help="ImageNet root containing train/ and val/ ImageFolder subdirs")
    p.add_argument("--archs", default="max,min,random",
                   help="comma-separated subset of: max, min, random")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--n-val", type=int, default=0,
                   help="val images to score (0 = all found; >=10000 recommended vs the "
                        "1.5pp bar — 2k alone is +-1.8pp noise)")
    p.add_argument("--bn-images", type=int, default=2000,
                   help="train images for BN recalibration (OFA's default)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--bar", type=float, default=1.5, help="DoD top-1 bar in pp")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    report = run_sanity(
        args.imagenet_path,
        archs=tuple(a.strip() for a in args.archs.split(",")),
        resolution=args.resolution, n_val=args.n_val, bn_images=args.bn_images,
        batch_size=args.batch_size, device=args.device, bar=args.bar, seed=args.seed,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
