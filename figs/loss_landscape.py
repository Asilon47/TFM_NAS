"""Filter-normalised loss landscape (Li et al., NeurIPS 2018) for a gate-pose checkpoint.

Sweeps a 2-D grid ``(alpha, beta)`` of *filter-normalised* random directions around a trained
weight vector ``theta*`` and evaluates the real validation loss at every grid point, writing a
3-column ``.dat`` surface (``alpha beta loss``) that a pgfplots ``\\addplot3[surf]`` renders.

Two models share one code path (they both expose Ultralytics' ``v8PoseLoss`` via
``model(batch_dict) -> (loss, items)``):

* ``winner``   — the searched OFA-backbone graft, rebuilt via
  :func:`detect.pose_model.build_grafted_pose_model` then ``load_state_dict`` from the trained
  ``models/graft/winner_v1_noneck.pt`` (the neck-less graft, mAP 0.841).
* ``baseline`` — the deployed ``yolo11n-pose`` (``runs/.../gate_baseline/weights/best.pt``),
  loaded through ``ultralytics.YOLO`` and unwrapped to its ``PoseModel``.

Why filter-normalise (the crux of Li et al.): BatchNorm makes the network scale-invariant, so a
naive random-direction walk mostly measures arbitrary per-filter rescaling, not curvature.
Rescaling each direction filter to ``||d_filter|| = ||theta_filter||`` removes that, and it is
what makes two *different* architectures' surfaces comparable at all. BN gamma/beta and biases are
left at ``theta*`` (their direction zeroed), per the paper.

Honesty gates (enforced, à la a data-generating script):
* ``loss(0, 0)`` must equal the checkpoint's real validation loss computed *before* the sweep
  (``--tol``, default 1e-3). This proves ``theta*`` is the true centre — it catches a wrong neck,
  a BN-mode slip, or an accumulation bug that would otherwise fake a plausible surface.
* No interpolation, no smoothing: every cell is a real forward-pass loss.
* No silent holes: a non-finite cell (large perturbation) is written verbatim (the LaTeX layer
  can clip); an OOM should be met by shrinking ``--batch``, never the grid.

Runs under ``.venv-nas`` (needs ``torch`` + ``ofa`` + ``ultralytics``). Auto-selects CUDA when
available, else CPU. See ``kaggle/run.py`` ``MODE="loss_landscape"`` for the remote-GPU driver.

    python -m figs.loss_landscape --model winner   --out figs/data/loss_landscape_winner.dat
    python -m figs.loss_landscape --model baseline --out figs/data/loss_landscape_baseline.dat
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WINNER_JSON = REPO_ROOT / "state" / "winner_v1" / "winner.json"
DEFAULT_WINNER_CKPT = REPO_ROOT / "models" / "graft" / "winner_v1_noneck.pt"
DEFAULT_BASELINE_CKPT = (
    REPO_ROOT / "runs" / "pose" / "experiments" / "gate_baseline" / "weights" / "best.pt")
DEFAULT_DATA_YAML = REPO_ROOT / "dataset" / "dataset.yaml"

# The v8PoseLoss gains both surfaces are pinned to, so their loss units match (comparability
# gate). These are the Ultralytics pose defaults (get_cfg(DEFAULT_CFG)); we set them explicitly on
# both models rather than trust each checkpoint's own train-args namespace.
LOSS_GAINS = ("box", "cls", "dfl", "pose", "kobj")


# --------------------------------------------------------------------------------------------
# Model construction — reuse the project's own builders; no new model code here.
# --------------------------------------------------------------------------------------------

def _pin_loss_gains(model: Any) -> dict[str, float]:
    """Set ``model.args.{box,cls,dfl,pose,kobj}`` to the pose defaults; drop any cached criterion.

    Ultralytics' ``v8PoseLoss`` reads its gains from ``model.args`` when the criterion is first
    built, so both models must carry the *same* gains for the two surfaces to share loss units.
    We overwrite the gains on a fresh default namespace and null ``model.criterion`` so it rebuilds
    against them on the next ``.loss()`` call (the ``prune/prune_baseline.py`` reset pattern).
    """
    from detect.pose_model import default_pose_args

    defaults = default_pose_args()
    model.args = defaults
    model.criterion = None  # force init_criterion() to re-read the pinned gains
    return {g: float(getattr(defaults, g)) for g in LOSS_GAINS}


def load_winner(ckpt: Path, winner_json: Path, device: str) -> Any:
    """Rebuild the neck-less winner graft and load its trained state_dict (strict)."""
    from detect.pose_model import build_grafted_pose_model
    from supernet.sampler import load_supernet

    arch = json.loads(Path(winner_json).read_text())["arch"]  # d=[2,2,4,3,3]
    supernet = load_supernet()
    model = build_grafted_pose_model(arch, supernet=supernet, neck=None)  # noneck = the 0.841 graft
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strict=True doubles as an honesty check that the rebuilt graph matches the trained weights.
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_baseline(ckpt: Path, device: str) -> Any:
    """Load the deployed yolo11n-pose and unwrap to its trainable ``PoseModel``."""
    from ultralytics import YOLO

    return YOLO(str(ckpt)).model.to(device)


# --------------------------------------------------------------------------------------------
# Fixed validation subset + loss.
# --------------------------------------------------------------------------------------------

def build_fixed_subset(data_yaml: Path, *, imgsz: int, batch: int, subset: int,
                       device: str) -> tuple[list[dict], int]:
    """Materialise a FIXED list of preprocessed, on-device val batches (same images every step).

    ``mode="val"`` gives ``shuffle=False`` so the image order is deterministic; ``subset`` caps to
    (at least) the first ``subset`` images by taking whole batches. ``subset<=0`` uses all of them.
    Returns ``(batches, n_images)`` — the real image count is stamped into the .dat header.
    """
    from eval.shortft import _build_pose_loader, _preprocess_batch, _to_device

    loader = _build_pose_loader(data_yaml, imgsz=imgsz, batch=batch, mode="val")
    batches: list[dict] = []
    n_images = 0
    for raw in loader:
        batches.append(_to_device(_preprocess_batch(raw), device))
        n_images += int(raw["img"].shape[0])
        if 0 < subset <= n_images:
            break
    if not batches:
        raise RuntimeError(f"no validation batches produced from {data_yaml}")
    return batches, n_images


@torch.no_grad()
def eval_loss(model: Any, batches: list[dict]) -> float:
    """Summed v8PoseLoss over the fixed subset, in eval mode (BN running stats, no autograd)."""
    was_training = model.training
    model.eval()
    total = 0.0
    for batch in batches:
        loss, _items = model(batch)  # dict -> BaseModel.loss -> v8PoseLoss (mode-agnostic)
        total += float(loss.sum())
    if was_training:
        model.train()
    return total


# --------------------------------------------------------------------------------------------
# Filter-normalised directions + the grid sweep.
# --------------------------------------------------------------------------------------------

def filter_normalised_directions(
    theta: dict[str, Tensor], seed: int
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Two Gaussian directions, each filter-normalised to ``||d_filter|| = ||theta_filter||``.

    Per-output-channel (dim-0) rescaling for conv/linear weights (ndim>=2); BN gamma/beta and
    biases (ndim<2) keep ``theta*`` (direction zeroed). Same ``seed`` for both models per Li et al.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand_dir() -> dict[str, Tensor]:
        # Generate on CPU (deterministic across devices) then move to each param's device/dtype.
        return {k: torch.randn(v.shape, generator=gen, dtype=torch.float32).to(v)
                for k, v in theta.items()}

    def normalise(direction: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, dv in direction.items():
            tv = theta[key]
            if dv.ndim >= 2:  # conv/linear weight -> per-output-channel filter
                for i in range(dv.shape[0]):
                    dv[i].mul_(tv[i].norm() / (dv[i].norm() + 1e-12))
            else:             # BN gamma/beta, biases -> stay at theta*
                dv.zero_()
        return direction

    return normalise(rand_dir()), normalise(rand_dir())


def sweep(model: Any, batches: list[dict], grid: int, seed: int) -> tuple[list[str], float]:
    """Row-major (alpha outer, beta inner) sweep of alpha,beta in [-1,1]; return (rows, loss@0)."""
    theta = {k: v.detach().clone() for k, v in model.state_dict().items()}
    d1, d2 = filter_normalised_directions(theta, seed)
    axis = torch.linspace(-1.0, 1.0, grid)

    rows: list[str] = []
    base: float | None = None
    for ai, a in enumerate(axis.tolist()):
        for b in axis.tolist():
            perturbed = {k: theta[k] + a * d1[k] + b * d2[k] for k in theta}
            model.load_state_dict(perturbed, strict=True)  # restore-by-replace: no accumulation
            loss_val = eval_loss(model, batches)
            if abs(a) < 1e-9 and abs(b) < 1e-9:
                base = loss_val
            rows.append(f"{a:.4f} {b:.4f} {loss_val:.4f}")
        print(f"  row {ai + 1}/{grid} (alpha={a:+.4f}) done", flush=True)

    model.load_state_dict(theta, strict=True)  # restore theta* for good measure
    if base is None:  # odd grids include 0; guard against an even grid skipping the centre cell
        raise RuntimeError(f"grid={grid} excludes (0,0) — use an odd grid so theta* is a cell")
    return rows, base


def write_dat(out: Path, rows: list[str], *, tag: str, ckpt: Path, grid: int, seed: int,
              n_images: int, gains: dict[str, float], base: float) -> None:
    """Write the 3-line-header + row-major surface .dat (alpha outer, beta inner)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    gains_str = " ".join(f"{k}={v:g}" for k, v in gains.items())
    header = (
        f"# loss_landscape_{tag}.dat: filter-normalised val loss surface ({tag})\n"
        f"# provenance: generated {date.today().isoformat()} by figs/loss_landscape.py; "
        f"ckpt {ckpt}; grid {grid}x{grid} alpha,beta in [-1,1]; dirs seed={seed}; "
        f"loss=v8PoseLoss sum(box+cls+dfl+pose+kobj) gains[{gains_str}]; "
        f"val subset={n_images} imgs (dataset/images/val, fixed); loss(0,0)={base:.4f}\n"
        "# alpha beta loss\n"
    )
    out.write_text(header + "\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("winner", "baseline"), required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: per-model)")
    ap.add_argument("--winner-json", default=str(DEFAULT_WINNER_JSON))
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    ap.add_argument("--out", default=None,
                    help="output .dat (default: figs/data/loss_landscape_<model>.dat)")
    ap.add_argument("--grid", type=int, default=41, help="grid points per axis (odd, incl. 0)")
    ap.add_argument("--subset", type=int, default=0, help="cap val images (<=0 = all 140)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3, help="honesty gate: |loss(0,0) - ref| bound")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else (DEFAULT_WINNER_CKPT if args.model == "winner"
                                              else DEFAULT_BASELINE_CKPT)
    out = (Path(args.out) if args.out
           else REPO_ROOT / "figs" / "data" / f"loss_landscape_{args.model}.dat")

    print(f"[loss-landscape] model={args.model} ckpt={ckpt} device={device} "
          f"grid={args.grid} subset={args.subset or 'all'} batch={args.batch}", flush=True)

    if args.model == "winner":
        model = load_winner(ckpt, Path(args.winner_json), device)
    else:
        model = load_baseline(ckpt, device)
    gains = _pin_loss_gains(model)

    batches, n_images = build_fixed_subset(
        Path(args.data), imgsz=args.imgsz, batch=args.batch, subset=args.subset, device=device)
    print(f"[loss-landscape] fixed subset: {n_images} imgs, {len(batches)} batch(es)", flush=True)

    ref = eval_loss(model, batches)  # theta* reference BEFORE the sweep
    print(f"[loss-landscape] reference loss(theta*) = {ref:.4f}", flush=True)

    rows, base = sweep(model, batches, args.grid, args.seed)

    # Honesty gate: the (0,0) cell must reproduce the real val loss to within tol.
    print(f"[loss-landscape] loss(0,0)={base:.4f}  ref={ref:.4f}  |d|={abs(base - ref):.2e}",
          flush=True)
    if not math.isclose(base, ref, abs_tol=args.tol):
        raise SystemExit(
            f"HONESTY GATE FAILED: loss(0,0)={base:.6f} != ref={ref:.6f} "
            f"(|delta|={abs(base - ref):.2e} > tol={args.tol}). theta* is not the true centre — "
            "aborting rather than writing a fake surface.")

    write_dat(out, rows, tag=args.model, ckpt=ckpt, grid=args.grid, seed=args.seed,
              n_images=n_images, gains=gains, base=base)
    print(f"[loss-landscape] wrote {out} ({len(rows)} rows, {args.grid}x{args.grid})", flush=True)


if __name__ == "__main__":
    main()
