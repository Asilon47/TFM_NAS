"""CP 3.5 — winner-v1 verification: reload α* and reproduce its cached proxy mAP.

The DoD that closes Phase 3 (``PROJECT_PLAN.md:248``): *"Can reload in a clean Python
session and reproduce cached accuracy within noise."* :mod:`search.select_winner` already
serialized α* to ``state/winner_v1/winner.json`` (arch + provenance, no weights). This module
supplies the other two-thirds of the deliverable:

1. **Reload from arch alone.** :func:`load_winner` reads the record; the driver rebuilds the
   grafted pose model straight from ``winner["arch"]`` (via :func:`eval.shortft.short_finetune`
   → :func:`detect.pose_model.build_grafted_pose_model`). Nothing is unpickled — proving the
   *architecture* is a sufficient description of the winner is the point of the checkpoint.
2. **Reproduce the accuracy + persist weights.** A fresh warm-head fine-tune (same CP 2.4
   protocol: frozen gate-head donor, 5 epochs) re-derives the proxy mAP in a clean session.
   Per ``PROJECT_PLAN.md:598`` the winner check **averages 3 seeds** — a single stochastic
   fine-tune is too noisy to certify reproduction. The first seed's ``state_dict`` is saved to
   ``weights.pt`` (the winner-v1 weights); Phase 8 distils the *deployable* weights from there.

**Split by where it runs.** The verdict logic (:class:`ReproVerdict`, :func:`load_winner`) is
pure → unit-tested under ``.venv`` / CI. :func:`verify_winner` calls ``short_finetune``, so it
needs a GPU + the dataset and is GPU-gated (Kaggle: ``kaggle/run.py`` ``MODE="verify_winner"``).

**Scope.** ``fresh`` mAPs are 5-epoch warm-head **PROXY** numbers (the CP 2.4 ranking signal),
reproducing the *cached proxy* — NOT the full-train deployable accuracy. This checkpoint
certifies the winner is a stable, reloadable artifact; Phase 8 turns it into the shipped model.

Run (GPU)::

    python -m eval.verify_winner --winner-dir state/winner_v1 \
        --head-weights <gate best.pt> --freeze-head --device cuda --imgsz 640 --batch 16
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# PROJECT_PLAN.md:598 — the winner check averages 3 seeds (a single warm-head fine-tune is too
# noisy to certify reproduction). Seeds are offset from the search seeds so this is a genuinely
# fresh draw, not a replay of a cached run.
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)

# --- CP 3.5 reproducibility band (the methodological knob) -------------------------------------
# The DoD says "reproduce cached accuracy *within noise*". This is that noise floor. It is a
# risk-appetite choice (D4-adjacent, like the CP 2.4 gates): too tight → a faithful pipeline
# false-fails on seed jitter; too loose → the DoD certifies nothing. Default reasoning: CP 2.4's
# warm-head re-test measured a single-pair reproduction spread of ~0.0145 mAP (repro-Δ). The
# cached value is one draw and ``fresh`` is a 3-seed mean, so |mean₃ − cached| has std
# ≈ √(1 + 1/3)·σ ≈ 1.15σ with σ ≈ 0.0145/√2 ≈ 0.010 → ≈ 0.012; a ~1.7σ band lands near 0.020.
REPRO_BAND = 0.020


def load_winner(winner_dir: Any) -> dict:
    """Read the serialized winner-v1 record that :func:`search.select_winner.serialize_winner`
    wrote (``winner.json`` under ``winner_dir``). Pure; no torch."""
    return json.loads((Path(winner_dir) / "winner.json").read_text())


@dataclass(frozen=True)
class ReproVerdict:
    """Did α*'s fresh warm-head re-run reproduce its cached proxy mAP within the noise band?

    ``fresh_seeds[i]`` is the proxy mAP from re-fine-tuning α* at the i-th verification seed;
    ``cached_acc`` is the value stored in ``winner.json`` (from the search). The verdict gates
    on the **3-seed mean** vs the cached value (see :attr:`passes`).
    """

    cached_acc: float
    fresh_seeds: list[float]
    band: float = REPRO_BAND

    @property
    def fresh_mean(self) -> float:
        return sum(self.fresh_seeds) / len(self.fresh_seeds)

    @property
    def delta(self) -> float:
        """Signed gap ``mean(fresh) − cached`` (positive = the re-run scored *higher*)."""
        return self.fresh_mean - self.cached_acc

    @property
    def worst_delta(self) -> float:
        """Largest single-seed deviation from cached — the diagnostic for a strict (worst-of-k)
        reading, reported alongside the mean-based verdict."""
        return max(abs(x - self.cached_acc) for x in self.fresh_seeds)

    @property
    def passes(self) -> bool:
        # DoD rule: the 3-seed MEAN reproduces the cached proxy within the band. Averaging is what
        # PROJECT_PLAN.md:598 prescribes for the winner check; a per-seed (worst-of-3) rule is the
        # stricter alternative — see ``worst_delta``. Flip here if the write-up wants strict.
        return abs(self.delta) <= self.band


def verify_winner(
    winner_dir: Any,
    *,
    head_weights: Any,
    freeze_head: bool = True,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    band: float = REPRO_BAND,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    epochs: int = 5,
    supernet: Any = None,
    save_weights: bool = True,
    max_steps: int | None = None,
) -> dict:
    """Reload α* from ``winner_dir`` and reproduce its cached proxy mAP over ``seeds`` fresh
    warm-head fine-tunes; write ``repro.json`` (and ``weights.pt`` from the first seed) beside
    the winner record. Returns the repro payload. GPU-gated (imports ``short_finetune`` lazily).
    """
    from eval.shortft import short_finetune  # lazy: torch/ultralytics/ofa only on the GPU run

    wdir = Path(winner_dir)
    rec = load_winner(wdir)
    arch = rec["arch"]
    cached = float(rec["acc"])

    fresh: list[float] = []
    for i, s in enumerate(seeds):
        # Persist weights once, from the first seed — the canonical winner-v1 state_dict.
        save_to = (wdir / "weights.pt") if (save_weights and i == 0) else None
        metrics = short_finetune(
            arch,
            epochs=epochs,
            seed=s,
            imgsz=imgsz,
            batch=batch,
            device=device,
            supernet=supernet,
            head_weights=head_weights,
            freeze_head=freeze_head,
            max_steps=max_steps,
            save_to=save_to,
        )
        fresh.append(float(metrics["map"]))

    verdict = ReproVerdict(cached_acc=cached, fresh_seeds=fresh, band=band)
    payload = {
        "cached_acc": cached,
        "fresh_seeds": fresh,
        "seeds": list(seeds),
        "fresh_mean": verdict.fresh_mean,
        "delta": verdict.delta,
        "worst_delta": verdict.worst_delta,
        "band": band,
        "passes": verdict.passes,
        "arch": arch,
        "method": rec.get("method"),
        "search_seed": rec.get("seed"),
        "latency_ms": rec.get("latency_ms"),
        "t_max_ms": rec.get("t_max_ms"),
        "weights": (str(wdir / "weights.pt") if save_weights else None),
        "head_weights": str(head_weights),
        "freeze_head": freeze_head,
        "epochs": epochs,
        "imgsz": imgsz,
        "note": (
            "fresh = warm-head 5-epoch PROXY mAP re-derived in a clean session; passes iff the "
            "3-seed mean is within band of the cached proxy acc. Deployable accuracy is Phase 8."
        ),
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (wdir / "repro.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="CP 3.5 — reload α* and reproduce its cached proxy mAP (3-seed warm-head).")
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--head-weights", type=Path, required=True,
                   help="frozen gate-head donor .pt (the CP 2.4 warm-head proxy protocol)")
    p.add_argument("--freeze-head", dest="freeze_head", action="store_true", default=True,
                   help="freeze the warm-started head (default; the established proxy protocol)")
    p.add_argument("--no-freeze-head", dest="freeze_head", action="store_false")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                   help="comma-separated verification seeds (default 3-seed avg)")
    p.add_argument("--band", type=float, default=REPRO_BAND,
                   help="reproducibility noise band on |mean(fresh) − cached| mAP")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", default=True)
    a = p.parse_args(argv)

    seeds = tuple(int(x) for x in a.seeds.split(","))
    out = verify_winner(
        a.winner_dir, head_weights=a.head_weights, freeze_head=a.freeze_head, seeds=seeds,
        band=a.band, device=a.device, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs,
        max_steps=a.max_steps, save_weights=a.save_weights,
    )
    verdict = "PASS" if out["passes"] else "FAIL"
    print(f"[CP 3.5] cached={out['cached_acc']:.4f}  fresh_mean={out['fresh_mean']:.4f} "
          f"(seeds {out['seeds']} -> {[round(x, 4) for x in out['fresh_seeds']]})")
    print(f"  delta={out['delta']:+.4f}  worst={out['worst_delta']:.4f}  band={out['band']:.4f} "
          f"-> {verdict}")
    print(f"  weights: {out['weights']}")
    return 0 if out["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
