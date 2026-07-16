"""CP 5.2 — the graft-interface ablation: V0–V3 × fresh seeds under the warm-head proxy.

The Phase-5 accuracy-seam experiment (procedure.md "Plan pivot"): the SAME winner-v1 backbone
under four interface treatments —

* ``v0_control``   — the historical graft (random 1×1 adapters, no neck),
* ``v1_net2wider`` — CP 4.4's identity-embedding adapter init,
* ``v2_topdown``   — V1 + the zero-gated top-down nano-neck (CP 5.1),
* ``v3_pan``       — V1 + the PAN variant (top-down + bottom-up) — **conditional**: run only if
  V2 beats V1 by more than 1× the observed noise (:func:`v3_warranted`), per PROJECT_PLAN
  CP 5.2 (``--include-v3`` forces, ``--skip-v3`` forbids).

Protocol = exactly the CP 3.5 de-noise oracle: 5-epoch warm-head fine-tune with the frozen
gate donor, imgsz 640, batch 16, fresh seeds {1, 2, 3}; resumable per-(variant, seed) jsonl
cache namespaced by the protocol (``graft_ablate_e5_r640``, the ``search/denoise.py`` cache
pattern) so a killed Kaggle session continues. For necked variants the fine-tuned **gate
magnitudes** are recorded per seed — the Stage-R diagnostic ("did the data ever turn the neck
on?"; zero-ish gates + no mAP gain = the literature-consistent negative result pre-registered
in docs/research/stageR_graft_interface.md).

Pure pieces (:func:`summarize`, :func:`v3_warranted`, :func:`gates_from_state_dict`, cache IO)
are ``.venv``/CI-tested; the driver is GPU-gated (Kaggle ``MODE="graft_ablate"``).
"""
from __future__ import annotations

import datetime as dt
import json
import statistics as st
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Variant → build_grafted_pose_model kwargs (the ONLY thing that changes across variants).
VARIANTS: dict[str, dict[str, Any]] = {
    "v0_control": {},
    "v1_net2wider": {"adapter_init": "net2wider"},
    "v2_topdown": {"adapter_init": "net2wider", "neck": "topdown"},
    "v3_pan": {"adapter_init": "net2wider", "neck": "pan"},
}
CORE_VARIANTS: tuple[str, ...] = ("v0_control", "v1_net2wider", "v2_topdown")
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)  # the CP 3.5 fresh seeds (search seed 0 is biased)
V3_MARGIN_SIGMA = 1.0                       # PROJECT_PLAN CP 5.2: V3 only if V2 > V1 by >1σ


# ---- pure logic (CPU-tested) --------------------------------------------------

def summarize(maps_by_variant: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Per-variant ``mean``/``std`` (population σ, matching search.denoise's convention)."""
    out: dict[str, dict[str, float]] = {}
    for name, maps in maps_by_variant.items():
        if not maps:
            raise ValueError(f"variant {name!r} has no mAPs to summarize")
        out[name] = {
            "mean": st.mean(maps),
            "std": st.pstdev(maps) if len(maps) > 1 else 0.0,
        }
    return out


def v3_warranted(summary: Mapping[str, Mapping[str, float]], *,
                 margin_sigma: float = V3_MARGIN_SIGMA) -> bool:
    """The CP 5.2 gate for the optional V3 arm: V2 beats V1 by more than ``margin_sigma``×σ.

    σ = the larger of the two variants' per-variant seed spreads (conservative). Raises if
    either arm is missing — the decision must never be made on absent data.
    """
    for need in ("v1_net2wider", "v2_topdown"):
        if need not in summary:
            raise ValueError(f"v3_warranted needs {need!r} in the summary")
    v1, v2 = summary["v1_net2wider"], summary["v2_topdown"]
    noise = max(v1["std"], v2["std"])
    return (v2["mean"] - v1["mean"]) > margin_sigma * noise


def gates_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, float]:
    """The neck's learned gate scalars out of a grafted model ``state_dict`` (``{}`` if none)."""
    gates: dict[str, float] = {}
    for key, value in state_dict.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf in ("g54", "g43", "g34", "g45"):
            gates[leaf] = float(value)
    return gates


def _load_cache(path: Any) -> dict[tuple[str, int], dict]:
    """(variant, seed) → cached row. Same jsonl pattern as search.denoise's cache."""
    memo: dict[tuple[str, int], dict] = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                memo[(row["variant"], int(row["seed"]))] = row
    return memo


def _append_cache(path: Any, row: dict) -> None:
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


# ---- GPU driver (resumable; finetune_fn injectable for the orchestration tests) ----

def run_ablation(
    arch: dict,
    *,
    head_weights: Any,
    variants: Sequence[str] = CORE_VARIANTS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    epochs: int = 5,
    supernet: Any = None,
    cache: Any = None,
    workdir: Any = None,
    finetune_fn: Callable[..., dict[str, float]] | None = None,
) -> dict[str, dict]:
    """Run (variant × seed) warm-head fine-tunes, resumably; return per-variant rows.

    Each (variant, seed) lands in ``cache`` as it finishes. For necked variants the fine-tuned
    model's state_dict is written to ``workdir`` (temp) just long enough to read the gate
    scalars out — ``short_finetune`` deliberately does not return the model.
    """
    if finetune_fn is None:
        from eval.shortft import short_finetune  # lazy: torch/ultralytics/ofa on the GPU run
        finetune_fn = short_finetune

    memo = _load_cache(cache)
    results: dict[str, dict] = {}
    for name in variants:
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
        kwargs = VARIANTS[name]
        maps: list[float] = []
        gates: dict[str, dict[str, float]] = {}
        for seed in seeds:
            if (name, seed) in memo:
                row = memo[(name, seed)]
            else:
                save_to = None
                if "neck" in kwargs and workdir is not None:
                    save_to = Path(workdir) / f"graft_ablate_{name}_s{seed}.pt"
                metrics = finetune_fn(
                    dict(arch), epochs=epochs, seed=seed, imgsz=imgsz, batch=batch,
                    device=device, supernet=supernet, head_weights=head_weights,
                    freeze_head=True, graft_kwargs=dict(kwargs), save_to=save_to,
                )
                row = {"variant": name, "seed": seed,
                       "map": float(metrics["map"]), "map50": float(metrics.get("map50", 0.0))}
                if save_to is not None and Path(save_to).exists():
                    import torch
                    sd = torch.load(save_to, map_location="cpu")
                    row["gates"] = gates_from_state_dict(sd)
                    Path(save_to).unlink()  # scratch only — winner weights come from CP 5.3
                memo[(name, seed)] = row
                _append_cache(cache, row)
            maps.append(float(row["map"]))
            if row.get("gates"):
                gates[str(seed)] = row["gates"]
        results[name] = {"maps": maps, "gates": gates}
    return results


def assemble_report(arch: dict, results: Mapping[str, dict], *, seeds: Sequence[int],
                    epochs: int, imgsz: int, batch: int) -> dict:
    """The graft_ablate.json payload: per-variant stats + the headline deltas + the V3 gate."""
    summary = summarize({name: r["maps"] for name, r in results.items()})
    report: dict[str, Any] = {
        "protocol": {"epochs": epochs, "imgsz": imgsz, "batch": batch,
                     "freeze_head": True, "seeds": list(seeds),
                     "note": "the CP 3.5 warm-head proxy oracle; ranking signal, not "
                             "deployable accuracy"},
        "arch": arch,
        "variants": {name: {**summary[name], "maps": r["maps"], "gates": r["gates"]}
                     for name, r in results.items()},
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if {"v0_control", "v1_net2wider"} <= set(summary):
        report["v1_vs_v0_delta"] = summary["v1_net2wider"]["mean"] - summary["v0_control"]["mean"]
    if {"v1_net2wider", "v2_topdown"} <= set(summary):
        report["v2_vs_v1_delta"] = summary["v2_topdown"]["mean"] - summary["v1_net2wider"]["mean"]
        report["v3_warranted"] = v3_warranted(summary)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="CP 5.2 graft-interface ablation (V0-V3 × fresh seeds, warm-head proxy).")
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1",
                   help="directory holding winner.json (the arch under ablation)")
    p.add_argument("--head-weights", type=Path, required=True,
                   help="the frozen gate-head donor (gate_best.pt)")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--device", default="cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cache", type=Path, default=None, help="resumable per-(variant,seed) jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "graft_ablate.json")
    v3 = p.add_mutually_exclusive_group()
    v3.add_argument("--include-v3", action="store_true", help="force the PAN arm")
    v3.add_argument("--skip-v3", action="store_true", help="forbid the PAN arm")
    a = p.parse_args(argv)

    arch = json.loads((a.winner_dir / "winner.json").read_text())["arch"]
    seeds = tuple(int(s) for s in a.seeds.split(","))
    workdir = a.out.parent
    workdir.mkdir(parents=True, exist_ok=True)

    common = dict(head_weights=a.head_weights, seeds=seeds, device=a.device, imgsz=a.imgsz,
                  batch=a.batch, epochs=a.epochs, cache=a.cache, workdir=workdir)
    results = run_ablation(arch, variants=CORE_VARIANTS, **common)  # type: ignore[arg-type]

    run_v3 = a.include_v3
    if not a.include_v3 and not a.skip_v3:
        run_v3 = v3_warranted(summarize({n: r["maps"] for n, r in results.items()}))
        print(f"[v3] auto-decision: warranted={run_v3} (V2-V1 margin vs 1σ rule)")
    if run_v3:
        results |= run_ablation(arch, variants=("v3_pan",), **common)  # type: ignore[arg-type]

    report = assemble_report(arch, results, seeds=seeds, epochs=a.epochs,
                             imgsz=a.imgsz, batch=a.batch)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[graft-ablate] -> {a.out}")
    for name, v in report["variants"].items():
        gate_note = f"  gates={v['gates']}" if v["gates"] else ""
        print(f"  {name:14s} mean={v['mean']:.4f}±{v['std']:.4f}  maps={v['maps']}{gate_note}")
    if "v2_vs_v1_delta" in report:
        print(f"  v1-v0 delta={report.get('v1_vs_v0_delta', float('nan')):+.4f}   "
              f"v2-v1 delta={report['v2_vs_v1_delta']:+.4f}   "
              f"v3_warranted={report.get('v3_warranted')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
