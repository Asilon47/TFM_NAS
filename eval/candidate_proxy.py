"""CP 10.3 — proxy-score a batch of MCU search candidates (the accuracy half of the screen).

One candidate = the `mcu.cycle_oracle` contract (arch + width spec + neck + imgsz). Per
candidate: build the graft (warm gate head, UNFROZEN), l2 spec-prune it (counts
spec-pinned = the same shape the oracle priced), then a 5-epoch bare-AdamW short
fine-tune of the whole net at the candidate's OWN resolution — i.e. the recover_graft
protocol truncated to proxy length. NOT the frozen-head CP 2.4 instrument: DepGraph
slices through the head (prune_graft refuses frozen params for exactly that reason), so
a pruned candidate has no intact donor head to freeze; the anchors' 5-ep-proxy-vs-100-ep-
final pairs are what validate THIS instrument (the pruned-fidelity leg). Rows append to a
jsonl (resumable by candidate key; Kaggle-kill-safe). Wave-1 runs 1 seed/candidate — the
screen's spread (res 128-192) dwarfs the recorded per-seed sigma 0.005-0.025; de-noise
happens at pick time, never here.

Run (CUDA + dataset; Kaggle MODE=candidate_proxy or locally under .venv-nas):
    python -m eval.candidate_proxy --candidates mcu/screens/wave1.json \\
        --out data/mcu/screen/wave1_proxy.jsonl --epochs 5 --device cuda
CPU smoke: append --max-steps 2 --limit 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DONOR = REPO / "runs/pose/experiments/gate_baseline/weights/best.pt"


def load_candidates(path: Path) -> list[dict]:
    """Accepts either a bare candidate list or the screen results file (status rows)."""
    data = json.loads(Path(path).read_text())
    out = []
    for row in data:
        cand = row.get("candidate", row) if isinstance(row, dict) else row
        if "arch" not in cand:
            raise ValueError(f"not a candidate row: {row!r:.120}")
        if row is not cand and row.get("status") not in (None, "ok"):
            continue                                   # infeasible on cycles -> no proxy
        out.append(cand)
    return out


def done_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line}


def proxy_one(cand: dict, *, donor: Path, epochs: int, seed: int, device: str,
              supernet: Any = None, max_steps: int | None = None,
              recipe: bool = False) -> dict:
    """Build -> spec-prune -> short-FT at the candidate's res -> row dict.

    ``recipe`` routes the fine-tune through ``recovery_finetune`` with the beat-n recipe-lite
    bundle (cos-LR + warmup + EMA + close-mosaic) instead of the bare-AdamW ``short_finetune``
    — the +5-pt lever, for the 100-ep FINALS. Rows are tagged so a recipe twin never overwrites
    its bare sibling in a resumable jsonl.
    """
    import torch

    from detect.pose_model import build_grafted_pose_model
    from mcu.cycle_oracle import candidate_key, canonical_candidate
    from prune.prune_baseline import TRACE_IMGSZ
    from prune.prune_graft import prune_graft
    from prune.recover_graft import spec_ratio_dict

    c = canonical_candidate(cand)
    model = build_grafted_pose_model(c["arch"], supernet=supernet, head_weights=donor,
                                     freeze_head=False, neck=c["neck"])
    prd, spec_ignored = spec_ratio_dict(model, c["arch"]["d"], c["spec"])
    report = prune_graft(model.to("cpu"), torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                         ratio=float(c["spec"]["rest_ratio"]), pruning_ratio_dict=prd,
                         extra_ignored=spec_ignored, importance="l2")
    if recipe:
        from prune.prune_baseline import recovery_finetune
        res = recovery_finetune(model, epochs=epochs, seed=seed, imgsz=c["imgsz"],
                                device=device, max_steps=max_steps, cos_lr=True,
                                warmup_epochs=3.0, ema=True,
                                close_mosaic=10 if epochs > 20 else 0)
        protocol = ("warm head UNFROZEN, l2 spec-prune, recipe-lite recovery (cos-LR+warmup"
                    "+EMA+close-mosaic, no KD) — the beat-n +5-pt lever on the graft final")
    else:
        from eval.shortft import short_finetune
        res = short_finetune(c["arch"], prebuilt=model, epochs=epochs, seed=seed,
                             imgsz=c["imgsz"], device=device, supernet=supernet,
                             max_steps=max_steps)
        protocol = ("warm head UNFROZEN, l2 spec-prune, bare-AdamW whole-net (recover_graft "
                    "truncated; pruned-fidelity leg validates it via the anchor 5ep-vs-100ep "
                    "pairs)")
    return {"key": candidate_key(c), "candidate": c, "params": report["params_after"],
            "proxy_map": res["map"], "proxy_map50": res["map50"],
            "epochs": epochs, "seed": seed, "recipe": recipe, "protocol": protocol}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--recipe", action="store_true",
                    help="recipe-lite recovery (finals; the beat-n +5-pt lever)")
    a = ap.parse_args(argv)

    from mcu.cycle_oracle import candidate_key
    from supernet.sampler import load_supernet

    cands = load_candidates(a.candidates)
    done = done_keys(a.out)
    todo = [c for c in cands if candidate_key(c) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[plan] {len(cands)} candidates, {len(done)} already done, {len(todo)} to run")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sn = load_supernet()
    for i, cand in enumerate(todo):
        row = proxy_one(cand, donor=a.donor, epochs=a.epochs, seed=a.seed,
                        device=a.device, supernet=sn, max_steps=a.max_steps,
                        recipe=a.recipe)
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {row['key']} res={row['candidate']['imgsz']} "
              f"neck={row['candidate']['neck']} params={row['params']:,} "
              f"proxy_map={row['proxy_map']:.4f}", flush=True)
    print(f"[done] rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
