"""Side experiment — full fine-tune of winner-v1: how good does this architecture actually get?

Not a checkpoint, not Phase 8. ``eval/verify_winner.py`` reproduces the 5-epoch warm-head PROXY
mAP (the CP 2.4 ranking signal) within a noise band; this script instead runs a LONGER, by
default *un-frozen* fine-tune of the exact same winner-v1 architecture to see how far it gets on
its own merits — no teacher, no distillation (that is Phase 8's job). It reuses the same bare-
AdamW :func:`eval.shortft.short_finetune` loop that ``eval/proxy_rank.py`` already established as
this repo's own "full" (100-epoch) precedent (the CP 2.4 full-train diagnostic), just applied to
winner-v1 specifically, with the head warm-started **and** left trainable — a combination neither
CP 2.4 (no warm-start) nor CP 3.5 (frozen head) used.

**Caveat carried into every output** (``note`` field + CLI print): this is NOT the Ultralytics
full-trainer recipe (SGD + LR decay + warmup + augmentation schedule) that produced the
0.8774/0.8819 mAP anchors — read the result as "how far does winner-v1 get under this repo's own
established full-train protocol" (comparable to CP 2.4's 0.778-0.850 full-train cluster and to
winner-v1's own 0.610 proxy), not as a strictly apples-to-apples number against the two
Ultralytics-recipe anchors.

Run (GPU)::

    python -m eval.full_finetune --winner-dir state/winner_v1 \
        --head-weights <gate best.pt> --no-freeze-head --device cuda --imgsz 640 --batch 16 \
        --epochs 100
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.verify_winner import load_winner

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_CAVEAT = (
    "full-train mAP via this repo's bare-AdamW short_finetune loop (the same 100-epoch protocol "
    "eval.proxy_rank's CP 2.4 diagnostic used), NOT the Ultralytics full-trainer recipe (SGD + LR "
    "decay + warmup + augmentation schedule) that produced the yolo11n/s anchor mAPs -- comparable "
    "to CP 2.4's own 0.778-0.850 full-train cluster and to winner-v1's 0.610 proxy, not strictly "
    "apples-to-apples with the anchors."
)


def full_finetune(
    winner_dir: Any,
    *,
    head_weights: Any = None,
    freeze_head: bool = False,
    seeds: Sequence[int] = (0,),
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    imgsz: int = 640,
    batch: int = 16,
    supernet: Any = None,
    save_weights: bool = True,
    max_steps: int | None = None,
) -> dict:
    """Reload winner-v1's arch from ``winner_dir`` and run a longer fine-tune to see its
    (proxy-protocol) full-train mAP. Writes ``full_finetune.json`` beside the winner record
    (and ``full_finetune_weights.pt`` from the first seed). Returns the payload.

    Unlike :func:`eval.verify_winner.verify_winner`, this has no pass/fail gate — it is an
    exploratory side experiment, not a DoD. ``head_weights=None`` (default) trains a
    from-scratch head; pass the gate donor with ``freeze_head=False`` (default) to warm-start
    *and* keep training it — the setup most likely to show what this backbone can reach.
    """
    from eval.shortft import short_finetune  # lazy: torch/ultralytics/ofa only on the GPU run

    wdir = Path(winner_dir)
    rec = load_winner(wdir)
    arch = rec["arch"]
    proxy_acc = float(rec["acc"])
    anchors = rec.get("anchors")

    maps: list[float] = []
    for i, s in enumerate(seeds):
        # Persist weights once, from the first seed.
        save_to = (wdir / "full_finetune_weights.pt") if (save_weights and i == 0) else None
        metrics = short_finetune(
            arch,
            epochs=epochs,
            seed=s,
            lr=lr,
            imgsz=imgsz,
            batch=batch,
            device=device,
            supernet=supernet,
            head_weights=head_weights,
            freeze_head=freeze_head,
            max_steps=max_steps,
            save_to=save_to,
        )
        maps.append(float(metrics["map"]))

    mean = sum(maps) / len(maps)
    std = (sum((m - mean) ** 2 for m in maps) / len(maps)) ** 0.5 if len(maps) > 1 else 0.0
    payload = {
        "arch": arch,
        "epochs": epochs,
        "lr": lr,
        "head_weights": (str(head_weights) if head_weights is not None else None),
        "freeze_head": freeze_head,
        "seeds": list(seeds),
        "maps": maps,
        "mean": mean,
        "std": std,
        "proxy_acc": proxy_acc,
        "delta_vs_proxy": mean - proxy_acc,
        "anchors": anchors,
        "weights": (str(wdir / "full_finetune_weights.pt") if save_weights else None),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "note": PROTOCOL_CAVEAT,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (wdir / "full_finetune.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Side experiment: a longer, un-frozen fine-tune of winner-v1 "
                     "(not a DoD, not Phase 8 distillation).")
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--head-weights", type=Path, default=None,
                   help="warm-start donor .pt (optional; omit for a from-scratch head)")
    p.add_argument("--freeze-head", dest="freeze_head", action="store_true", default=False)
    p.add_argument("--no-freeze-head", dest="freeze_head", action="store_false",
                    help="warm-start but keep training the head (default)")
    p.add_argument("--seeds", type=str, default="0",
                   help="comma-separated seed(s); default single-seed (this is the quick variant)")
    p.add_argument("--epochs", type=int, default=100, help="matches eval.proxy_rank's full_epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (CPU smoke)")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", default=True)
    a = p.parse_args(argv)

    seeds = tuple(int(x) for x in a.seeds.split(","))
    out = full_finetune(
        a.winner_dir, head_weights=a.head_weights, freeze_head=a.freeze_head, seeds=seeds,
        epochs=a.epochs, lr=a.lr, device=a.device, imgsz=a.imgsz, batch=a.batch,
        max_steps=a.max_steps, save_weights=a.save_weights,
    )
    print(f"[full-finetune] winner-v1 epochs={out['epochs']} freeze_head={out['freeze_head']} "
          f"seeds={out['seeds']} -> maps={[round(x, 4) for x in out['maps']]} "
          f"mean={out['mean']:.4f}")
    if out["anchors"]:
        a_acc = out["anchors"].get("a", {}).get("acc")
        b_acc = out["anchors"].get("b", {}).get("acc")
        print(f"  proxy={out['proxy_acc']:.4f}  full={out['mean']:.4f}  "
              f"anchor_a(yolo11n)={a_acc}  anchor_b(yolo11s)={b_acc}")
    print(f"  weights: {out['weights']}")
    print(f"  NOTE: {out['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
