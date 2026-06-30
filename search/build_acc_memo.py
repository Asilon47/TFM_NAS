"""Distill prior BO eval caches into an accuracy memo (compute-reuse across resolutions).

Accuracy is measured at a fixed ``imgsz`` (640) and is therefore independent of the
LUT *latency* resolution (the ``--res`` key). So the ``acc`` values cached by a @224
search are valid, as-is, for the @640 search. This collects every ``{arch, acc}`` from
the given cache shards into one memo JSON that ``search.bo --acc-memo`` consults before
the GPU oracle — turning already-spent fine-tunes into free lookups for *whichever*
method (BO or random search) re-proposes that arch, without injecting any arch into a
method's search history (so the BO-vs-random DoD stays a fair comparison).

    python -m search.build_acc_memo \
        --cache "data/cp33_kaggle_out/cp33_bo_cache_r224.*.jsonl" \
        --out data/cp33_acc_memo.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect(patterns: list[str]) -> list[dict]:
    """Read all matching JSONL shards into a deduped list of ``{arch, acc}`` records.

    Dedup is exact-arch (not canonical) — the loader (``bo.load_acc_memo``) canonicalizes
    and averages, so the same backbone seen under several seeds becomes one estimate there.
    """
    seen: dict[str, dict] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partial last line from a killed session
                if "arch" not in rec or "acc" not in rec:
                    continue
                key = json.dumps(rec["arch"], sort_keys=True) + f"|{rec['acc']}"
                seen[key] = {"arch": rec["arch"], "acc": float(rec["acc"])}
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", nargs="+", required=True, metavar="GLOB",
                    help="one or more globs for the JSONL eval-cache shards")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cp33_acc_memo.json")
    a = ap.parse_args(argv)

    records = collect(a.cache)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(records, indent=2) + "\n")
    distinct = {json.dumps(r["arch"], sort_keys=True) for r in records}
    print(f"wrote {len(records)} record(s) ({len(distinct)} distinct arch) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
