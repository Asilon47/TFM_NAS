"""Resumability: track which row_keys already live in data/lut.jsonl."""
import json
from pathlib import Path


def completed_keys(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get("row_key")
            if k:
                done.add(k)
    return done
