"""Build the GAP8 int8-calibration set from the gate-pose dataset.

Converts a seeded random sample of dataset images to grayscale PNGs at the
candidate MCU resolutions (the AI-deck's HM01B0 sensor is 320x320 gray; the
nets will run lower). NNTool's quantization step reads these to set the int8
scales; channel replication to a 3-ch stem happens at feed time, not here.

Calibration draws from the TRAIN split so val stays untouched for accuracy
claims. Output goes under data/ (gitignored); a manifest records the draw.

Run inside .venv-nas:  python mcu/prep_calib.py
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def build_calib(
    dataset: Path, split: str, n: int, resolutions: list[int], seed: int, out: Path
) -> dict:
    src_dir = dataset / "images" / split
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if len(images) < n:
        raise SystemExit(f"only {len(images)} images in {src_dir}, need {n}")
    picked = random.Random(seed).sample(images, n)

    for res in resolutions:
        res_dir = out / f"gray{res}"
        res_dir.mkdir(parents=True, exist_ok=True)
        for img_path in picked:
            gray = Image.open(img_path).convert("L").resize((res, res), Image.BILINEAR)
            gray.save(res_dir / f"{img_path.stem}.png")

    manifest = {
        "source_split": split,
        "n": n,
        "seed": seed,
        "resolutions": resolutions,
        "images": [p.name for p in picked],
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=REPO / "dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--res", type=int, nargs="+", default=[224, 192, 160])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "mcu" / "calib")
    args = ap.parse_args()

    manifest = build_calib(args.dataset, args.split, args.n, args.res, args.seed, args.out)
    print(
        f"calib set: {manifest['n']} images x {manifest['resolutions']} "
        f"(seed {manifest['seed']}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
