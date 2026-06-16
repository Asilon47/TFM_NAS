"""Download + verify the pretrained OFA-MBv3 supernet checkpoint.

Implements PROJECT_PLAN.md CP 1.2: pulls
``ofa_mbv3_d234_e346_k357_w1.0`` from MIT-HAN-Lab's mirror into
``<project_root>/.cache/ofa/`` and verifies SHA256 against the pin
recorded below.

Idempotent: re-running with the file already present and the hash
matching is a no-op (exit 0). A mismatch fails loudly so an upstream
file rotation cannot silently corrupt downstream training runs.

The script uses only the standard library so it does not pull in
``gdown`` / ``PIL`` (transitive deps of ``ofa.model_zoo``). Those
arrive at CP 1.3 when the sampler actually instantiates the supernet.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_NAME = "ofa_mbv3_d234_e346_k357_w1.0"
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_nets/"
    + CHECKPOINT_NAME
)
PINNED_SHA256 = "a7def36bb4e4c688c16d37eb60d5d34b2e6dcf6438c05bc86dea918fda04c6c7"

# Project-relative cache (was ~/.cache/ofa/ — see procedure.md "Cache
# relocation"). Derived from this file's location, not CWD, so the
# contract holds whether the script is run as `python -m supernet.download_ofa`
# from the repo root or `python supernet/download_ofa.py` from anywhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "ofa"
CHECKPOINT_PATH = CACHE_DIR / CHECKPOINT_NAME


def compute_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n         -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if total:
                pct = 100.0 * read / total
                print(f"  {read / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="\r", file=sys.stderr)
    print("", file=sys.stderr)
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true",
                   help="re-download even if the cached file matches the pin")
    args = p.parse_args(argv)

    if args.force and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    if not CHECKPOINT_PATH.exists():
        download(CHECKPOINT_URL, CHECKPOINT_PATH)

    actual = compute_sha256(CHECKPOINT_PATH)
    size_mb = CHECKPOINT_PATH.stat().st_size / 1e6

    if not PINNED_SHA256:
        print(f"  path:   {CHECKPOINT_PATH}", file=sys.stderr)
        print(f"  size:   {size_mb:.1f} MB", file=sys.stderr)
        print(f"  sha256: {actual}", file=sys.stderr)
        print("PINNED_SHA256 is empty. Copy the sha256 above into "
              "download_ofa.py and re-run to verify.", file=sys.stderr)
        return 1

    if actual != PINNED_SHA256:
        print(f"SHA256 mismatch for {CHECKPOINT_PATH}", file=sys.stderr)
        print(f"  expected: {PINNED_SHA256}", file=sys.stderr)
        print(f"  actual:   {actual}", file=sys.stderr)
        print("Upstream may have rotated the file. Investigate before bumping the pin.",
              file=sys.stderr)
        return 2

    print(f"OK  {CHECKPOINT_PATH}  ({size_mb:.1f} MB, sha256={actual[:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
