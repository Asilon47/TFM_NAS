#!/usr/bin/env bash
# Fetch the AutoTiler core library (LibTile.a) — the single closed-source
# component of the GAP8 toolchain.
#
# GreenWaves Technologies is defunct (both greenwaves-technologies.com and
# www.* fail DNS, checked 2026-07-15) and the official distribution flow
# (tools/autotiler_v3/get_tiler.py -> registration -> emailed personal URL)
# is dead with it. Two independent public gap_sdk forks preserved the
# byte-identical libtile.4.3.5.a their owners downloaded while the server
# was alive; we fetch from either and verify the SHA-256 before use.
#
# The blob is proprietary (GreenWaves AutoTiler EULA, mirrored at
# gap_sdk/tools/autotiler_v3/LICENSE: use restricted to compiling code for
# GAP targets — exactly what we do). It is fetched on demand and NEVER
# committed to this repo (mcu/vendor/ is gitignored), same discipline as
# the OFA checkpoint in supernet/download_ofa.py.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")" && pwd)/vendor"
DEST="${DEST_DIR}/LibTile.a"
# libtile.4.3.5.a — matches TILER_VER=4.3.5 pinned by gap_sdk master's
# tools/autotiler_v3/Makefile. Attested byte-identical (2026-07-15) at:
SHA256="541f4978f3e55c6650207b3f8eebcfe0d311acd3320f81eb0a59890ec928c728"
URLS=(
  "https://raw.githubusercontent.com/edoardobonura/gap_sdk/master/tools/autotiler_v3/Autotiler/LibTile.a"
  "https://raw.githubusercontent.com/boomer319/gap_sdk/master/tools/autotiler_v3/Autotiler/LibTile.a"
)
# Known alternative (NOT used): cbezaitis/gap_sdk carries an older/smaller
# LibTile.a (sha256 329a6228..., 976850 B) that needed a GenTilingDebug stub.

verify() { echo "${SHA256}  $1" | sha256sum --check --status; }

if [[ -f "${DEST}" ]] && verify "${DEST}"; then
    echo "LibTile.a already present and verified: ${DEST}"
    exit 0
fi

mkdir -p "${DEST_DIR}"
for url in "${URLS[@]}"; do
    echo "Fetching ${url}"
    if curl -fsSL --retry 3 -o "${DEST}.tmp" "${url}" && verify "${DEST}.tmp"; then
        mv "${DEST}.tmp" "${DEST}"
        echo "OK: $(sha256sum "${DEST}")"
        exit 0
    fi
    echo "WARN: download or checksum failed for ${url}" >&2
    rm -f "${DEST}.tmp"
done

echo "FATAL: could not obtain a verified LibTile.a from any source." >&2
exit 1
