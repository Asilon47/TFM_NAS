"""Shared Colab-side helpers — a THIRD search backend after ``kaggle/`` and ``jetson/``.

The compute (``search.tpe`` / ultralytics) is identical to Kaggle; only the two I/O
planes move:

* **big data** (the 1.6 GB gate set + LUT + frozen head donor + accuracy memo) is pulled
  from the *existing* ``<user>/tfm-nas-gate-pose`` Kaggle Dataset via the Kaggle CLI — so
  nothing is hand-uploaded to Colab;
* **durable state** (resume caches, the ``cp34_tpe.json`` output, the anchor-B run dir)
  lives on a mounted **Google Drive** so it survives Colab's ~12 h VM recycle.

Credentials mirror the repo's ``secrets/`` contract: a new-style ``KGAT_`` token at
``<drive>/secrets/access_token`` + the username at ``<drive>/secrets/kaggle_username``
(or the classic ``KAGGLE_USERNAME``/``KAGGLE_KEY`` env vars / a Colab userdata secret).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/Asilon47/TFM_NAS.git"  # public — clones without a token
DATASET_SLUG = "tfm-nas-gate-pose"                    # the big-data Kaggle Dataset (no <user>/)
CACHE_SLUG = "tfm-nas-cp33-bo-cache"                  # optional resume-cache seed (see run_colab)
DRIVE_DEFAULT = "/content/drive/MyDrive/tfm_nas"      # durable plane root on the mounted Drive


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def pin_torch_and_install(extra_pkgs: str) -> None:
    """Install the NAS stack WITHOUT letting it upgrade Colab's CUDA torch.

    Same discipline as ``kaggle/run.py`` and the laptop ``.venv`` 2.3.1 pin: read the
    resident torch version, write it as a pip constraint, then install ``extra_pkgs``
    under that constraint so ofa/ultralytics/optuna can never swap the CUDA build for a
    PyPI CPU wheel (which would silently drop the T4).
    """
    torch_ver = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"]
    ).decode().strip()
    # tempdir, not /content: the same helper now serves Lightning studios (2026-07-13).
    import tempfile
    constraint = Path(tempfile.gettempdir()) / "tfm_nas_constraints.txt"
    constraint.write_text(f"torch=={torch_ver}\n")
    print(f"[torch] pinning resident CUDA torch=={torch_ver}", flush=True)
    sh(f"{sys.executable} -m pip install -q --constraint {constraint} {extra_pkgs}")


def ensure_kaggle_credentials(drive_root: Path) -> str:
    """Materialise Kaggle CLI credentials and return the username.

    Resolution order (first hit wins), so the same secrets you already have work with no
    retyping: env vars → ``<drive>/secrets/`` files (the repo contract) → Colab userdata.
    Writes ``~/.kaggle/kaggle.json`` (chmod 600) so the CLI authenticates regardless of
    whether the token is a classic key or a new-style ``KGAT_`` token.
    """
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")

    sec = drive_root / "secrets"
    if not user and (sec / "kaggle_username").exists():
        user = (sec / "kaggle_username").read_text().strip()
    if not key and (sec / "access_token").exists():
        key = (sec / "access_token").read_text().strip()

    if not (user and key):  # last resort: Colab's own secret store
        try:
            from google.colab import userdata  # type: ignore

            user = user or userdata.get("KAGGLE_USERNAME")
            key = key or userdata.get("KAGGLE_KEY")
        except Exception:  # noqa: BLE001 — not on Colab, or secret unset
            pass

    if not (user and key):
        raise SystemExit(
            "No Kaggle credentials. Put your KGAT_ token at "
            f"{sec}/access_token and username at {sec}/kaggle_username "
            "(upload your repo secrets/ to Drive), or set KAGGLE_USERNAME/KAGGLE_KEY."
        )

    # New-style KGAT_ tokens authenticate via KAGGLE_API_TOKEN (as kaggle/push.sh does);
    # the new gRPC CLI 403s if such a token is placed in kaggle.json / KAGGLE_KEY, and a
    # stale kaggle.json would SHADOW the env token — so remove it, don't just supplement.
    # Classic 32-char hex keys still use kaggle.json. Username only builds slugs either way.
    os.environ["KAGGLE_USERNAME"] = user
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if key.startswith("KGAT"):
        os.environ["KAGGLE_API_TOKEN"] = key
        os.environ.pop("KAGGLE_KEY", None)
        if kaggle_json.exists():
            kaggle_json.unlink()
        mode = "KAGGLE_API_TOKEN (new-style)"
    else:
        kaggle_json.parent.mkdir(exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": user, "key": key}))
        kaggle_json.chmod(0o600)
        os.environ["KAGGLE_KEY"] = key
        mode = "kaggle.json (classic)"
    print(f"[kaggle] authenticated as {user} via {mode}", flush=True)
    return user


def _unzip_all(root: Path) -> None:
    """Extract ``root``'s top-level zip, then any nested zips (dataset/ is stored zipped
    by ``push.sh --dir-mode zip``), until only plain files remain."""
    seen: set[Path] = set()
    while True:
        zips = [z for z in root.rglob("*.zip") if z not in seen]
        if not zips:
            break
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            seen.add(z)


def stage_kaggle_dataset(user: str, dest: Path) -> Path:
    """Download + fully unzip ``<user>/tfm-nas-gate-pose`` into ``dest``; return ``dest``.

    Idempotent: if it already holds ``dataset.yaml`` (a prior cell staged it this VM), skip
    the ~1.6 GB re-download. Kaggle nests ``dataset/`` as an inner zip, so unzip recursively.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if find(dest, "dataset.yaml"):
        print(f"[data] {dest} already staged — skipping download", flush=True)
        return dest
    sh(f"kaggle datasets download -d {user}/{DATASET_SLUG} -p {dest}")
    _unzip_all(dest)
    if not find(dest, "dataset.yaml"):
        raise SystemExit(f"FATAL: dataset.yaml not found under {dest} after unzip — "
                         f"is {user}/{DATASET_SLUG} the right slug?")
    return dest


def find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name)) if root.exists() else []
    return hits[0] if hits else None


def wire_repo_data(repo: Path, staged: Path) -> dict[str, Path]:
    """Symlink the staged data into the repo layout the search expects, exactly like
    ``run.py`` does with ``/kaggle/input``. Returns the resolved artefact paths.
    """
    yaml_src = find(staged, "dataset.yaml")
    lut_src = find(staged, "lut.jsonl")
    donor = find(staged, "gate_best.pt")
    frontier = find(staged, "phase3_nsga2_frontier.json")
    memo = find(staged, "cp33_acc_memo.json")
    missing = [n for n, v in (("dataset.yaml", yaml_src), ("lut.jsonl", lut_src),
                              ("gate_best.pt", donor)) if v is None]
    if missing:
        raise SystemExit(f"FATAL: {missing} missing from the staged dataset {staged}.")

    assert yaml_src and lut_src  # narrowed by the missing-check above
    sh(f"rm -rf {repo / 'dataset'} && ln -s {yaml_src.parent} {repo / 'dataset'}")
    (repo / "data").mkdir(exist_ok=True)
    sh(f"ln -sf {lut_src} {repo / 'data' / 'lut.jsonl'}")
    if frontier:
        sh(f"ln -sf {frontier} {repo / 'data' / 'phase3_nsga2_frontier.json'}")
    return {"yaml": yaml_src, "lut": lut_src, "donor": donor,  # type: ignore[dict-item]
            "frontier": frontier, "memo": memo}  # type: ignore[dict-item]


def download_ofa(repo: Path) -> None:
    """Fetch the SHA-pinned OFA supernet checkpoint (internet on; verified in-module)."""
    sh(f"{sys.executable} -m supernet.download_ofa")
