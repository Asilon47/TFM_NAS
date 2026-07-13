#!/usr/bin/env python3
"""Drive one winner-v2-OFA graft run on a Lightning AI studio (lightning_sdk).

The Lightning leg of the wave-1 matrix (Kaggle quota = 0 this week; Colab CLI is the other
leg). A studio's disk PERSISTS across stop/start, so the entry's 10-epoch resume ckpt makes
interrupted runs cheap; `--stop` (default) releases the machine after the pull so free
credits only burn while training.

Auth (one-time, from lightning.ai → profile → Keys):
    export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...

Run (laptop, .venv-cloud):
    .venv-cloud/bin/python cloud/lightning_job.py --name tfm-w1 --machine T4 -- \
        --spec prune/specs/v2_act292.json --seed 0
    .venv-cloud/bin/python cloud/lightning_job.py --name tfm-w1 --pull-only   # artifacts only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Asilon47/TFM_NAS.git"
KACCT_SFX = {1: "", 2: " (Copy)", 3: " (Copy 2)"}   # kaggle/push.sh multi-account contract


def remote_command(entry_args: list[str]) -> str:
    """Clone/refresh the repo on the studio, then run the platform-agnostic entry."""
    entry = " ".join(entry_args)
    return ("set -e; [ -d ~/TFM_NAS ] || git clone " + REPO_URL + " ~/TFM_NAS; "
            "cd ~/TFM_NAS; git fetch origin && git reset --hard origin/main; "
            "python colab/run_prune_graft.py --secrets-root ~/tfm_secrets "
            f"--out-dir ~/tfm_out {entry}")


def _env_from_secrets() -> None:
    """LIGHTNING_USER_ID / LIGHTNING_API_KEY from gitignored secrets/ files when unset
    (lightning.ai → profile → Keys; same secrets-dir contract as the Kaggle creds)."""
    import os

    for env, fname in (("LIGHTNING_USER_ID", "lightning_user_id"),
                       ("LIGHTNING_API_KEY", "lightning_api_key")):
        f = ROOT / "secrets" / fname
        if not os.environ.get(env) and f.exists():
            os.environ[env] = f.read_text().strip()


def tripwire(root: Path) -> None:
    """The studio clones GitHub — refuse to launch with an unpushed HEAD."""
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
        up = subprocess.check_output(["git", "-C", str(root), "rev-parse", "@{u}"]).strip()
    except subprocess.CalledProcessError:
        return
    if head != up:
        raise SystemExit("REFUSING: local HEAD != upstream — the studio clones GitHub, "
                         "push first (or pass --push-anyway).")


def main() -> None:
    ap = argparse.ArgumentParser(description="winner-v2-OFA graft run on a Lightning studio.")
    ap.add_argument("--name", default="tfm-w1")
    ap.add_argument("--machine", default="T4")
    ap.add_argument("--teamspace", default=None)
    ap.add_argument("--kacct", type=int, default=1, choices=(1, 2, 3),
                    help="which secrets/access_token* pair stages the Kaggle dataset")
    ap.add_argument("--keep", action="store_true", help="leave the studio running")
    ap.add_argument("--pull-only", action="store_true", help="just download ~/tfm_out")
    ap.add_argument("--push-anyway", action="store_true")
    ap.add_argument("entry", nargs="*", help="run_prune_graft.py args (after --)")
    a = ap.parse_args()

    _env_from_secrets()
    from lightning_sdk import Machine, Studio  # .venv-cloud

    studio = Studio(a.name, teamspace=a.teamspace, create_ok=True)
    print(f"[lightning] studio {studio.name} status={studio.status}", flush=True)

    dest = ROOT / "data" / "lightning_out" / a.name
    if a.pull_only:
        _pull(studio, dest)
        return

    if not a.push_anyway:
        tripwire(ROOT)
    sfx = KACCT_SFX[a.kacct]
    token = ROOT / "secrets" / f"access_token{sfx}"
    user = ROOT / "secrets" / f"kaggle_username{sfx}"
    if not (token.exists() and user.exists()):
        raise SystemExit(f"missing {token} / {user}")

    studio.start(Machine.from_str(a.machine))
    print(f"[lightning] started on {a.machine}", flush=True)
    studio.upload_file(str(token), remote_path="tfm_secrets/secrets/access_token")
    studio.upload_file(str(user), remote_path="tfm_secrets/secrets/kaggle_username")

    cmd = remote_command(a.entry)
    print(f"[lightning] + {cmd}", flush=True)
    out = studio.run(cmd)
    print(out, flush=True)

    _pull(studio, dest)
    if not a.keep:
        studio.stop()
        print("[lightning] studio stopped (disk persists — resume is free)", flush=True)


def _pull(studio, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    studio.run("tar czf ~/tfm_out.tgz -C ~ tfm_out")
    tgz = dest / "tfm_out.tgz"
    studio.download_file("tfm_out.tgz", str(tgz))
    with tarfile.open(tgz) as tf:
        tf.extractall(dest, filter="data")
    tgz.unlink()
    inner = dest / "tfm_out"
    if inner.is_dir():
        for f in inner.iterdir():
            f.rename(dest / f.name)
        inner.rmdir()
    print(f"[lightning] pulled -> {dest}", flush=True)
    for f in sorted(dest.iterdir()):
        print("  ", f.name, flush=True)


if __name__ == "__main__":
    sys.exit(main())
