#!/usr/bin/env python3
"""Drive one winner-v2-OFA graft run on a Lightning AI studio (lightning_sdk).

The Lightning leg of the wave-1 matrix (Kaggle quota = 0 this week; Colab CLI is the other
leg). Mirrors ``cloud/colab_job.sh``'s launch|poll|pull|stop model: a 100-epoch run outlives
any single blocking ``studio.run`` call, so ``launch`` starts the entry DETACHED (nohup >
studio-log) and returns; ``poll`` tails that log (``ENTRY_EXIT=<rc>`` = done). The studio disk
persists across stop/start, so the entry's 10-epoch resume ckpt makes interruptions cheap.

Auth: ``secrets/lightning_user_id`` + ``secrets/lightning_api_key`` (gitignored; from
lightning.ai → profile → Keys), or the LIGHTNING_USER_ID / LIGHTNING_API_KEY env vars.

    python cloud/lightning_job.py launch --name tfm-w1 --machine T4 -- \
        --ratios 0.50 --technique global_taylor --seed 0
    python cloud/lightning_job.py poll --name tfm-w1
    python cloud/lightning_job.py pull --name tfm-w1
    python cloud/lightning_job.py stop --name tfm-w1
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Asilon47/TFM_NAS.git"
KACCT_SFX = {1: "", 2: " (Copy)", 3: " (Copy 2)"}   # kaggle/push.sh multi-account contract


def _env_from_secrets() -> None:
    """LIGHTNING_* creds from gitignored secrets/ files when the env vars are unset."""
    for env, fname in (("LIGHTNING_USER_ID", "lightning_user_id"),
                       ("LIGHTNING_API_KEY", "lightning_api_key")):
        f = ROOT / "secrets" / fname
        if not os.environ.get(env) and f.exists():
            os.environ[env] = f.read_text().strip()


def _studio(name: str, teamspace: str):
    _env_from_secrets()
    from lightning_sdk import Studio, Teamspace  # .venv-cloud

    if "/" in teamspace:
        owner, ts_name = teamspace.split("/", 1)
        ts: object = Teamspace(name=ts_name, user=owner)   # SDK won't parse the slash itself
    else:
        ts = teamspace
    return Studio(name, teamspace=ts, create_ok=True)


def _launch_cmd(entry_args: list[str], session: str) -> str:
    """Clone/refresh the repo on the studio, then nohup the entry with an rc sentinel."""
    entry = " ".join(entry_args)
    return (
        "set -e; [ -d ~/TFM_NAS ] || git clone " + REPO_URL + " ~/TFM_NAS; "
        "cd ~/TFM_NAS && git fetch origin && git reset --hard origin/main; "
        f"LOG=~/run_{session}.log; "
        "nohup bash -c \"python colab/run_prune_graft.py --secrets-root ~/tfm_secrets "
        f"--out-dir ~/tfm_out {entry}; echo ENTRY_EXIT=\\$? >> $LOG\" > $LOG 2>&1 & "
        "echo LAUNCHED pid=$!")


def _tripwire() -> None:
    import subprocess
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
        up = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "@{u}"]).strip()
    except subprocess.CalledProcessError:
        return
    if head != up and not os.environ.get("KPUSH_ANYWAY"):
        raise SystemExit("REFUSING: local HEAD != upstream — the studio clones GitHub, push "
                         "first (or KPUSH_ANYWAY=1).")


def main() -> None:
    ap = argparse.ArgumentParser(description="winner-v2-OFA graft run on a Lightning studio.")
    ap.add_argument("action", choices=("launch", "poll", "pull", "stop"))
    ap.add_argument("--name", default="tfm-w1")
    ap.add_argument("--machine", default="T4")
    ap.add_argument("--teamspace", default="asilarnous/data-optimization-project")
    ap.add_argument("--kacct", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("entry", nargs="*", help="run_prune_graft.py args (after --)")
    a = ap.parse_args()

    session = a.name
    studio = _studio(a.name, a.teamspace)

    if a.action == "launch":
        _tripwire()
        sfx = KACCT_SFX[a.kacct]
        token = ROOT / "secrets" / f"access_token{sfx}"
        user = ROOT / "secrets" / f"kaggle_username{sfx}"
        if not (token.exists() and user.exists()):
            raise SystemExit(f"missing {token} / {user}")
        from lightning_sdk import Machine
        if str(studio.status) not in ("Status.RUNNING", "RUNNING"):
            studio.start(Machine.from_str(a.machine))
        print(f"[lightning] {session} status={studio.status}", flush=True)
        studio.upload_file(str(token), remote_path="tfm_secrets/secrets/access_token")
        studio.upload_file(str(user), remote_path="tfm_secrets/secrets/kaggle_username")
        entry = a.entry[1:] if a.entry and a.entry[0] == "--" else a.entry
        print(studio.run(_launch_cmd(entry, session)), flush=True)
        print(f"[lightning] poll: python cloud/lightning_job.py poll --name {session}",
              flush=True)

    elif a.action == "poll":
        print(studio.run(
            f"tail -n 40 ~/run_{session}.log 2>/dev/null; echo '---'; "
            "pgrep -f run_prune_graft >/dev/null && echo RUNNING || echo NOT_RUNNING"),
            flush=True)

    elif a.action == "pull":
        dest = a.dest or ROOT / "data" / "lightning_out" / session
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

    elif a.action == "stop":
        studio.stop()
        print(f"[lightning] stopped {session} (disk persists — resume is free)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
