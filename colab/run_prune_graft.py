#!/usr/bin/env python3
"""Remote entry — winner-v2-OFA wave runs on a free-tier GPU VM (Colab CLI / Lightning / notebook).

Kaggle GPU quota is exhausted for the week (2026-07-13), so the graft prune-then-train runs
move to whatever free GPU answers: a **Colab CLI** session (``cloud/colab_job.sh``), a
**Lightning studio** (``cloud/lightning_job.py``), or a pasted notebook cell. This entry is
platform-agnostic: creds → NAS stack (pinned to the resident CUDA torch) → gate dataset off
Kaggle (Datasets API needs no GPU quota) → OFA checkpoint → ``prune.recover_graft`` with the
requested config. The train loop checkpoints every 10 epochs and auto-resumes (same
``--out-dir`` + config ⇒ a VM death costs ≤ 10 epochs).

    python colab/run_prune_graft.py --out-dir /content/tfm_out \
        --spec prune/specs/v2_act292.json --seed 0            # v2_act292_kd
    python colab/run_prune_graft.py --out-dir /content/tfm_out \
        --ratios 0.50 --technique global_taylor --seed 0      # r50_gtay_kd
    python colab/run_prune_graft.py --out-dir /content/tfm_out \
        --arch-json prune/specs/minact_arch.json --spec prune/specs/u30.json  # min-act probe

KD is ON by default (teacher = the staged gate donor; measured +0.85 on this training
state); ``--teacher-pt`` swaps in a bigger gate-trained teacher (Track 2t), ``--no-kd``
disables. Secrets: ``<--secrets-root>/secrets/{access_token,kaggle_username}`` (the repo
contract; the drivers upload them) or KAGGLE_USERNAME/KAGGLE_KEY env vars.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "colab"))
import colab_common as C  # noqa: E402

# onnx: torch.onnx.export's onnxscript hook. kaggle: the dataset-staging CLI. Both are bundled
# in Colab's base image but NOT on a bare Lightning studio (smokes caught each 2026-07-13).
STACK = ("'ofa==0.1.0.post202307202001' 'ultralytics>=8.3' 'torch-pruning>=1.4,<2' "
         "onnx kaggle")


def compose_recover_cmd(a: argparse.Namespace, *, donor: Path, data_yaml: Path,
                        python: str = sys.executable) -> str:
    """The exact ``prune.recover_graft`` invocation for this config (pure — tested)."""
    cmd = (f"{python} -m prune.recover_graft --head-weights {donor} "
           f"--data-yaml {data_yaml} --out-dir {a.out_dir} --device {a.device} "
           f"--imgsz 640 --batch {a.batch} --epochs {a.epochs} --seed {a.seed} "
           f"--ckpt-every {a.ckpt_every}")
    if a.max_steps is not None:
        cmd += f" --max-steps {a.max_steps}"
    # --technique is ALWAYS passed: with a spec it selects the importance metric
    # (global_taylor picks better channels; per-stage counts stay spec-pinned, so shapes
    # are importance-invariant) — without it, recover_graft would default to uniform/l2.
    cmd += f" --technique {a.technique}"
    if a.spec:
        cmd += f" --ratio-spec {a.spec}"
    else:
        cmd += f" --ratios {a.ratios}"
    if a.arch_json:
        cmd += f" --arch-json {a.arch_json}"
    if a.kd:
        teacher = a.teacher_pt or donor
        cmd += f" --teacher {teacher} --kd-alpha {a.kd_alpha}"
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="winner-v2-OFA graft run on a free GPU VM.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="durable output dir (Drive folder / studio disk / VM dir the "
                         "driver polls) — the resume ckpt lives here")
    ap.add_argument("--secrets-root", type=Path, default=Path(C.DRIVE_DEFAULT),
                    help="dir holding secrets/{access_token,kaggle_username} "
                         "(default: the mounted-Drive contract)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="dataset staging dir (default /content/kagdata or ~/kagdata)")
    ap.add_argument("--spec", type=str, default=None,
                    help="repo-relative allocation spec (prune/specs/v2_act292.json | "
                         "v2_act273.json | u30.json)")
    ap.add_argument("--ratios", type=str, default="0.50",
                    help="uniform-ladder fallback when no --spec (e.g. r50_gtay)")
    ap.add_argument("--technique", type=str, default="global_taylor")
    ap.add_argument("--arch-json", type=str, default=None,
                    help="repo-relative probe arch (prune/specs/minact_arch.json)")
    kd = ap.add_mutually_exclusive_group()
    kd.add_argument("--kd", dest="kd", action="store_true", default=True)
    kd.add_argument("--no-kd", dest="kd", action="store_false")
    ap.add_argument("--kd-alpha", type=float, default=1.0)
    ap.add_argument("--teacher-pt", type=str, default=None,
                    help="override KD teacher .pt (Track 2t ladder; default = gate donor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke tests — the S0 2-epoch dry run)")
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()

    data_root = a.data_root or (Path("/content/kagdata") if Path("/content").exists()
                                else Path.home() / "kagdata")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    user = C.ensure_kaggle_credentials(a.secrets_root)
    C.pin_torch_and_install(STACK)
    staged = C.stage_kaggle_dataset(user, data_root)
    wired = C.wire_repo_data(REPO, staged)
    C.download_ofa(REPO)

    donor = wired["donor"]
    if donor is None:
        raise SystemExit("FATAL: gate_best.pt missing from the staged dataset")
    cmd = compose_recover_cmd(a, donor=donor, data_yaml=wired["yaml"])
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=REPO).returncode

    report = a.out_dir / "recover_graft.json"
    if report.exists():
        payload = json.loads(report.read_text())
        for row in payload.get("rows", []):
            print(f"[done] tag-artifacts in {a.out_dir}  params={row.get('params'):,} "
                  f"map={row.get('map'):.4f}  (Δ vs 0.841 anchor "
                  f"{row.get('delta_map_vs_donor'):+.4f})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
