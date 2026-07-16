#!/usr/bin/env python3
"""AGX Jetson entry — CP 3.3 warm-head Bayesian-Optimization search (single-GPU).

A continuation of the Kaggle run (``kaggle/run.py``) minus the Kaggle plumbing: no git
clone (the code is baked into the image), no ``/kaggle/input`` dataset round-trip (data is
bind-mounted at ``/data``), and the dual-T4 fan-out collapses to the 1-GPU branch — one
``search.bo --seed-list 0,1,2,3,4`` call whose per-seed cache resumes unfinished seeds and
reloads finished ones. There is no 12 h kill on the Jetson, so it runs to completion in one
process; the eval cache makes it crash / reboot-resumable.

The board is a COMPUTE node, not a measurement node: ``search.bo`` reads ``data/lut.jsonl``
(the Orin-Nano-measured latencies) as a static file, so the search still optimizes for Orin
Nano latency. Mount the data plane at ``/data`` (see ``jetson/deploy.sh``)::

    /data/lut.jsonl                   the @640 LUT (resolution-aware catalog)
    /data/gate_best.pt                frozen warm-head donor
    /data/cp33_acc_memo.json          prior fine-tunes (free reuse; ~31% hit rate)
    /data/phase3_nsga2_frontier.json  BO warm-start seeds
    /data/dataset/dataset.yaml        the gate-pose target
    /data/out/cp33_bo_cache_r640.*    the resume shards pulled from Kaggle
    /data/out/cp33_bo.json            the merged DoD verdict (written here)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/workspace/TFM_NAS")
sys.path.insert(0, str(REPO))  # in-process `from search.bo import ...` (script dir != repo root)

# ---- CONFIG (the @640 real-DoD regime — keep identical to the Kaggle run) ----------
DATA = Path(os.environ.get("DATA_DIR", "/data"))
OUT = DATA / "out"
RES = 640                 # @640 LUT + 12.75 ms baseline -> the real DoD regime
T_MAX_MS = 12.75          # hard latency ceiling = measured yolo11n-pose @640 (baseline_anchor.json)
SEEDS = 5
N_INIT = 20
BUDGET = int(os.environ.get("BUDGET", "50"))       # BO budget per seed (decision D2)
CALIBRATE = int(os.environ.get("CALIBRATE", "1"))  # time N real evals first (warms ultralytics)
# MODE="full_finetune" (CP 5.3): a long fine-tune of one graft-interface variant on winner-v1's
# backbone — deploy.sh passes the FT_* selectors through docker -e (see procedure.md
# "CP 5.2 CLOSED" for the variant table).
# MODE="prune_recover" (winner-v2-OFA close, 2026-07-15 pivot): one prune-then-train recovery
# run per container — the AGX replaces the preempting free-tier VMs (all training on the AGX
# is a user decision; procedure.md "Pivot 2026-07-15"). The PG_* selectors mirror
# colab/run_prune_graft.py's compose_recover_cmd contract; that entry is NOT imported here —
# it pip-installs a stack and stages the dataset off Kaggle, both wrong for this baked image.
# Resume is same-VM only: the ckpt in /data/out/prune_recover is safe to reuse HERE because
# the re-prune is deterministic on one platform — never seed it with a Colab/Lightning ckpt.
# Everything else = the CP 3.3 search (default).
MODE = os.environ.get("MODE", "search")   # search | full_finetune | prune_recover | baseline_train
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "100"))
FT_SEEDS = os.environ.get("FT_SEEDS", "0")
FT_NECK = os.environ.get("FT_NECK", "")            # "" | topdown | pan
FT_ADAPTER_INIT = os.environ.get("FT_ADAPTER_INIT", "")  # "" | net2wider
FT_TAG = os.environ.get("FT_TAG", "")              # output suffix, e.g. v3pan
FT_FREEZE_HEAD = os.environ.get("FT_FREEZE_HEAD", "0") == "1"
PG_SPEC = os.environ.get("PG_SPEC", "")            # repo-relative spec (prune/specs/*.json)
PG_RATIOS = os.environ.get("PG_RATIOS", "0.50")    # uniform-ladder fallback when no PG_SPEC
PG_TECH = os.environ.get("PG_TECH", "global_taylor")
PG_ARCH_JSON = os.environ.get("PG_ARCH_JSON", "")  # probe topology (prune/specs/minact_arch.json)
PG_KD = os.environ.get("PG_KD", "1") == "1"        # KD default ON (teacher = the gate donor)
PG_KD_ALPHA = os.environ.get("PG_KD_ALPHA", "1.0")
PG_TEACHER = os.environ.get("PG_TEACHER", "")      # teacher .pt override (Track 2t ladder)
PG_SEED = os.environ.get("PG_SEED", "0")
PG_EPOCHS = os.environ.get("PG_EPOCHS", "100")
PG_BATCH = os.environ.get("PG_BATCH", "16")
PG_LR = os.environ.get("PG_LR", "1e-3")
PG_CKPT_EVERY = os.environ.get("PG_CKPT_EVERY", "10")
PG_IMGSZ = os.environ.get("PG_IMGSZ", "640")   # MCU arms train at 160; rides the artifact tag
PG_NECK = os.environ.get("PG_NECK", "")        # "" | topdown | pan — Phase-5 nano-neck
# MODE="baseline_train" (MCU resolution question, 2026-07-16): the baseline's matched arm —
# COCO-seeded yolo11n-pose gate-trained AT the target resolution, recipe pinned to the deployed
# baseline's own args.yaml. See detect/train_baseline.py.
BT_MODEL = os.environ.get("BT_MODEL", "yolo11n-pose.pt")
BT_IMGSZ = os.environ.get("BT_IMGSZ", "160")
BT_EPOCHS = os.environ.get("BT_EPOCHS", "")    # "" = the recipe's 2000 (patience 300 ends it)
BT_BATCH = os.environ.get("BT_BATCH", "")      # "" = the recipe's 4
# ------------------------------------------------------------------------------------


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO)


def find(name: str):
    hits = sorted(DATA.rglob(name)) if DATA.exists() else []
    return hits[0] if hits else None


def prune_recover_cmd(donor: Path, data_yaml: Path, out_dir: Path) -> str:
    """The ``prune.recover_graft`` invocation for the PG_* config.

    Mirrors ``colab/run_prune_graft.compose_recover_cmd`` (the tested free-tier contract):
    --technique is ALWAYS passed (with a spec it selects the importance metric; the per-stage
    counts stay spec-pinned, so shapes are importance-invariant), PG_SPEC wins over PG_RATIOS,
    and the KD teacher defaults to the gate donor.
    """
    cmd = (f"python3 -m prune.recover_graft --head-weights {donor} --data-yaml {data_yaml} "
           f"--out-dir {out_dir} --device cuda --imgsz {PG_IMGSZ} --batch {PG_BATCH} "
           f"--lr {PG_LR} --epochs {PG_EPOCHS} --seed {PG_SEED} "
           f"--ckpt-every {PG_CKPT_EVERY} --technique {PG_TECH}")
    cmd += f" --ratio-spec {PG_SPEC}" if PG_SPEC else f" --ratios {PG_RATIOS}"
    if PG_ARCH_JSON:
        cmd += f" --arch-json {PG_ARCH_JSON}"
    if PG_NECK:
        cmd += f" --neck {PG_NECK}"
    if PG_KD:
        cmd += f" --teacher {PG_TEACHER or donor} --kd-alpha {PG_KD_ALPHA}"
    return cmd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Locate the mounted data plane robustly; fail loudly (printing the tree) like run.py.
    print("=== /data (2 levels) ===", flush=True)
    if DATA.exists():
        for d in sorted(DATA.iterdir()):
            print("  ", d, flush=True)
            if d.is_dir():
                for sub in sorted(d.iterdir())[:12]:
                    print("      ", sub, flush=True)
    print("=== end tree ===", flush=True)

    lut = find("lut.jsonl")
    head = find("gate_best.pt")
    memo = find("cp33_acc_memo.json")
    yaml_src = find("dataset.yaml")
    frontier = find("phase3_nsga2_frontier.json")
    missing = [n for n, v in (("lut.jsonl", lut), ("gate_best.pt", head),
                              ("dataset.yaml", yaml_src)) if v is None]
    if missing:
        raise SystemExit(f"FATAL: {missing} not found under {DATA} — is the data volume "
                         "mounted (-v <host>:/data)? See the tree above.")

    # Wire the repo's expected paths to the mounted files (mirror run.py:102-106).
    sh(f"rm -rf dataset && ln -s {yaml_src.parent} dataset")   # detect.evaluate.DEFAULT_DATA_YAML
    (REPO / "data").mkdir(exist_ok=True)
    sh(f"ln -sf {lut} data/lut.jsonl")
    if frontier:  # search.bo._load_nsga_frontier reads ROOT/data/phase3_nsga2_frontier.json
        sh(f"ln -sf {frontier} data/phase3_nsga2_frontier.json")

    if MODE == "full_finetune":
        # CP 5.3: one variant's long fine-tune (the winner record is baked into the image).
        # Outputs land beside state/winner_v1 AND are copied to /data/out for the laptop pull.
        import shutil

        winner_dir = REPO / "state" / "winner_v1"
        freeze = "--freeze-head" if FT_FREEZE_HEAD else "--no-freeze-head"
        cmd = (f"python3 -m eval.full_finetune --winner-dir {winner_dir} "
               f"--head-weights {head} {freeze} --device cuda --imgsz 640 --batch 16 "
               f"--epochs {FT_EPOCHS} --seeds {FT_SEEDS}")
        if FT_ADAPTER_INIT:
            cmd += f" --adapter-init {FT_ADAPTER_INIT}"
        if FT_NECK:
            cmd += f" --neck {FT_NECK}"
        if FT_TAG:
            cmd += f" --tag {FT_TAG}"
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True, cwd=REPO).returncode
        suffix = f"_{FT_TAG}" if FT_TAG else ""
        for name in (f"full_finetune{suffix}.json", f"full_finetune{suffix}_weights.pt"):
            src = winner_dir / name
            if src.exists():
                shutil.copy(src, OUT / name)
        out_json = OUT / f"full_finetune{suffix}.json"
        if not out_json.exists():
            raise SystemExit(f"eval.full_finetune produced no {out_json.name} (rc={rc})")
        payload = json.loads(out_json.read_text())
        print(f"[done] {out_json.name}: mean={payload.get('mean')} "
              f"delta_vs_proxy={payload.get('delta_vs_proxy')} "
              f"graft_kwargs={payload.get('graft_kwargs')}", flush=True)
        return

    if MODE == "baseline_train":
        # MCU resolution question: the baseline's matched arm, gate-trained AT BT_IMGSZ with the
        # deployed baseline's own recipe (multi_scale + 2000ep/patience300). Resumable in place.
        bt_out = OUT / f"baseline_r{BT_IMGSZ}"
        cmd = (f"python3 -m detect.train_baseline --model {BT_MODEL} --imgsz {BT_IMGSZ} "
               f"--data-yaml {yaml_src} --out-dir {bt_out} --device 0")
        if BT_EPOCHS:
            cmd += f" --epochs {BT_EPOCHS}"
        if BT_BATCH:
            cmd += f" --batch {BT_BATCH}"
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True, cwd=REPO).returncode
        report = bt_out / f"baseline_{Path(BT_MODEL).stem}_r{BT_IMGSZ}.json"
        if not report.exists():
            raise SystemExit(f"detect.train_baseline produced no {report.name} (rc={rc})")
        p = json.loads(report.read_text())
        print(f"[done] {p['tag']}: map={p['map']:.4f} map50={p['map50']:.4f} "
              f"(vs @640 anchor: {p['delta_vs_640_anchor']:+.4f})", flush=True)
        return

    if MODE == "prune_recover":
        # winner-v2-OFA runs: ledger + tagged artifacts + the same-VM resume ckpt all land in
        # /data/out/prune_recover (deploy.sh --pull → data/cp33_kaggle_out/prune_recover/).
        pg_out = OUT / "prune_recover"
        pg_out.mkdir(parents=True, exist_ok=True)
        cmd = prune_recover_cmd(head, yaml_src, pg_out)
        print("+", cmd, flush=True)
        # taylor importance forwards+backwards the full graft at batch×640² — same
        # fragmentation headroom the free-tier entry needed.
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        rc = subprocess.run(cmd, shell=True, cwd=REPO, env=env).returncode
        report = pg_out / "recover_graft.json"
        if not report.exists():
            raise SystemExit(f"prune.recover_graft produced no {report.name} (rc={rc})")
        for row in json.loads(report.read_text()).get("rows", []):
            print(f"[done] {row.get('technique')}/{row.get('arch_tag')}/ratio={row.get('ratio')} "
                  f"seed={row.get('seed')} params={row.get('params'):,} "
                  f"map={row.get('map'):.4f} map50={row.get('map50'):.4f} "
                  f"kd={'on' if row.get('kd') else 'off'}", flush=True)
        return

    common = (f"--device cuda --imgsz 640 --res {RES} --lut {lut} "
              f"--head-weights {head} --freeze-head --t-max-ms {T_MAX_MS}")
    if memo:  # prior fine-tunes are accuracy-only (imgsz-fixed) -> valid across resolutions
        common += f" --acc-memo {memo}"
        print(f"[acc-memo] attached {memo}", flush=True)

    if CALIBRATE:  # report real per-eval wall-clock on THIS board before committing days
        sh(f"python3 -m search.bo --calibrate {CALIBRATE} {common}")

    cache = OUT / f"cp33_bo_cache_r{RES}"   # shard names match the pulled Kaggle files
    out = OUT / "cp33_bo.json"

    # How much each seed still owes (from the restored caches) -> a continuity check.
    from search.bo import seed_remaining_evals
    remaining = {s: seed_remaining_evals(cache, s, BUDGET) for s in range(SEEDS)}
    ndone = sum(1 for s in remaining if remaining[s] == 0)
    print(f"[resume] {ndone}/{SEEDS} seeds complete; remaining evals/seed: {remaining}",
          flush=True)

    # Single GPU: one call over all seeds; the cache resumes the unfinished ones and reloads
    # the finished ones instantly. rc=1 is a valid DoD-FAIL verdict (data in the JSON), so
    # only a missing output file is fatal.
    seed_list = ",".join(str(s) for s in range(SEEDS))
    cmd = (f"python3 -m search.bo --seed-list {seed_list} "
           f"--budget {BUDGET} --n-init {N_INIT} {common} --out {out} --cache {cache}")
    print("+", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=REPO).returncode
    if not out.exists():
        raise SystemExit(f"search.bo produced no results (rc={rc})")

    payload = json.loads(out.read_text())
    done = payload.get("complete")
    nxt = ("COMPLETE — DoD final" if done else
           "PARTIAL — re-run `deploy.sh --run` to resume (the cache continues)")
    print(f"[done] {out.name}: complete={done} passes={payload.get('passes')} -> {nxt}",
          flush=True)


if __name__ == "__main__":
    main()
