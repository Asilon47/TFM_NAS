#!/usr/bin/env python3
"""Kaggle kernel entry — CP 3.3/3.4 warm-head architecture search.

Pushed by ``kaggle/push.sh`` and run on Kaggle GPU. It clones the (public) repo,
installs the NAS stack *without* disturbing Kaggle's torch, wires the attached
data Dataset (gate dataset + LUT + NSGA-II seeds + the frozen gate head donor), fetches
the SHA-pinned OFA checkpoint, then runs the search — ``search.<METHOD>``, where
``METHOD`` selects the proposer: ``bo`` (CP 3.3 Bayesian optimization, closed) or
``tpe`` (CP 3.4 Optuna MOTPE fallback). Both share the DoD, oracle, ceiling, and cache.

The CONFIG block is the full 5-seed DoD, *resumable across Kaggle sessions*: each commit
stops starting new evals after ``DEADLINE_H`` hours — a clean boundary under Kaggle's 12 h
kill — having appended per-seed caches to ``/kaggle/working``. Between sessions
``kaggle/push.sh --resume`` versions those caches into the ``tfm-nas-cp33-bo-cache`` Dataset
(a notebook can't read its OWN output as input); re-running restores them and continues,
until the merged JSON reports ``complete: true``. Outputs land in ``/kaggle/working`` and
are pulled by ``push.sh --pull``.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---- CONFIG (the full 5-seed DoD; resumable across sessions) ----------------
REPO_URL   = "https://github.com/Asilon47/TFM_NAS.git"
DATASET    = "tfm-nas-gate-pose"  # attached Kaggle Dataset slug (no <user>/ prefix)
MODE       = "full_finetune"      # search | verify_winner (CP 3.5 DoD) | denoise (CP 3.5
#   re-score) | full_finetune (side experiment; see eval/full_finetune.py) | graft_ablate
#   (CP 5.2 ablation, CLOSED) | honest_search (Phase 3b: re-search under the honest
#   Stage-0 cost model — backbone sum <= 7.16ms == e2e <= the baseline's 12.75ms;
#   procedure.md "Phase 3b LAUNCHED")
# Phase-3b honest-search knobs (MODE="honest_search"):
HS_T_MAX   = 7.16                 # (12.75 - 0.926 - 3.837) / 1.115 — the honest sum ceiling
HS_SEEDS   = "0,1"
HS_BUDGET  = 30
HS_N_INIT  = 15
METHOD     = "tpe"                 # search proposer: "bo" (CP 3.3, closed) or "tpe" (CP 3.4).
#   Same DoD/oracle/ceiling; only the sampler differs. TPE writes '.seed{s}.tpe.jsonl' shards
#   and REUSES CP 3.3's cached '.seed{s}.rs.jsonl' random control (free), so a TPE session
#   only fine-tunes TPE's own novel proposals (memo-assisted -> much cheaper than CP 3.3).
OUT_NAME   = {"bo": "cp33_bo.json", "tpe": "cp34_tpe.json"}[METHOD]
RES        = 640                  # @640 LUT + 12.75ms baseline landed -> the real DoD regime
# Hard latency ceiling, matched to RES (T_max and the LUT regime must agree). The @640
# baseline anchor (data/baseline_anchor.json) measured yolo11n-pose at 12.75 ms — tighter
# than the 60-FPS 16.7 ms target, so T_max=min(.)=12.75 at the deploy resolution. @224 is
# the provisional proving regime (a different, much smaller latency scale); it keeps its
# original 16.7 so its already-run RES-namespaced caches stay valid.
T_MAX_MS   = 12.75 if RES == 640 else 16.7
CALIBRATE  = 1                    # time N real evals first (also warms ultralytics)
SEEDS      = 5                    # the 5-seed DoD
BUDGET     = 50                   # BO budget per seed (decision D2)
N_INIT     = 20                   # initial-design size
DEADLINE_H = 10.5                 # stop new evals after this many h (clean resumable boundary)
# CP 3.5 (MODE="verify_winner"): reload state/winner_v1/winner.json + reproduce its cached proxy
# mAP over VERIFY_SEEDS fresh warm-head fine-tunes; PASS iff the mean is within VERIFY_BAND.
VERIFY_SEEDS = "1,2,3"            # 3-seed averaging for the winner check (PROJECT_PLAN.md:598)
VERIFY_BAND  = 0.020             # reproducibility noise band on |mean(fresh) − cached| mAP
# MODE="full_finetune": a longer, un-frozen fine-tune of winner-v1 to see its (proxy-protocol)
# full-train mAP. Single-seed by default — this is the "quick" side experiment, not a DoD.
FULL_FT_EPOCHS      = 100         # matches eval.proxy_rank's own full_epochs precedent
FULL_FT_SEEDS       = "0"
FULL_FT_FREEZE_HEAD = False       # warm-start but keep training the head (default in the script)
# CP 5.3 variant list for MODE="full_finetune": (tag, neck, adapter_init); "" = none.
# One variant per T4 when 2 GPUs are visible (honest_search's split pattern); the bare
# winner-v1 control is NOT re-run — its 0.841 (full_finetune.json, 2026-07-05) is the anchor.
FULL_FT_VARIANTS = [
    ("v3pan", "pan", ""),
    ("v2topdown", "topdown", ""),
]
# Dense-family arm (plan amendment 2026-07-07, user decision A+B1+B2; three Kaggle accounts
# run campaigns in parallel — push each with KACCT=N KMODE=<mode> bash kaggle/push.sh):
# MODE="prune_baseline" — CP 6.2-B control arm: DepGraph ladder on the gate-trained yolo11n
#   donor (gate_best.pt IS the 0.877 baseline), recover, export deploy ONNX per point.
# Round 2 (2026-07-08): the Nano canary showed the pruned baseline is ~25% FASTER than the
# depthwise graft-loser, so both dense arms map their frontier FINER. Extended prune ratios
# (new points around the done 0.15/0.30/0.45); fresh Kaggle session ⇒ its own output dir.
PB_RATIOS = "0.10,0.20,0.35,0.55"
PB_EPOCHS = 50
# Pruning-as-search knobs (CP 6.2-G program): technique in {uniform, global_l2, global_taylor},
# PB_ITER>1 = iterative with interleaved recovery, PB_SEED != 0 = de-noise rerun (fresh out dir).
PB_TECH = "uniform"
PB_ITER = 1
PB_SEED = 0
# MODE="dense_scaling" — yolo11-pose scaling grid, stock recipe, from scratch, one candidate
#   subset per T4; latencies measured later on the Nano. DS_WAVE selects the wave (2 = the
#   finer width sweep; depth is a dead knob below n, see CP 3c.1). DS_SEED != 0 = de-noise
#   rerun (row files are not seed-namespaced → the runner suffixes the out dir).
DS_EPOCHS = 100
DS_WAVE = "2"
DS_SEED = 0
# MODE="prune_graft" — CP 6.2-G (graft arm): train the pruned graft to its recovered pose mAP.
#   Self-contained: the gate donor warm-starts the head, no trained-graft input needed. One
#   ratio per T4 when two are visible. Technique ladder: PG_TECH/PG_ITER as above; PG_INDEX
#   non-empty = prune denoise_candidates[PG_INDEX] instead of the winner (G1 topology probe;
#   the Stage-0-benched fallbacks are idx 3 and 11).
PG_RATIOS = "0.50,0.60"
PG_EPOCHS = 100
PG_TECH = "global_taylor"
PG_ITER = 1
PG_SEED = 0
PG_INDEX = ""
# PG_SPECS: comma list of repo-relative HALP-lite specs (prune/specs/halp_*.json) — when set,
# they replace the ratio ladder (one spec per T4; --ratio-spec overrides --ratios/--technique).
PG_SPECS = ""
# -----------------------------------------------------------------------------


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    START = time.time()
    work = Path("/kaggle/working")
    repo = work / "TFM_NAS"

    # 1. code: shallow clone (public; GITHUB_TOKEN Kaggle secret supports a private repo)
    token = os.environ.get("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
    if not repo.exists():
        sh(f"git clone --depth 1 {url} {repo}")
    os.chdir(repo)
    sys.path.insert(0, str(repo))  # run.py executes as /kaggle/src/script.py, so the cloned
    #   repo is the cwd but NOT on sys.path; in-process imports (search.bo) need it explicitly.
    #   (subprocess `python -m search.bo` works without this because -m adds cwd to sys.path.)

    # 2. deps: pin Kaggle's torch so botorch/ultralytics never upgrade it (the same
    #    discipline the .venv uses for the 2.3.1 pin) — then add the NAS + BO stack.
    torch_ver = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"]).decode().strip()
    constraint = work / "constraints.txt"
    constraint.write_text(f"torch=={torch_ver}\n")
    sh(f"{sys.executable} -m pip install -q --constraint {constraint} "
       "'ofa==0.1.0.post202307202001' 'ultralytics>=8.3' gdown botorch gpytorch optuna "
       "'torch-pruning>=1.4,<2'")

    # 3. data: locate the attached Dataset ROBUSTLY. Kaggle's mount path is not
    #    guaranteed to be /kaggle/input/<slug>, so search /kaggle/input and fail
    #    loudly (printing the tree) rather than silently skip a missing file.
    input_root = Path("/kaggle/input")
    print("=== /kaggle/input (2 levels) ===", flush=True)
    if input_root.exists():
        for d in sorted(input_root.iterdir()):
            print("  ", d, flush=True)
            if d.is_dir():
                for sub in sorted(d.iterdir())[:25]:
                    print("      ", sub, flush=True)
    else:
        print("  (/kaggle/input missing — no dataset attached)", flush=True)
    print("=== end tree ===", flush=True)

    def find(name: str):
        hits = sorted(input_root.rglob(name)) if input_root.exists() else []
        return hits[0] if hits else None

    lut_src = find("lut.jsonl")
    frontier_src = find("phase3_nsga2_frontier.json")
    memo_src = find("cp33_acc_memo.json")                    # prior accs (cross-res compute reuse)
    head = find("gate_best.pt")                              # warm-head donor (frozen)
    yaml_src = find("dataset.yaml")                          # the gate-pose data root
    missing = [n for n, v in (("lut.jsonl", lut_src), ("gate_best.pt", head),
                              ("dataset.yaml", yaml_src)) if v is None]
    if missing:
        raise SystemExit(f"FATAL: {missing} not found under /kaggle/input — is the "
                         f"'{DATASET}' dataset attached? See the tree above.")

    sh(f"rm -rf dataset && ln -s {yaml_src.parent} dataset")  # detect.evaluate.DEFAULT_DATA_YAML
    Path("data").mkdir(exist_ok=True)
    sh(f"ln -sf {lut_src} data/lut.jsonl")
    if frontier_src:
        sh(f"ln -sf {frontier_src} data/phase3_nsga2_frontier.json")

    # 4. OFA supernet checkpoint (internet on; SHA-verified in supernet/download_ofa.py)
    sh(f"{sys.executable} -m supernet.download_ofa")

    # 4.7 CP 3.5 winner verification: a single warm-head re-fine-tune of the serialized α*, not a
    #     search. Reuses the head donor + dataset + checkpoint wired above; writes repro.json
    #     (+ weights.pt) beside state/winner_v1/ and copies both to /kaggle/working for --pull.
    if MODE == "verify_winner":
        winner_dir = repo / "state" / "winner_v1"
        cmd = (f"{sys.executable} -m eval.verify_winner --winner-dir {winner_dir} "
               f"--head-weights {head} --freeze-head --device cuda --imgsz 640 "
               f"--batch 16 --seeds {VERIFY_SEEDS} --band {VERIFY_BAND}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode  # rc=1 is a valid DoD-FAIL verdict
        for name in ("repro.json", "weights.pt"):
            src = winner_dir / name
            if src.exists():
                shutil.copy(src, work / name)
        if not (work / "repro.json").exists():
            raise SystemExit(f"eval.verify_winner produced no repro.json (rc={rc})")
        payload = json.loads((work / "repro.json").read_text())
        print(f"[done] repro.json: passes={payload.get('passes')} "
              f"cached={payload.get('cached_acc')} fresh_mean={payload.get('fresh_mean')} "
              f"delta={payload.get('delta')}", flush=True)
        return

    # 4.75 Side experiment: a longer, un-frozen fine-tune of winner-v1 (NOT a checkpoint, NOT
    #      Phase 8 distillation — see eval/full_finetune.py's module docstring for the protocol
    #      caveat). Reuses the head donor + dataset + checkpoint wired above; writes
    #      full_finetune.json (+ full_finetune_weights.pt) beside state/winner_v1/ and copies
    #      both to /kaggle/working for --pull.
    if MODE == "full_finetune":
        winner_dir = repo / "state" / "winner_v1"
        freeze_flag = "--freeze-head" if FULL_FT_FREEZE_HEAD else "--no-freeze-head"

        def ft_cmd(tag: str, neck: str, adapter_init: str) -> str:
            cmd = (f"{sys.executable} -m eval.full_finetune --winner-dir {winner_dir} "
                   f"--head-weights {head} {freeze_flag} --device cuda --imgsz 640 --batch 16 "
                   f"--epochs {FULL_FT_EPOCHS} --seeds {FULL_FT_SEEDS}")
            if adapter_init:
                cmd += f" --adapter-init {adapter_init}"
            if neck:
                cmd += f" --neck {neck}"
            if tag:
                cmd += f" --tag {tag}"
            return cmd

        ngpu = int(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
        ).decode().strip() or "0")
        if ngpu >= 2 and len(FULL_FT_VARIANTS) > 1:
            procs = []
            for i, (tag, neck, ainit) in enumerate(FULL_FT_VARIANTS):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ngpu))
                print(f"+ [gpu{i % ngpu}] variant {tag or 'plain'}", flush=True)
                procs.append(subprocess.Popen(ft_cmd(tag, neck, ainit), shell=True, env=env))
                time.sleep(10)  # guard ultralytics first-touch race
            for pr in procs:
                pr.wait()
        else:
            for tag, neck, ainit in FULL_FT_VARIANTS:
                cmd = ft_cmd(tag, neck, ainit)
                print("+", cmd, flush=True)
                subprocess.run(cmd, shell=True)

        missing = []
        for tag, _neck, _ainit in FULL_FT_VARIANTS:
            suffix = f"_{tag}" if tag else ""
            for name in (f"full_finetune{suffix}.json", f"full_finetune{suffix}_weights.pt"):
                src = winner_dir / name
                if src.exists():
                    shutil.copy(src, work / name)
            out_json = work / f"full_finetune{suffix}.json"
            if not out_json.exists():
                missing.append(out_json.name)
                continue
            payload = json.loads(out_json.read_text())
            print(f"[done] {out_json.name}: proxy={payload.get('proxy_acc')} "
                  f"full_mean={payload.get('mean')} delta_vs_proxy={payload.get('delta_vs_proxy')}",
                  flush=True)
        if missing:
            raise SystemExit(f"eval.full_finetune produced no {', '.join(missing)}")
        return

    # 4.76 CP 6.2-B (dense-family arm): the pruned-BASELINE control ladder. Single process —
    #      3 sequential prune->recover->export points (~2-3 h); gate_best.pt is the donor.
    if MODE == "prune_baseline":
        pb_suffix = (f"_{PB_TECH}" if PB_TECH != "uniform" else "") + \
                    (f"_it{PB_ITER}" if PB_ITER > 1 else "") + \
                    (f"_s{PB_SEED}" if PB_SEED != 0 else "")
        out_dir = work / f"prune_baseline{pb_suffix}"
        cmd = (f"{sys.executable} -m prune.prune_baseline --donor {head} "
               f"--ratios {PB_RATIOS} --epochs {PB_EPOCHS} --device cuda "
               f"--technique {PB_TECH} --iterative-steps {PB_ITER} --seed {PB_SEED} "
               f"--imgsz 640 --batch 16 --out-dir {out_dir}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode
        rep = out_dir / "prune_baseline.json"
        if not rep.exists():
            raise SystemExit(f"prune.prune_baseline produced no report (rc={rc})")
        payload = json.loads(rep.read_text())
        print(f"[done] prune_baseline.json: donor_map={payload['donor'].get('map')} rows="
              f"{[(r['ratio'], round(r['map'], 4)) for r in payload['rows']]}", flush=True)
        return

    # 4.78 CP 6.2 (graft arm): train the pruned winner graft to its recovered mAP — the accuracy
    #      half of the prune-graft screen (models/screen_prune_graft/). One ratio per T4 when two
    #      are visible; self-contained (gate donor warm-starts the head, no trained-graft input).
    if MODE == "prune_graft":
        from prune.recover_graft import run_tag  # repo already on sys.path (step 1)

        base_out = work / "recover_graft"
        ratios = [r for r in PG_RATIOS.split(",") if r]
        pg_extra = (f"--technique {PG_TECH} --iterative-steps {PG_ITER} --seed {PG_SEED}"
                    + (f" --index {PG_INDEX}" if PG_INDEX != "" else ""))
        pg_prefix = f"idx{PG_INDEX}_" if PG_INDEX != "" else ""

        def pg_cmd(ratio_csv: str, sub: str, spec: str = "") -> str:
            spec_arg = f" --ratio-spec {repo}/{spec}" if spec else ""
            return (f"{sys.executable} -m prune.recover_graft "
                    f"--winner-dir {repo}/state/winner_v1 --head-weights {head} "
                    f"--ratios {ratio_csv} --epochs {PG_EPOCHS} {pg_extra}{spec_arg} "
                    f"--device cuda --imgsz 640 --batch 16 --out-dir {base_out}_{sub}")

        ngpu = int(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
        ).decode().strip() or "0")
        specs = [s for s in PG_SPECS.split(",") if s]
        if specs:
            # Wave B: one HALP spec per T4 (spec overrides ratios/technique in recover_graft).
            procs = []
            for i, s in enumerate(specs):
                sub = Path(s).stem + (f"_s{PG_SEED}" if PG_SEED != 0 else "")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(ngpu, 1)))
                print(f"+ [gpu{i % max(ngpu, 1)}] spec {s} → {sub}", flush=True)
                procs.append(subprocess.Popen(pg_cmd(PG_RATIOS, sub, spec=s),
                                              shell=True, env=env))
                time.sleep(10)
            for pr in procs:
                pr.wait()
        elif ngpu >= 2 and len(ratios) > 1:
            procs = []
            for i, r in enumerate(ratios):
                sub = pg_prefix + run_tag(float(r), technique=PG_TECH,
                                          iterative_steps=PG_ITER, seed=PG_SEED)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ngpu))
                print(f"+ [gpu{i % ngpu}] ratio {r} → {sub}", flush=True)
                procs.append(subprocess.Popen(pg_cmd(r, sub), shell=True, env=env))
                time.sleep(10)  # guard ultralytics first-touch race
            for pr in procs:
                pr.wait()
        else:
            subprocess.run(pg_cmd(PG_RATIOS, pg_prefix + "all"), shell=True)

        found = []
        for sub_dir in sorted(work.glob("recover_graft*")):
            rep = sub_dir / "recover_graft.json"
            if not rep.exists():
                continue
            shutil.copy(rep, work / f"{sub_dir.name}.json")
            found.append(rep)
            payload = json.loads(rep.read_text())
            anchor = payload["donor"].get("map")
            for row in payload["rows"]:
                print(f"[done] {sub_dir.name} ratio={row['ratio']} map={round(row['map'], 4)} "
                      f"(anchor {anchor}, {row.get('delta_map_vs_donor'):+.4f})", flush=True)
        if not found:
            raise SystemExit("prune.recover_graft produced no report")
        return

    # 4.77 Phase 3c wave 1 (dense-family arm): yolo11-pose scaling grid — stripe the wave
    #      across the two T4s (per-tag row files make the split coordination-free), then
    #      assemble the report in one --report-only pass.
    if MODE == "dense_scaling":
        from search.dense_family import WAVES, wave_tags  # repo already on sys.path (step 1)

        out_dir = work / ("dense_scaling" + (f"_s{DS_SEED}" if DS_SEED != 0 else ""))
        out_dir.mkdir(exist_ok=True)
        base_cmd = (f"{sys.executable} -m search.dense_family --data dataset/dataset.yaml "
                    f"--epochs {DS_EPOCHS} --imgsz 640 --batch 16 --wave {DS_WAVE} "
                    f"--seed {DS_SEED} --out-dir {out_dir}")
        tags = wave_tags(WAVES[DS_WAVE])
        ngpu = int(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
        ).decode().strip() or "0")
        if ngpu >= 2:
            procs = []
            for g in range(2):
                subset = ",".join(tags[g::2])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                print(f"+ [gpu{g}] tags {subset}", flush=True)
                procs.append(subprocess.Popen(f"{base_cmd} --only {subset} --device 0",
                                              shell=True, env=env))
                time.sleep(10)  # guard ultralytics first-touch race
            for pr in procs:
                pr.wait()
        else:
            sh(f"{base_cmd} --device 0")
        sh(f"{base_cmd} --report-only")
        rep = out_dir / "dense_scaling.json"
        if not rep.exists():
            raise SystemExit("search.dense_family produced no dense_scaling.json")
        payload = json.loads(rep.read_text())
        print(f"[done] dense_scaling.json: "
              f"{[(r['tag'], r['params'], round(r['map'], 4)) for r in payload['rows']]}",
              flush=True)
        return

    # 4.8 CP 3.5 de-noise: re-score the pinned top-K candidates at fresh seeds to remove the
    #     single-seed winner's curse. Resumable via a per-(arch,seed) cache round-tripped through
    #     /kaggle/working (Kaggle wipes it per commit; push.sh --resume versions it if partial).
    if MODE == "denoise":
        cand = repo / "state" / "winner_v1" / "denoise_candidates.json"
        cache = work / "denoise_cache.jsonl"
        prior = sorted(input_root.rglob("denoise_cache*.jsonl")) if input_root.exists() else []
        for src in prior:
            if not cache.exists():
                shutil.copy(src, cache)
                print(f"[resume] restored denoise cache {src.name}", flush=True)
        out_dn = work / "denoise.json"
        cmd = (f"{sys.executable} -m search.denoise --candidates {cand} "
               f"--head-weights {head} --freeze-head --device cuda --imgsz 640 --batch 16 "
               f"--seeds {VERIFY_SEEDS} --cache {cache} --out {out_dn}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode
        if not out_dn.exists():
            raise SystemExit(f"search.denoise produced no denoise.json (rc={rc})")
        print("[done] denoise.json written — pull it, then re-select on CPU via "
              "search.denoise.select_denoised.", flush=True)
        return

    # 4.85 CP 5.2 graft-interface ablation: V0/V1/V2 (+V3 auto-gated by the 1σ rule) × 3 fresh
    #      seeds on winner-v1's backbone — the Phase-5 accuracy-seam experiment (procedure.md
    #      "Plan pivot"). Same oracle as the de-noise re-score; resumable per-(variant,seed)
    #      cache round-tripped exactly like the denoise cache.
    if MODE == "graft_ablate":
        cache = work / "graft_ablate_e5_r640.jsonl"
        prior = (sorted(input_root.rglob("graft_ablate_e5_r640*.jsonl"))
                 if input_root.exists() else [])
        for src in prior:
            if not cache.exists():
                shutil.copy(src, cache)
                print(f"[resume] restored graft-ablate cache {src.name}", flush=True)
        out_ab = work / "graft_ablate.json"
        cmd = (f"{sys.executable} -m eval.graft_ablate --winner-dir {repo}/state/winner_v1 "
               f"--head-weights {head} --device cuda --imgsz 640 --batch 16 --epochs 5 "
               f"--seeds {VERIFY_SEEDS} --cache {cache} --out {out_ab}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode
        if not out_ab.exists():
            raise SystemExit(f"eval.graft_ablate produced no graft_ablate.json (rc={rc})")
        payload = json.loads(out_ab.read_text())
        for vname, v in payload.get("variants", {}).items():
            print(f"[done] {vname}: mean={v.get('mean')} std={v.get('std')} "
                  f"gates={v.get('gates')}", flush=True)
        print(f"[done] v1-v0={payload.get('v1_vs_v0_delta')} "
              f"v2-v1={payload.get('v2_vs_v1_delta')} "
              f"v3_warranted={payload.get('v3_warranted')}", flush=True)
        return

    # 4.9 Phase-3b HONEST-CEILING RE-SEARCH (user-directed 2026-07-07): Stage 0 gave the
    #     honest cost model e2e ~= 1.115*sum + 0.93 + 3.84, so beating the baseline needs a
    #     backbone sum <= 7.16ms — a 0.31ms band above the space's floor (min corner 6.85).
    #     Uniform sampling starves there, so BO warm-starts from the pinned band seeds
    #     (state/honest_search/, 113 stratified feasible archs); the incumbent-mutation pools
    #     keep proposals band-local. The RS control WILL starve (bounded attempts) — the
    #     HV-vs-random number is not a claim of this run. Own cache namespace; the CP 3.3/3.4
    #     DoD caches are untouched.
    if MODE == "honest_search":
        sh(f"ln -sf {repo}/state/honest_search/nsga2_seeds_tmax716.json "
           "data/phase3_nsga2_frontier.json")   # override the DoD frontier symlink
        cache = work / "hs_bo_cache_r640"
        restored_hs = 0
        for src in (sorted(input_root.rglob("hs_bo_cache_r640*.jsonl"))
                    if input_root.exists() else []):
            dst = work / src.name
            if not dst.exists():
                shutil.copy(src, dst)
                restored_hs += 1
        print(f"[resume] restored {restored_hs} honest-search shard(s)", flush=True)
        out_hs = work / "phase3b_honest_search.json"
        common_hs = (f"--device cuda --imgsz 640 --res {RES} --lut data/lut.jsonl "
                     f"--head-weights {head} --freeze-head --t-max-ms {HS_T_MAX}")
        if memo_src:
            common_hs += f" --acc-memo {memo_src}"
        deadline_s = max(600, int(DEADLINE_H * 3600 - (time.time() - START)))

        # --- Multi-GPU splitting for honest_search ---
        ngpu = int(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
        ).decode().strip() or "0")

        hs_seed_list = [int(s) for s in HS_SEEDS.split(",")]

        if ngpu >= 2 and len(hs_seed_list) > 1:
            from search.bo import assign_seeds_to_gpus, seed_remaining_evals
            remaining = {s: seed_remaining_evals(cache, s, HS_BUDGET, method="bo")
                         for s in hs_seed_list}
            assignments = assign_seeds_to_gpus(remaining, ngpu)
            procs, parts = [], []
            for g, seeds_g in enumerate(assignments):
                if not seeds_g:
                    continue
                part = work / f"phase3b_honest_search.part{g}.json"
                parts.append(part)
                sl = ",".join(map(str, seeds_g))
                cmd = (f"{sys.executable} -m search.bo --seed-list {sl} --budget {HS_BUDGET} "
                       f"--n-init {HS_N_INIT} --deadline-s {deadline_s} {common_hs} "
                       f"--out {part} --cache {cache}")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                print(f"+ [gpu{g}] seeds {seeds_g}", flush=True)
                procs.append(subprocess.Popen(cmd, shell=True, env=env))
                time.sleep(10) # guard ultralytics first-touch race

            for pr in procs:
                pr.wait()

            sh(f"{sys.executable} -m search.bo --merge {' '.join(map(str, parts))} --out {out_hs}")
        else:
            # Fallback to single GPU execution
            cmd = (f"{sys.executable} -m search.bo --seed-list {HS_SEEDS} --budget {HS_BUDGET} "
                   f"--n-init {HS_N_INIT} --deadline-s {deadline_s} {common_hs} "
                   f"--out {out_hs} --cache {cache}")
            print("+", cmd, flush=True)
            rc = subprocess.run(cmd, shell=True).returncode

        if not out_hs.exists():
            raise SystemExit(f"search.bo produced no {out_hs.name}")
        payload = json.loads(out_hs.read_text())
        print(f"[done] {out_hs.name}: complete={payload.get('complete')} — honest-ceiling "
              f"frontier under sum<={HS_T_MAX}ms; de-noise the top-K before any pick.",
              flush=True)
        return

    # 4.5 resume: Kaggle wipes /kaggle/working between commits, so the eval caches must
    #     round-trip through the persistent /kaggle/input plane. A notebook can't attach
    #     its OWN output as input, so the caches live in the tfm-nas-cp33-bo-cache Dataset
    #     (versioned between sessions by `kaggle/push.sh --resume`, attached via
    #     kernel-metadata). Restore its shards so the workers continue. First session: none.
    restored = 0
    for src in (sorted(input_root.rglob("cp33_bo_cache_r*.jsonl"))
                if input_root.exists() else []):
        dst = work / src.name
        if not dst.exists():            # never clobber a shard this session already wrote
            shutil.copy(src, dst)
            restored += 1
    print(f"[resume] restored {restored} eval-cache shard(s) from prior output", flush=True)

    # 5. search: timed calibration first (de-risks the budget), then the search run.
    common = (f"--device cuda --imgsz 640 --res {RES} --lut data/lut.jsonl "
              f"--head-weights {head} --freeze-head --t-max-ms {T_MAX_MS}")
    if memo_src:  # reuse prior fine-tunes (acc is imgsz-fixed, so resolution-independent)
        common += f" --acc-memo {memo_src}"
        print(f"[acc-memo] attached {memo_src}", flush=True)
    if CALIBRATE:
        sh(f"{sys.executable} -m search.{METHOD} --calibrate {CALIBRATE} {common}")
    out = work / OUT_NAME
    cache = work / f"cp33_bo_cache_r{RES}"   # RES-namespaced so @224 and @640 caches stay distinct
    deadline_s = max(600, int(DEADLINE_H * 3600 - (time.time() - START)))
    budget = f"--budget {BUDGET} --n-init {N_INIT} --deadline-s {deadline_s}"
    print(f"[deadline] workers stop new evals after ~{deadline_s / 3600:.1f} h", flush=True)

    # how much work each seed still owes (from the restored caches) -> rebalance per session
    from search.bo import assign_seeds_to_gpus, seed_remaining_evals
    remaining = {s: seed_remaining_evals(cache, s, BUDGET, method=METHOD) for s in range(SEEDS)}
    ndone = sum(1 for s in remaining if remaining[s] == 0)
    print(f"[resume] {ndone}/{SEEDS} seeds complete; remaining evals/seed: {remaining}", flush=True)

    ngpu = int(subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
    ).decode().strip() or "0")
    print(f"[gpu] {ngpu} CUDA device(s) visible", flush=True)

    if ngpu >= 2 and SEEDS > 1:
        # Each session, LPT-balance the UNFINISHED work across the GPUs (one worker per
        # device, in PARALLEL) and merge. Done seeds are still assigned (they reload from
        # cache instantly), so every seed lands in a part and the merge covers the whole DoD.
        # calibrate above warmed ultralytics; a short stagger guards the first-touch race.
        assignments = assign_seeds_to_gpus(remaining, ngpu)
        procs, parts = [], []
        for g, seeds_g in enumerate(assignments):
            if not seeds_g:
                continue
            part = work / f"{out.stem}.part{g}.json"
            parts.append(part)
            sl = ",".join(map(str, seeds_g))
            wcmd = (f"{sys.executable} -m search.{METHOD} --seed-list {sl} "
                    f"{budget} {common} --out {part} --cache {cache}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            print(f"+ [gpu{g}] seeds {seeds_g}", flush=True)
            procs.append(subprocess.Popen(wcmd, shell=True, env=env))
            time.sleep(10)  # let the first worker win any ultralytics first-touch race
        for pr in procs:
            pr.wait()  # ignore rc: a worker exits 1 on a partial DoD-FAIL; check outputs
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise SystemExit(f"GPU worker(s) produced no output: {missing}")
        sh(f"{sys.executable} -m search.{METHOD} --merge {' '.join(map(str, parts))} --out {out}")
    else:
        # 1 GPU (or 1 seed): run every seed; the cache resumes the unfinished ones and reloads
        # the finished ones instantly. exit 1 = a valid DoD-FAIL verdict, so only a missing
        # output file is fatal — the verdict itself is data in the JSON.
        allseeds = ",".join(str(s) for s in range(SEEDS))
        cmd = (f"{sys.executable} -m search.{METHOD} --seed-list {allseeds} {budget} "
               f"{common} --out {out} --cache {cache}")
        print("+", cmd, flush=True)
        rc = subprocess.run(cmd, shell=True).returncode
        if not out.exists():
            raise SystemExit(f"search.{METHOD} produced no results (rc={rc})")
    # report completion so you know whether to re-run (resume) or stop
    try:
        payload = json.loads(out.read_text())
        done = payload.get("complete")
        nxt = ("COMPLETE — DoD final" if done else
               "PARTIAL — re-run the kernel (with this output attached as input) to resume")
        print(f"[done] {out.name}: complete={done} passes={payload.get('passes')} -> {nxt}",
              flush=True)
    except (OSError, ValueError) as e:
        print(f"[done] wrote {out} (could not parse completion: {e})", flush=True)


if __name__ == "__main__":
    main()
