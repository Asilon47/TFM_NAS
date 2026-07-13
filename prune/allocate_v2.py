"""Winner-v2-OFA Track 1 — honest-fence allocation (+ Track 1b OFA topology screen).

v1 (``prune/allocate.py``, HALP-lite) modeled stage latency LINEAR in retained channels off
the @640 LUT rows — measured +23–28 % optimistic on concentrated cuts (procedure.md
2026-07-12); its predicted-latency claims are retired. v2 prices every candidate by the
PHYSICAL law the graft family obeys (``search/latency_model.py`` ``PHYSICAL_GRAFT_*``:
ms = b + a·act_MB, LOO 2–3 %) applied to the candidate's ACTUAL activation bytes: build the
pruned graft on CPU, export the deploy ONNX, read ``act_mbytes``. Channel COUNTS at fixed
per-stage ratios are importance-invariant, so a cheap local-l2 prune prices exactly the
shapes global_taylor will train remotely.

Ranking (lexicographic; measured evidence, not saliency): among specs under the act fence —
(1) max act, spend the whole latency budget (mAP is ~monotone in act within the family:
254 MB→0.777, 303→0.795, 344→0.816); (2) min max(stage_ratios) — balanced beats concentrated
(uniform r40 0.8163 no-KD > halp_10p4 0.8126 +KD at MORE act); (3) min rest_ratio (the KD
signal flows through the head); (4) min mean(stage_ratios).

Track 1b (``--topology-screen``) answers "reduce latency by re-search": enumerate OFA (e,d)
subnets (ks kept = winner's — act-neutral), price each by exact act bytes, and emit the
min-act probe arch (best depth_sum zero-cost signal under the fence) for a wave-1 run with
fully inherited supernet weights.

Run (laptop ``.venv-nas``, CPU)::

    python -m prune.allocate_v2 --target-fp16-ms 7.2 --target-fp16-ms 6.8
    python -m prune.allocate_v2 --topology-screen
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path
from typing import Any

from search.latency_model import (
    PHYSICAL_GRAFT_FP16,
    PHYSICAL_GRAFT_FP32,
    act_bytes_to_ms,
    act_limit_for_ms,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "prune" / "specs"
WORKDIR = ROOT / "data" / "allocate_v2"          # gitignored scratch (probe ONNX exports)

STAGE_RUNGS = tuple(round(0.1 * i, 1) for i in range(8))   # 0.0 … 0.7 per-stage ratio
REST_RUNGS = tuple(round(0.1 * i, 1) for i in range(1, 7))  # 0.1 … 0.6 (must be > 0)
REST_BASE = REST_RUNGS[0]           # sensitivity baseline uses the minimum legal rest
PROBE_RATIO = 0.4                   # Phase-A one-knob probe amplitude
ACT_BIN_MB = 4.0                    # "equal act" tie-band for the balance tiebreak

# OFA-MBv3 topology constants (catalog/ofa_mbv3.py): 5 stages × up to 4 blocks.
N_STAGES = 5
MAX_BLOCKS_PER_STAGE = 4


# --- pure: linear act predictor + ranking -----------------------------------------------------

def predict_act(stage_ratios: Any, rest_ratio: float, sens: dict) -> float:
    """Linear screening estimate of act_mbytes for a spec, from Phase-A sensitivities.

    ``sens`` = {"base_act": act at all-zero stages/rest=REST_BASE, "stage": [MB-per-unit-ratio
    ×5], "rest": MB-per-unit-ratio}. Screening-only; survivors are re-priced honestly.
    """
    act = sens["base_act"]
    for r, c in zip(stage_ratios, sens["stage"], strict=True):
        act -= c * r
    act -= sens["rest"] * (rest_ratio - REST_BASE)
    return act


def enumerate_specs(stage_rungs: tuple = STAGE_RUNGS,
                    rest_rungs: tuple = REST_RUNGS) -> list[tuple[tuple, float]]:
    """The full (stage_ratios, rest_ratio) grid — 8^5·6 ≈ 197k combos, cheap to scan."""
    return [(sr, rr) for sr in itertools.product(stage_rungs, repeat=N_STAGES)
            for rr in rest_rungs]


def rank_candidates(cands: list[dict], act_max: float,
                    bin_mb: float = ACT_BIN_MB) -> list[dict]:
    """Feasible candidates best-first: max act (binned) → balance → rest → mean.

    Each cand: {"stage_ratios", "rest_ratio", "act"}. Infeasible (act > act_max) dropped.
    """
    feasible = [c for c in cands if c["act"] <= act_max]

    def key(c: dict) -> tuple:
        return (-int(c["act"] // bin_mb),
                max(c["stage_ratios"]),
                c["rest_ratio"],
                sum(c["stage_ratios"]) / len(c["stage_ratios"]))

    return sorted(feasible, key=key)


def spec_payload(stage_ratios: Any, rest_ratio: float, *, act_honest: float,
                 act_predicted: float, params_after: int, fence_fp16_ms: float,
                 act_max: float, extra: dict | None = None) -> dict:
    """The emitted spec JSON — the 5 legacy keys recover_graft copies + v2 provenance."""
    pred_fp32 = act_bytes_to_ms(act_honest, **_fit_args(PHYSICAL_GRAFT_FP32))
    pred_fp16 = act_bytes_to_ms(act_honest, **_fit_args(PHYSICAL_GRAFT_FP16))
    fence_fp32 = act_bytes_to_ms(act_max, **_fit_args(PHYSICAL_GRAFT_FP32))
    return {
        # legacy keys (recover_graft.py row["spec"] copies these five)
        "stage_ratios": [float(r) for r in stage_ratios],
        "rest_ratio": float(rest_ratio),
        "predicted_fp32_ms": round(pred_fp32, 3),
        "fp16_estimate_ms": round(pred_fp16, 3),
        "target_fp32_ms": round(fence_fp32, 3),
        # v2 metadata
        "act_mbytes_honest": round(act_honest, 2),
        "act_mbytes_predicted": round(act_predicted, 2),
        "params_after": int(params_after),
        "fence": {"fp16_target_ms": fence_fp16_ms, "act_max_mb": round(act_max, 2),
                  "fit": "search/graft_latency_fit.json"},
        "objective": "max act ≤ fence; ties: min max-stage, min rest, min mean "
                     "(balance beats concentration — measured, procedure.md 2026-07-12)",
        "provenance": "allocate_v2 honest fence: CPU-built pruned graft → deploy ONNX → "
                      "act_mbytes → physical graft fit (LOO ~2-3 %); linear predictor is "
                      "screening-only, this spec's act is the honest re-price",
        "arch_tag": "winner",
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(extra or {}),
    }


def _fit_args(fit: dict) -> dict:
    return {"slope": fit["slope"], "intercept": fit["intercept"]}


# --- pure: Track 1b topology grid --------------------------------------------------------------

D_PATTERNS = (
    [2, 2, 2, 2, 2],       # minimal
    [2, 2, 2, 2, 3],
    [2, 2, 2, 3, 3],
    [2, 2, 3, 3, 3],
    [2, 2, 4, 3, 3],       # winner-v1 anchor
    [3, 3, 2, 2, 2],       # early-heavy control (expensive where spatial is high)
)
E_PATTERNS = (
    [3, 3, 3, 3, 3],       # uniform minimum
    [4, 4, 4, 4, 4],
    [6, 6, 6, 6, 6],       # uniform maximum (OFA default-ish)
    [3, 3, 4, 4, 4],       # lean early, mid late
    [3, 3, 6, 6, 6],       # lean early, rich late (the roofline-shaped guess)
    [6, 6, 3, 3, 3],       # rich early control
)


def expand_per_stage(per_stage: list[int],
                     blocks_per_stage: int = MAX_BLOCKS_PER_STAGE) -> list[int]:
    """Per-stage value → the flat per-block list OFA arch dicts use (5×4 = 20 entries)."""
    if len(per_stage) != N_STAGES:
        raise ValueError(f"need {N_STAGES} per-stage values (got {len(per_stage)})")
    return [v for v in per_stage for _ in range(blocks_per_stage)]


def screen_grid(winner_ks: list[int]) -> list[dict]:
    """The Track-1b candidate arch dicts: (d, e) grid × the winner's ks (act-neutral)."""
    out = []
    for d in D_PATTERNS:
        for e in E_PATTERNS:
            tag = "minact_d" + "".join(map(str, d)) + "_e" + "".join(map(str, e))
            out.append({"tag": tag, "d": list(d), "e_stage": list(e),
                        "arch": {"ks": list(winner_ks), "e": expand_per_stage(e),
                                 "d": list(d)}})
    return out


def pick_minact(rows: list[dict], act_max: float) -> dict:
    """The probe arch: under the fence, best zero-cost signal first (depth_sum), then max act
    (spend the budget). Raises if nothing fits."""
    fits = [r for r in rows if r["act_mbytes"] <= act_max]
    if not fits:
        raise ValueError(f"no screened topology fits act ≤ {act_max:.0f} MB "
                         f"(min seen {min(r['act_mbytes'] for r in rows):.0f})")
    return max(fits, key=lambda r: (sum(r["d"]), r["act_mbytes"]))


PAIR_RUNGS = (0.2, 0.3, 0.4, 0.5)   # uniform pair-spec ratios tried lightest-first


def pick_probe(rows: list[dict], act_max: float) -> tuple[dict, bool]:
    """(probe row, needs_pair): pure pick if any topology fits; else the best candidate to
    PAIR with a light uniform width spec — max depth_sum (zero-cost signal), then MIN act
    (the lightest required prune). Measured 2026-07-13: the pure floor is 340 MB ≈ 8.2 ms
    fp16 → pairing is the expected path."""
    try:
        return pick_minact(rows, act_max), False
    except ValueError:
        return max(rows, key=lambda r: (sum(r["d"]), -r["act_mbytes"])), True


# --- impure: honest CPU builds -----------------------------------------------------------------

def honest_spec_features(arch: dict, spec: dict | None, *, supernet: Any = None,
                         imgsz: int = 640, workdir: Any = None,
                         name: str = "probe") -> dict:
    """Build the (optionally spec-pruned) graft on CPU → deploy ONNX → exact act bytes.

    ``spec=None`` prices the UNPRUNED subnet (Track 1b). Head weights are irrelevant to
    shapes, so no donor is needed; l2/local importance keeps counts identical to what
    global_taylor produces at the same per-stage ratios.
    """
    import torch

    from detect.pose_model import build_grafted_pose_model
    from eval.shortft import _seed_everything
    from prune.prune_baseline import TRACE_IMGSZ, _export_deploy_onnx
    from prune.prune_graft import prune_graft
    from prune.recover_graft import spec_ratio_dict
    from search.latency_model import extract_onnx_features
    from supernet.sampler import load_supernet

    sn = supernet if supernet is not None else load_supernet()
    _seed_everything(0)
    model = build_grafted_pose_model(arch, supernet=sn)
    report = None
    if spec is not None:
        prd, ignored = spec_ratio_dict(model, arch["d"], spec)
        report = prune_graft(model, torch.randn(1, 3, TRACE_IMGSZ, TRACE_IMGSZ),
                             ratio=spec["rest_ratio"], pruning_ratio_dict=prd,
                             extra_ignored=ignored)
    wd = Path(workdir) if workdir is not None else WORKDIR
    wd.mkdir(parents=True, exist_ok=True)
    onnx_path = wd / f"{name}.onnx"
    _export_deploy_onnx(model, onnx_path, imgsz=imgsz)
    feats = extract_onnx_features(onnx_path)
    params = (report["params_after"] if report is not None
              else sum(p.numel() for p in model.parameters()))
    return {"act_mbytes": float(feats["act_mbytes"]), "coverage": float(feats["coverage"]),
            "params_after": int(params), "onnx": str(onnx_path)}


def fit_sensitivities(arch: dict, *, supernet: Any = None, imgsz: int = 640,
                      workdir: Any = None) -> dict:
    """Phase A: 8 honest builds → the linear per-knob act model for grid screening."""
    def zspec(**over: Any) -> dict:
        s = {"stage_ratios": [0.0] * N_STAGES, "rest_ratio": REST_BASE}
        s.update(over)
        return s

    unpruned = honest_spec_features(arch, None, supernet=supernet, imgsz=imgsz,
                                    workdir=workdir, name="sens_unpruned")
    base = honest_spec_features(arch, zspec(), supernet=supernet, imgsz=imgsz,
                                workdir=workdir, name="sens_base")
    stage_sens = []
    for s in range(N_STAGES):
        ratios = [0.0] * N_STAGES
        ratios[s] = PROBE_RATIO
        probe = honest_spec_features(arch, zspec(stage_ratios=ratios), supernet=supernet,
                                     imgsz=imgsz, workdir=workdir, name=f"sens_s{s}")
        stage_sens.append((base["act_mbytes"] - probe["act_mbytes"]) / PROBE_RATIO)
    rest_hi = 0.5
    probe_r = honest_spec_features(arch, zspec(rest_ratio=rest_hi), supernet=supernet,
                                   imgsz=imgsz, workdir=workdir, name="sens_rest")
    rest_sens = (base["act_mbytes"] - probe_r["act_mbytes"]) / (rest_hi - REST_BASE)
    return {"base_act": base["act_mbytes"], "stage": stage_sens, "rest": rest_sens,
            "unpruned_act": unpruned["act_mbytes"], "params_unpruned": unpruned["params_after"]}


def search_specs(arch: dict, fence_fp16_ms: list[float], *, supernet: Any = None,
                 imgsz: int = 640, workdir: Any = None, out_dir: Any = None,
                 verify_top: int = 12, max_verify: int = 36) -> list[Path]:
    """Phase A + B + honest verification; one emitted spec per fp16 fence."""
    from supernet.sampler import load_supernet

    sn = supernet if supernet is not None else load_supernet()
    out_dir = Path(out_dir) if out_dir is not None else SPEC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sens = fit_sensitivities(arch, supernet=sn, imgsz=imgsz, workdir=workdir)
    print(f"[phase A] base_act={sens['base_act']:.1f} MB  stage_sens="
          f"{[round(c, 1) for c in sens['stage']]}  rest_sens={sens['rest']:.1f}", flush=True)

    grid = enumerate_specs()
    cands = [{"stage_ratios": sr, "rest_ratio": rr, "act": predict_act(sr, rr, sens)}
             for sr, rr in grid]

    written: list[Path] = []
    for fence in fence_fp16_ms:
        act_max = act_limit_for_ms(fence, PHYSICAL_GRAFT_FP16)
        ranked = rank_candidates(cands, act_max)
        if not ranked:
            raise ValueError(f"fence {fence} ms (act ≤ {act_max:.0f} MB) infeasible "
                             f"even at max rungs")
        chosen = None
        verified: list[dict] = []
        for i, c in enumerate(ranked[:max_verify]):
            if i >= verify_top and chosen is not None:
                break
            spec = {"stage_ratios": list(c["stage_ratios"]), "rest_ratio": c["rest_ratio"]}
            h = honest_spec_features(arch, spec, supernet=sn, imgsz=imgsz, workdir=workdir,
                                     name=f"verify_f{str(fence).replace('.', 'p')}_{i}")
            row = {**spec, "act_predicted": round(c["act"], 2),
                   "act_honest": round(h["act_mbytes"], 2), "params": h["params_after"],
                   "coverage": round(h["coverage"], 3),
                   "fits": h["act_mbytes"] <= act_max}
            verified.append(row)
            print(f"[verify {fence}ms #{i}] stages={spec['stage_ratios']} "
                  f"rest={spec['rest_ratio']} act={h['act_mbytes']:.1f} "
                  f"(pred {c['act']:.1f}) {'OK' if row['fits'] else 'OVER'}", flush=True)
            if row["fits"] and chosen is None:
                chosen = (spec, c, h)
        if chosen is None:
            raise ValueError(f"fence {fence} ms: none of the top {max_verify} verified under "
                             f"{act_max:.0f} MB — widen rungs or raise the fence")
        spec, c, h = chosen
        payload = spec_payload(spec["stage_ratios"], spec["rest_ratio"],
                               act_honest=h["act_mbytes"], act_predicted=c["act"],
                               params_after=h["params_after"], fence_fp16_ms=fence,
                               act_max=act_max,
                               extra={"sensitivities": {k: sens[k] for k in
                                                        ("base_act", "stage", "rest")},
                                      "verified": verified})
        out = out_dir / f"v2_act{round(act_max)}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[emit] {out.name}: stages={payload['stage_ratios']} "
              f"rest={payload['rest_ratio']} act={payload['act_mbytes_honest']} MB "
              f"pred fp16={payload['fp16_estimate_ms']} fp32={payload['predicted_fp32_ms']}",
              flush=True)
        written.append(out)
    return written


def topology_screen(winner_arch: dict, *, fence_fp16_ms: float = 7.2, supernet: Any = None,
                    imgsz: int = 640, workdir: Any = None, out_dir: Any = None) -> Path:
    """Track 1b: price the (d,e) grid by exact act bytes, emit the min-act probe arch."""
    from supernet.sampler import load_supernet

    sn = supernet if supernet is not None else load_supernet()
    out_dir = Path(out_dir) if out_dir is not None else SPEC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    act_max = act_limit_for_ms(fence_fp16_ms, PHYSICAL_GRAFT_FP16)

    rows = []
    for cand in screen_grid(winner_arch["ks"]):
        h = honest_spec_features(cand["arch"], None, supernet=sn, imgsz=imgsz,
                                 workdir=workdir, name=cand["tag"])
        row = {"tag": cand["tag"], "d": cand["d"], "e_stage": cand["e_stage"],
               "depth_sum": sum(cand["d"]), "act_mbytes": round(h["act_mbytes"], 2),
               "params": h["params_after"], "coverage": round(h["coverage"], 3),
               "pred_fp16_ms": round(act_bytes_to_ms(
                   h["act_mbytes"], **_fit_args(PHYSICAL_GRAFT_FP16)), 3),
               "pred_fp32_ms": round(act_bytes_to_ms(
                   h["act_mbytes"], **_fit_args(PHYSICAL_GRAFT_FP32)), 3),
               "arch": cand["arch"]}
        rows.append(row)
        print(f"[screen] {row['tag']:28s} act={row['act_mbytes']:7.1f} MB "
              f"pred fp16={row['pred_fp16_ms']:5.2f} d_sum={row['depth_sum']}", flush=True)

    rows.sort(key=lambda r: r["act_mbytes"])
    probe, needs_pair = pick_probe(rows, act_max)
    pair_spec: dict | None = None
    paired: dict | None = None
    if needs_pair:
        print(f"[pair] no pure topology fits ≤ {act_max:.0f} MB (floor "
              f"{rows[0]['act_mbytes']:.0f}) — pairing {probe['tag']} with the lightest "
              f"uniform width spec", flush=True)
        for r in PAIR_RUNGS:
            spec = {"stage_ratios": [r] * N_STAGES, "rest_ratio": max(r, REST_BASE)}
            h = honest_spec_features(probe["arch"], spec, supernet=sn, imgsz=imgsz,
                                     workdir=workdir,
                                     name=f"{probe['tag']}_pair{int(r * 100)}")
            fits = h["act_mbytes"] <= act_max
            print(f"[pair r={r}] act={h['act_mbytes']:.1f} MB "
                  f"{'OK' if fits else 'OVER'}", flush=True)
            if fits:
                pair_spec, paired = spec, h
                break
        if pair_spec is None or paired is None:
            raise ValueError(f"no uniform pair spec ≤ {PAIR_RUNGS[-1]} brings "
                             f"{probe['tag']} under {act_max:.0f} MB")

    payload = {
        "tag": probe["tag"], "arch": probe["arch"], "d": probe["d"],
        "e_stage": probe["e_stage"], "depth_sum": probe["depth_sum"],
        "act_mbytes": probe["act_mbytes"], "pred_fp16_ms": probe["pred_fp16_ms"],
        "pred_fp32_ms": probe["pred_fp32_ms"], "params": probe["params"],
        "fence": {"fp16_target_ms": fence_fp16_ms, "act_max_mb": round(act_max, 2),
                  "fit": "search/graft_latency_fit.json"},
        "pick_rule": "pure: act ≤ fence, max depth_sum (zero-cost signal, CP 2.4 rho=0.843) "
                     "then max act; paired: max depth_sum then MIN act (lightest prune)",
        "ks_note": "ks copied from winner-v1 (act-neutral; keeps the probe in the fitted "
                   "family)",
        "needs_pair": needs_pair,
        "screen": [{k: r[k] for k in ("tag", "depth_sum", "act_mbytes", "pred_fp16_ms",
                                      "pred_fp32_ms", "params")} for r in rows],
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = out_dir / "minact_arch.json"
    if pair_spec is not None and paired is not None:
        pair_payload = spec_payload(
            pair_spec["stage_ratios"], pair_spec["rest_ratio"],
            act_honest=paired["act_mbytes"], act_predicted=paired["act_mbytes"],
            params_after=paired["params_after"], fence_fp16_ms=fence_fp16_ms,
            act_max=act_max,
            extra={"arch_tag": probe["tag"], "paired_arch_json": "prune/specs/minact_arch.json",
                   "note": "uniform pair spec for the Track-1b probe: run recover_graft with "
                           "--arch-json prune/specs/minact_arch.json --ratio-spec THIS file"})
        pair_out = out_dir / f"u{int(pair_spec['rest_ratio'] * 100)}.json"
        pair_out.write_text(json.dumps(pair_payload, indent=2) + "\n")
        payload["pair_spec_file"] = pair_out.name
        payload["act_after_pair"] = paired["act_mbytes"]
        payload["params_after_pair"] = paired["params_after"]
        payload["pred_fp16_ms_after_pair"] = round(act_bytes_to_ms(
            paired["act_mbytes"], **_fit_args(PHYSICAL_GRAFT_FP16)), 3)
        payload["pred_fp32_ms_after_pair"] = round(act_bytes_to_ms(
            paired["act_mbytes"], **_fit_args(PHYSICAL_GRAFT_FP32)), 3)
        print(f"[emit] {pair_out.name}: uniform r={pair_spec['rest_ratio']} on "
              f"{probe['tag']} → act={paired['act_mbytes']:.1f} MB "
              f"pred fp16={payload['pred_fp16_ms_after_pair']}", flush=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[emit] {out.name}: {probe['tag']} act={probe['act_mbytes']} MB "
          f"pred fp16={probe['pred_fp16_ms']} needs_pair={needs_pair} "
          f"(fence {fence_fp16_ms})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-fp16-ms", type=float, action="append",
                   help="repeatable; fp16 is the binding axis (bar 7.75 → fences 7.2/6.8)")
    p.add_argument("--topology-screen", action="store_true",
                   help="Track 1b: OFA (d,e) act screen → prune/specs/minact_arch.json")
    p.add_argument("--fence-fp16-ms", type=float, default=7.2,
                   help="fence for the topology-screen pick")
    p.add_argument("--winner-dir", type=Path, default=ROOT / "state" / "winner_v1")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--verify-top", type=int, default=12)
    a = p.parse_args(argv)

    from eval.verify_winner import load_winner

    arch = load_winner(a.winner_dir)["arch"]
    if a.topology_screen:
        topology_screen(arch, fence_fp16_ms=a.fence_fp16_ms, imgsz=a.imgsz,
                        workdir=a.workdir, out_dir=a.out_dir)
        return 0
    targets = a.target_fp16_ms or [7.2, 6.8]
    search_specs(arch, targets, imgsz=a.imgsz, workdir=a.workdir, out_dir=a.out_dir,
                 verify_top=a.verify_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
