"""GAP8 cycle oracle — the MCU-side latency oracle for the CP 10.3 search (2026-07-19).

One candidate = one deployable graft graph::

    {"arch": {"ks": [...], "e": [...], "d": [...]},   # OFA subnet
     "spec": {"stage_ratios": [...], "rest_ratio": ...},  # width (prune) spec
     "neck": None | "topdown" | "pan",
     "imgsz": 160}

``cycles_for`` prices it on GVSOC (int8, matched 84 KB AutoTiler L2 — the budget every
recorded CP 10.1 number uses) and caches by content key. Three properties make this a
better search oracle than the Jetson ever was: cycles are **deterministic** (no clocks,
no contention, no fresh-cache medians), **weight-free** (the export prunes data-free l2;
counts are spec-pinned = importance-invariant, the CP 10.1 parity fact), and **local**
(laptop docker; no board session).

A failed stage is a RESULT, not an error (cyc_probe doctrine): the oracle caches
``{"status": "infeasible", "stage": ...}`` so the search treats the candidate as
constraint-violating and never re-prices it. Sim cycles are RANKING-ONLY, as always.

Orchestration shells out (``.venv-nas`` python for the export, docker for the sim), so
this module itself stays importable under either venv; the pure parts are tested in
``.venv`` (tests/test_cycle_oracle.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORACLE_DIR = REPO / "data" / "mcu" / "oracle"
CYC_DIR = REPO / "data" / "mcu" / "cyc"
NAS_PY = REPO / ".venv-nas" / "bin" / "python"
CYC_PROBE = REPO / "mcu" / "probes" / "cyc_probe.sh"
MATCHED_L2 = 84000                       # the CP 10.1 matched budget; every recorded number
NECK_KINDS = (None, "topdown", "pan")


def canonical_candidate(cand: dict) -> dict:
    """Normalize a candidate to the exact fields that determine the graph (and the key)."""
    arch = cand["arch"]
    spec = cand["spec"]
    neck = cand.get("neck")
    if neck not in NECK_KINDS:
        raise ValueError(f"unknown neck {neck!r}; known: {NECK_KINDS}")
    return {
        "arch": {"ks": list(arch["ks"]), "e": list(arch["e"]), "d": list(arch["d"])},
        "spec": {"stage_ratios": [float(r) for r in spec["stage_ratios"]],
                 "rest_ratio": float(spec["rest_ratio"])},
        "neck": neck,
        "imgsz": int(cand.get("imgsz", 160)),
    }


def candidate_key(cand: dict) -> str:
    """Content key: same graph → same key, field order irrelevant."""
    c = canonical_candidate(cand)
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def _parse_cyc_json(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {"status": "ok",
            "cycles": int(d["cycles_at_total"]),
            "ms_at_175mhz": float(d["ms_at_175mhz"]),
            "fps_at_175mhz": float(d["fps_at_175mhz"]),
            "ops": int(d["operations"]),
            "ops_per_cycle": float(d["ops_per_cycle"]),
            "n_nodes": int(d["n_nodes"]),
            "autotiler_l2_budget": int(d["autotiler_l2_budget"])}


def cycles_for(cand: dict, *, l2: int = MATCHED_L2, force: bool = False) -> dict:
    """Price one candidate. Returns the cached/parsed result dict (see module doc)."""
    c = canonical_candidate(cand)
    key = candidate_key(c)
    name = f"cand_{key}"
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    cache = ORACLE_DIR / f"{name}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    imgsz = c["imgsz"]
    workdir = ORACLE_DIR / name
    workdir.mkdir(parents=True, exist_ok=True)
    arch_path = workdir / "arch.json"
    spec_path = workdir / "spec.json"
    arch_path.write_text(json.dumps({"arch": c["arch"]}))
    spec_path.write_text(json.dumps(c["spec"]))
    onnx_out = REPO / "models" / f"res{imgsz}" / f"{name}.onnx"
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    export_cmd = [str(NAS_PY), "-m", "mcu.export_pruned",
                  "--spec", str(spec_path), "--arch-meta", str(arch_path),
                  "--imgsz", str(imgsz), "--out", str(onnx_out)]
    if c["neck"]:
        export_cmd += ["--neck", c["neck"]]
    exp = subprocess.run(export_cmd, cwd=REPO, capture_output=True, text=True)
    if exp.returncode != 0 or not onnx_out.exists():
        result = {"status": "infeasible", "stage": "export", "key": key,
                  "candidate": c, "detail": (exp.stderr or exp.stdout)[-800:]}
        cache.write_text(json.dumps(result, indent=2))
        return result

    env = dict(os.environ, CYC_RES=str(imgsz), CYC_L2=str(l2))
    sim = subprocess.run(["bash", str(CYC_PROBE), name], cwd=REPO, env=env,
                         capture_output=True, text=True)
    cyc_json = CYC_DIR / f"{name}.json"
    if not cyc_json.exists():
        stage = "codegen" if "GEN_FAIL" in sim.stdout else \
                "gvsoc" if "RUN_FAIL" in sim.stdout else "sim"
        result = {"status": "infeasible", "stage": stage, "key": key,
                  "candidate": c, "detail": sim.stdout[-800:]}
        cache.write_text(json.dumps(result, indent=2))
        return result

    result = {**_parse_cyc_json(cyc_json), "key": key, "candidate": c}
    cache.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", type=Path, required=True,
                    help="json file holding one candidate dict")
    ap.add_argument("--l2", type=int, default=MATCHED_L2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    res = cycles_for(json.loads(a.candidate.read_text()), l2=a.l2, force=a.force)
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
