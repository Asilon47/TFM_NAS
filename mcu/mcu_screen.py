"""CP 10.3 Stage 1 — structured screen of the MCU search space through the real cycle oracle.

The cheap-surrogate route was tried and REJECTED (2026-07-19): per-class linear fits over
the six priced graft graphs LOO at 52-61 % MAPE — channel-width-dependent tiling efficiency
(the measured 1.52x ops/cycle swing) defeats aggregate-linear models at this sample size.
Stage 1 therefore prices a stratified sample of the space with `mcu.cycle_oracle` directly:
~13 min/candidate, laptop-local, content-cached, every number real (ranking-only, as
always). The screen's Pareto front (proxy mAP comes later) seeds the MOTPE stage.

Space (the CP 10.3 dims): OFA (ks, e, d) x width spec (stage_ratios menu, rest_ratio) x
neck {None, topdown, pan} x res {128, 160, 192}. The sampler is seeded + stratified across
neck/res cells so the screen covers the corners a random draw would miss at n~20.

Run (laptop, any venv; the oracle shells into .venv-nas + docker):
    python -m mcu.mcu_screen --n 18 --seed 0          # sample + price (background this)
    python -m mcu.mcu_screen --n 18 --seed 0 --dry    # sample only, print the plan
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "mcu" / "screen"

# OFA-MBv3 vocabulary (supernet/sampler.py contract): 5 stages, 4 blocks each.
KS_CHOICES = (3, 5, 7)
E_CHOICES = (3, 4, 6)
D_CHOICES = (2, 3, 4)
STAGE_RATIO_MENU = (0.0, 0.2, 0.3, 0.45, 0.6)
REST_MENU = (0.1, 0.2, 0.3, 0.45)
NECKS = (None, "topdown", "pan")
RESES = (128, 160, 192)


def sample_candidate(rng: random.Random) -> dict:
    d = [rng.choice(D_CHOICES) for _ in range(5)]
    n_blocks = 20                                    # 5 stages x max depth 4
    return {
        "arch": {"ks": [rng.choice(KS_CHOICES) for _ in range(n_blocks)],
                 "e": [rng.choice(E_CHOICES) for _ in range(n_blocks)],
                 "d": d},
        "spec": {"stage_ratios": [rng.choice(STAGE_RATIO_MENU) for _ in range(5)],
                 "rest_ratio": rng.choice(REST_MENU)},
        "neck": rng.choice(NECKS),
        "imgsz": rng.choice(RESES),
    }


def sample_screen(n: int, seed: int, *, necks: list | None = None,
                  reses: list | None = None) -> list[dict]:
    """Stratified: every (neck, res) cell gets floor(n/cells) draws, remainder random.

    ``necks``/``reses`` restrict the cells — the wave-2 focus knob. Wave-1 found the
    192-topdown region owns the proxy Pareto, so wave-2 prices that region densely
    (``necks=["topdown"], reses=[160, 192]``) for its latency-feasible frontier.
    """
    rng = random.Random(seed)
    nk_cells = list(necks) if necks is not None else list(NECKS)
    r_cells = list(reses) if reses is not None else list(RESES)
    cells = [(nk, r) for nk in nk_cells for r in r_cells]
    out: list[dict] = []
    per_cell, extra = divmod(n, len(cells))
    for i, (nk, r) in enumerate(cells):
        take = per_cell + (1 if i < extra else 0)
        for _ in range(take):
            c = sample_candidate(rng)
            c["neck"], c["imgsz"] = nk, r
            out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--necks", nargs="+", default=None,
                    help="restrict neck cells (use the literal 'None' for the neck-less cell)")
    ap.add_argument("--reses", nargs="+", type=int, default=None, help="restrict res cells")
    ap.add_argument("--tag", type=str, default=None, help="output name (default screen_nN_seedS)")
    ap.add_argument("--dry", action="store_true", help="print the plan, price nothing")
    a = ap.parse_args(argv)

    necks = None if a.necks is None else [None if x == "None" else x for x in a.necks]
    cands = sample_screen(a.n, a.seed, necks=necks, reses=a.reses)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / f"{a.tag or f'screen_n{a.n}_seed{a.seed}'}.json"
    plan.write_text(json.dumps(cands, indent=2))
    print(f"[plan] {len(cands)} candidates -> {plan}")
    if a.dry:
        for c in cands:
            print(f"  d={c['arch']['d']} ratios={c['spec']['stage_ratios']} "
                  f"rest={c['spec']['rest_ratio']} neck={c['neck']} res={c['imgsz']}")
        return 0

    from mcu.cycle_oracle import candidate_key, cycles_for

    results_path = OUT / f"{a.tag or f'screen_n{a.n}_seed{a.seed}'}_results.json"
    results: list[dict[str, Any]] = []
    for i, c in enumerate(cands):
        r = cycles_for(c)
        results.append(r)
        stat = (f"{r['cycles']:,} cyc {r['fps_at_175mhz']} FPS" if r["status"] == "ok"
                else f"{r['status']} @ {r.get('stage')}")
        print(f"[{i + 1}/{len(cands)}] {candidate_key(c)} ({r['status']}): {stat}", flush=True)
        results_path.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if r["status"] == "ok"]
    print(f"[done] {len(ok)}/{len(cands)} priced ok -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
