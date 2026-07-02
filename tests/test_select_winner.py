"""CP 3.5 — winner-v1 selection over the BO∪TPE frontier (two-anchor iso-J λ).

``search.select_winner`` is the pure argmax half of CP 3.5: given the search
frontiers (``cp33_bo.json`` + ``cp34_tpe.json``) and two reference models, it
calibrates λ (``search.objective.lambda_from_anchors``), maximises
``J = acc_eff − λ·latency`` over the feasible frontier, and serialises α*. Every
number it needs is already in the frontier JSON (each point carries ``latency_ms`` +
``acc_eff``), so this is LUT/GPU-free and fully unit-testable — mirroring how the
pure half of ``search.bo`` is tested in ``.venv``/CI.

The synthetic frontier below stores ``acc``/``latency`` on each point directly, so J
is fully controlled by the fixture (no LUT recompute), which makes the argmax and the
ceiling filter deterministic.
"""
import json
import random

import pytest

from search.arch_to_blocks import random_arch_dict
from search.objective import Anchor
from search.select_winner import (
    lambda_grid,
    lambda_sensitivity,
    load_frontier,
    main,
    read_anchor,
    select_winner,
    serialize_winner,
    winner_record,
)
from search.space import decode, encode

_RNG = random.Random(0)


def _pt(acc: float, lat: float) -> dict:
    """A frontier point in bo.py's record shape (acc_eff==acc, μ=0 in v1)."""
    return {"arch": random_arch_dict(_RNG), "acc": acc, "latency_ms": lat, "acc_eff": acc}


def _bo_payload(seed_points: list[list[dict]], *, t_max: float = 12.75) -> dict:
    """A cp33_bo.json-shaped payload: per-seed ``bo_frontier`` lists, no ``method`` key."""
    runs = [{"seed": s, "bo_hv": 0.0, "rs_hv": 0.0, "complete": True, "bo_frontier": pts}
            for s, pts in enumerate(seed_points)]
    return {"passes": True, "n_seeds": len(runs), "bo_hv_mean": 0.0, "bo_hv_std": 0.0,
            "rs_hv_mean": 0.0, "rs_hv_std": 0.0, "t_max_ms": t_max, "res": 640,
            "budget": 50, "structural": False, "complete": True, "runs": runs}


def _tpe_payload(seed_points: list[list[dict]], *, t_max: float = 12.75) -> dict:
    """A cp34_tpe.json-shaped payload: ``method='tpe'`` + per-seed ``tpe_frontier``."""
    runs = [{"seed": s, "tpe_hv": 0.0, "rs_hv": 0.0, "complete": True, "tpe_frontier": pts}
            for s, pts in enumerate(seed_points)]
    return {"method": "tpe", "passes": True, "n_seeds": len(runs), "tpe_hv_mean": 0.0,
            "tpe_hv_std": 0.0, "rs_hv_mean": 0.0, "rs_hv_std": 0.0, "t_max_ms": t_max,
            "res": 640, "budget": 50, "structural": False, "complete": True, "runs": runs}


# ---- anchors -----------------------------------------------------------------

def test_read_anchor_assembles_from_split_latency_and_map_files(tmp_path):
    """Anchor A lives in two files (bench_model latency + pose_map accuracy); read_anchor
    joins them into one (acc, latency) point."""
    lat = tmp_path / "anchor_lat.json"
    mp = tmp_path / "anchor_map.json"
    lat.write_text(json.dumps({"latency_ms": {"mean": 12.755, "p50": 12.7}, "name": "n"}))
    mp.write_text(json.dumps({"map": 0.877, "map50": 0.94}))
    a = read_anchor(lat, mp)
    assert isinstance(a, Anchor)
    assert a.latency_ms == pytest.approx(12.755)
    assert a.acc == pytest.approx(0.877)


# ---- frontier loading --------------------------------------------------------

def test_load_frontier_unions_bo_and_tpe_and_tags_provenance(tmp_path):
    """The union frontier pulls every seed's points from both a bo- and a tpe-shaped
    payload, tagging each point with the method it came from."""
    bo = tmp_path / "cp33_bo.json"
    tpe = tmp_path / "cp34_tpe.json"
    bo.write_text(json.dumps(_bo_payload([[_pt(0.5, 7.0), _pt(0.6, 9.0)], [_pt(0.55, 8.0)]])))
    tpe.write_text(json.dumps(_tpe_payload([[_pt(0.62, 10.0)]])))

    pts = load_frontier([bo, tpe])
    assert len(pts) == 4                                   # 2 + 1 + 1
    methods = sorted({p["method"] for p in pts})
    assert methods == ["bo", "tpe"]
    assert all({"arch", "acc", "latency_ms", "acc_eff"} <= set(p) for p in pts)


def test_load_frontier_reads_bo_only_when_tpe_absent(tmp_path):
    """TPE is optional — a BO-only union is valid (the run before CP 3.4 lands)."""
    bo = tmp_path / "cp33_bo.json"
    bo.write_text(json.dumps(_bo_payload([[_pt(0.5, 7.0), _pt(0.65, 12.0)]])))
    pts = load_frontier([bo])
    assert len(pts) == 2
    assert {p["method"] for p in pts} == {"bo"}


# ---- selection: argmax J under the hard ceiling ------------------------------

def test_select_winner_argmaxes_J_and_excludes_over_ceiling():
    """α* maximises acc − λ·latency; a point above T_max (however accurate) is dropped."""
    fast, mid, slow = _pt(0.50, 7.0), _pt(0.60, 10.0), _pt(0.65, 12.0)
    over = _pt(0.99, 20.0)                                  # infeasible: latency > 12.75
    frontier = [fast, mid, slow, over]

    # small λ: accuracy dominates -> the slow, most-accurate feasible point wins
    assert select_winner(frontier, lam=0.01, t_max=12.75)["arch"] == slow["arch"]
    # large λ: latency dominates -> the fast point wins
    assert select_winner(frontier, lam=0.05, t_max=12.75)["arch"] == fast["arch"]
    # the over-ceiling point never wins despite acc=0.99
    for lam in (0.001, 0.01, 0.05, 0.2):
        assert select_winner(frontier, lam=lam, t_max=12.75)["arch"] != over["arch"]


def test_select_winner_raises_when_every_point_is_infeasible():
    """A ceiling below the whole frontier has no valid winner — raise, don't return junk."""
    frontier = [_pt(0.5, 7.0), _pt(0.6, 10.0)]
    with pytest.raises(ValueError, match="ceiling"):
        select_winner(frontier, lam=0.01, t_max=5.0)


def test_higher_lambda_never_selects_a_slower_arch():
    """Monotonicity of the trade-off: as λ (ms-penalty) rises, α*'s latency is
    non-increasing — the selector trades accuracy for speed, never the reverse."""
    frontier = [_pt(0.50, 7.0), _pt(0.58, 9.0), _pt(0.62, 11.0), _pt(0.65, 12.5)]
    lats = [select_winner(frontier, lam=lam, t_max=12.75)["latency_ms"]
            for lam in (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)]
    assert all(b <= a + 1e-12 for a, b in zip(lats, lats[1:], strict=False))


# ---- λ sensitivity sweep -----------------------------------------------------

def test_lambda_grid_is_log_centred_on_lambda():
    """The sweep grid brackets λ geometrically ([λ/span, λ·span]); its geometric centre
    is λ itself, so the committed exchange rate sits mid-sweep."""
    grid = lambda_grid(0.01, n=5, span=4.0)
    assert len(grid) == 5
    assert grid[0] == pytest.approx(0.01 / 4.0)
    assert grid[-1] == pytest.approx(0.01 * 4.0)
    assert grid[len(grid) // 2] == pytest.approx(0.01)     # odd n -> exact centre


def test_lambda_sensitivity_reports_one_winner_per_lambda():
    """The sweep returns an α* per λ so the report can show winner stability."""
    frontier = [_pt(0.50, 7.0), _pt(0.60, 10.0), _pt(0.65, 12.0)]
    sweep = lambda_sensitivity(frontier, t_max=12.75, lambdas=[0.005, 0.05])
    assert [row["lambda"] for row in sweep] == [0.005, 0.05]
    assert all({"lambda", "arch", "acc", "latency_ms", "J"} <= set(row) for row in sweep)
    # extreme λ endpoints straddle the front (slow at small λ, fast at large λ)
    assert sweep[0]["latency_ms"] > sweep[1]["latency_ms"]


# ---- serialisation: the winner-v1 export -------------------------------------

def test_winner_record_carries_arch_vector_lambda_anchors_and_proxy_caveat():
    """The exported record is self-describing: arch + its encode() vector, the λ and both
    anchors it was selected under, J, and the proxy-vs-deployable caveat."""
    winner = _pt(0.65, 12.0)
    a, b = Anchor(0.877, 12.755), Anchor(0.92, 30.0)
    rec = winner_record(winner, lam=0.0025, anchor_a=a, anchor_b=b, t_max=12.75,
                        sources=["cp33_bo.json"], sensitivity=[])
    assert rec["arch"] == winner["arch"]
    assert rec["vector"] == encode(winner["arch"])
    assert rec["lambda"] == pytest.approx(0.0025)
    assert rec["anchors"]["a"]["acc"] == pytest.approx(0.877)
    assert rec["anchors"]["b"]["latency_ms"] == pytest.approx(30.0)
    assert rec["J"] == pytest.approx(0.65 - 0.0025 * 12.0)
    assert "proxy" in rec["note"].lower()                  # honest scale caveat present


def test_serialize_winner_writes_a_reloadable_arch(tmp_path):
    """DoD groundwork: the serialized winner reloads in a clean read and its vector
    round-trips back to the arch (decode∘encode), so a fresh session can rebuild α*."""
    winner = _pt(0.65, 12.0)
    rec = winner_record(winner, lam=0.0025, anchor_a=Anchor(0.877, 12.755),
                        anchor_b=Anchor(0.92, 30.0), t_max=12.75, sources=[], sensitivity=[])
    out = serialize_winner(rec, tmp_path / "winner_v1")
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert decode(reloaded["vector"]) == winner["arch"]


# ---- CLI ---------------------------------------------------------------------

def test_main_dry_run_with_lambda_override_prints_without_writing(tmp_path, capsys):
    """`--dry-run --lambda X` selects from the frontier without anchor files and writes
    nothing — the useful-tonight mode (BO frontier + a hypothetical λ)."""
    bo = tmp_path / "cp33_bo.json"
    bo.write_text(json.dumps(_bo_payload([[_pt(0.50, 7.0), _pt(0.65, 12.0)]])))
    out_dir = tmp_path / "winner_v1"

    rc = main(["--frontier", str(bo), "--lambda", "0.01", "--dry-run",
               "--out-dir", str(out_dir)])
    assert rc == 0
    assert not out_dir.exists()                            # dry-run never serialises
    assert "lambda" in capsys.readouterr().out.lower()


def test_main_calibrates_lambda_from_anchor_files_and_writes_winner(tmp_path):
    """End-to-end: baseline + anchor-B files -> λ -> α* over the frontier -> winner.json."""
    bo = tmp_path / "cp33_bo.json"
    bo.write_text(json.dumps(_bo_payload([[_pt(0.50, 7.0), _pt(0.60, 10.0), _pt(0.65, 12.0)]])))
    base_lat = tmp_path / "baseline_anchor.json"
    base_map = tmp_path / "baseline_anchor_map.json"
    b_lat = tmp_path / "anchor_teacher.json"
    b_map = tmp_path / "anchor_teacher_map.json"
    base_lat.write_text(json.dumps({"latency_ms": {"mean": 12.755}}))
    base_map.write_text(json.dumps({"map": 0.877}))
    b_lat.write_text(json.dumps({"latency_ms": {"mean": 30.0}}))       # bigger -> slower
    b_map.write_text(json.dumps({"map": 0.92}))                        # bigger -> more accurate
    out_dir = tmp_path / "winner_v1"

    rc = main(["--frontier", str(bo), "--baseline-latency", str(base_lat),
               "--baseline-map", str(base_map), "--anchor-latency", str(b_lat),
               "--anchor-map", str(b_map), "--out-dir", str(out_dir)])
    assert rc == 0
    rec = json.loads((out_dir / "winner.json").read_text())
    # λ = (0.877-0.92)/(12.755-30) ≈ 0.0025 (tiny) -> accuracy dominates -> the 0.65 pt
    assert rec["acc"] == pytest.approx(0.65)
    assert rec["lambda"] == pytest.approx((0.877 - 0.92) / (12.755 - 30.0))
    assert rec["anchors"]["b"]["acc"] == pytest.approx(0.92)


def test_main_requires_lambda_or_both_anchor_files(tmp_path):
    """Without --lambda and without a full anchor B, λ is undefined — fail loudly."""
    bo = tmp_path / "cp33_bo.json"
    bo.write_text(json.dumps(_bo_payload([[_pt(0.5, 7.0)]])))
    with pytest.raises(SystemExit):
        main(["--frontier", str(bo), "--out-dir", str(tmp_path / "w")])
