"""prune/prune_baseline.py — pure parts of the CP 6.2-B control-arm ladder."""
import pytest

from prune.prune_baseline import TECHNIQUES, assemble_ladder_report, ladder_plan, run_tag


def test_ladder_plan_canonicalizes() -> None:
    assert ladder_plan([0.45, 0.15, 0.30, 0.15]) == [0.15, 0.30, 0.45]


def test_techniques_ladder_vocabulary() -> None:
    """The CP 6.2-G technique names → prune_graft knobs; 'uniform' must stay the floor config
    (per-layer magnitude — every pre-program artifact was produced by it)."""
    assert set(TECHNIQUES) == {"uniform", "global_l2", "global_taylor"}
    assert TECHNIQUES["uniform"] == {"global_pruning": False, "importance": "l2"}
    assert TECHNIQUES["global_taylor"]["importance"] == "taylor"
    assert all(TECHNIQUES[t]["global_pruning"] for t in ("global_l2", "global_taylor"))


def test_run_tag_default_keeps_legacy_names() -> None:
    # prune_base_r15.pt / recover_graft_r60.pt etc. were tagged pre-program — the default
    # point must keep producing exactly those names.
    assert run_tag(0.15) == "r15"
    assert run_tag(0.60) == "r60"


def test_run_tag_encodes_technique_iter_seed() -> None:
    assert run_tag(0.50, technique="global_l2") == "r50_gl2"
    assert run_tag(0.50, technique="global_taylor", iterative_steps=3) == "r50_gtay_it3"
    assert run_tag(0.60, seed=2) == "r60_s2"
    with pytest.raises(ValueError, match="technique"):
        run_tag(0.5, technique="nope")
    with pytest.raises(ValueError, match="iterative_steps"):
        run_tag(0.5, iterative_steps=0)


def test_ladder_plan_guards() -> None:
    with pytest.raises(ValueError, match="empty"):
        ladder_plan([])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ladder_plan([0.15, 1.0])


def test_assemble_ladder_report_deltas_and_best() -> None:
    donor = {"map": 0.877, "params": 2_900_000}
    rows = [
        {"ratio": 0.30, "params": 2_000_000, "map": 0.850},
        {"ratio": 0.15, "params": 2_400_000, "map": 0.870},
    ]
    rep = assemble_ladder_report(donor, rows)
    assert [r["ratio"] for r in rep["rows"]] == [0.15, 0.30]          # sorted
    assert rep["rows"][0]["delta_map_vs_donor"] == pytest.approx(-0.007)
    assert rep["best_row_ratio"] == 0.15
    assert "measured-only" in rep["note"]


def test_assemble_ladder_report_needs_donor_map() -> None:
    with pytest.raises(ValueError, match="donor"):
        assemble_ladder_report({}, [])


def test_trace_imgsz_stays_small() -> None:
    """DepGraph tracing holds every activation via grad_fn: tracing at the 640 deploy
    size OOM-killed Kaggle's ~13 GB host (rc=137, 2026-07-07). The groups + data-free
    importance are resolution-independent — keep the trace at the DoD-tested scale."""
    from prune.prune_baseline import TRACE_IMGSZ

    assert 32 <= TRACE_IMGSZ <= 160


# donor-dependent guards for the 2026-07-08 criterion/args reset (Kaggle rc=1). Gated on
# ultralytics + the trained donor being present → run locally (.venv-nas), skip in CI.
_DONOR = __import__("pathlib").Path(
    "runs/pose/experiments/gate_baseline/weights/best.pt")


@pytest.mark.skipif(not _DONOR.exists(), reason="gate-trained donor not present")
def test_load_baseline_model_resets_loss_plumbing() -> None:
    pytest.importorskip("ultralytics")
    from prune.prune_baseline import load_baseline_model

    model = load_baseline_model(_DONOR)
    # criterion dropped so .loss() re-inits it on the model's CURRENT device (proj is a plain
    # CPU tensor the checkpoint cached; model.to("cuda") would strand it → device mismatch).
    assert model.criterion is None
    # args is a namespace (not the checkpoint's dict) so the fresh criterion's hyp.box works.
    assert hasattr(model.args, "box")
    assert not isinstance(model.args, dict)


def test_train_ckpt_roundtrip_and_stale_tolerance(tmp_path) -> None:
    """Free-tier resume (2026-07-13): save/load round-trips epoch+step+weights atomically;
    a shape-drifted ckpt is refused WITHOUT touching the optimizer (fresh-start signal)."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from prune.prune_baseline import _load_train_ckpt, _save_train_ckpt

    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # one step so the optimizer has real state
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    ckpt = tmp_path / "ckpt_r50_gtay_kd.pt"
    _save_train_ckpt(ckpt, model, opt, epoch=29, step=1234)
    assert ckpt.exists() and not ckpt.with_suffix(".tmp").exists()   # atomic rename

    fresh = nn.Linear(4, 2)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    resumed = _load_train_ckpt(ckpt, fresh, fresh_opt)
    assert resumed == (30, 1234)
    assert torch.equal(fresh.weight, model.weight)

    drifted = nn.Linear(8, 2)          # pruned differently between sessions
    drifted_opt = torch.optim.AdamW(drifted.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in drifted.state_dict().items()}
    assert _load_train_ckpt(ckpt, drifted, drifted_opt) is None
    assert torch.equal(drifted.weight, before["weight"])             # untouched on refusal
