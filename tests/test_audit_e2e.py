import json

from lut.orchestrate.audit_e2e import audit_row


def _row(tmp_path, name, mean, std, p50, p95, precision="fp16"):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({
        "name": name, "precision": precision,
        "latency_ms": {"mean": mean, "std": std, "p50": p50, "p95": p95, "n": 200},
    }))
    return p


def test_clean_row_is_not_suspect(tmp_path) -> None:
    # graft_r50_gtay_640_fp16 as actually measured: std 0.2 % of mean, p95 11 us off p50.
    r = audit_row(_row(tmp_path, "clean", 7.482, 0.0155, 7.481, 7.492))
    assert r is not None and r["suspect"] is False
    assert r["reasons"] == []


def test_contention_fingerprint_is_flagged(tmp_path) -> None:
    # prune_base_r45_640_fp16 as recorded 2026-07-08 — the row that was misdiagnosed as
    # "±20 % TRT build variance". A build choice shifts the mean and keeps std tight; this
    # has std at 14.7 % of mean and p95 3.4 ms above p50, which only contention explains.
    r = audit_row(_row(tmp_path, "r45", 7.185, 1.0582, 6.950, 10.333))
    assert r is not None and r["suspect"] is True
    assert len(r["reasons"]) == 2  # both the std blow-up and the detached tail
    assert r["rel_std_pct"] > 10.0
    assert r["tail_ms"] > 3.0


def test_tail_alone_flags_even_when_std_passes(tmp_path) -> None:
    # yolo11s_pose_640_fp16: std 2.2 % — over the floor, but the tail is the clearer tell.
    r = audit_row(_row(tmp_path, "s", 14.932, 0.3265, 14.870, 15.605))
    assert r is not None and r["suspect"] is True
    assert any("p95-p50" in why for why in r["reasons"])


def test_left_skew_is_flagged_too(tmp_path) -> None:
    # dense_ctrl_n_640_fp16 drifts the OTHER way (mean 8.111 BELOW p50 8.440). Whatever the
    # mechanism, an unstable run is an untrustworthy row — the check must not assume a
    # right-tail only, or it silently passes half the contaminated rows.
    r = audit_row(_row(tmp_path, "ctrl_n", 8.111, 0.7756, 8.440, 8.970))
    assert r is not None and r["suspect"] is True
    assert r["p50_vs_mean_pct"] < 0


def test_row_without_a_distribution_is_skipped(tmp_path) -> None:
    p = tmp_path / "no_dist.json"
    p.write_text(json.dumps({"name": "x", "latency_ms": {"mean": 1.0}}))
    assert audit_row(p) is None


def test_threshold_is_tunable(tmp_path) -> None:
    path = _row(tmp_path, "s", 14.932, 0.3265, 14.870, 15.605)
    tight = audit_row(path, rel_std_pct=1.0, tail_ms=10.0)
    loose = audit_row(path, rel_std_pct=5.0, tail_ms=10.0)
    assert tight is not None and loose is not None
    assert tight["suspect"] is True
    assert loose["suspect"] is False
