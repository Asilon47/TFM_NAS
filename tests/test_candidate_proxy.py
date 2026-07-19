"""eval/candidate_proxy.py — pure parts (the heavy path needs CUDA + dataset)."""
import json

from eval.candidate_proxy import done_keys, load_candidates

CAND = {"arch": {"ks": [3] * 20, "e": [4] * 20, "d": [2] * 5},
        "spec": {"stage_ratios": [0.2] * 5, "rest_ratio": 0.1},
        "neck": None, "imgsz": 160}


def test_load_candidates_accepts_bare_and_result_rows(tmp_path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps([
        CAND,                                            # bare candidate
        {"status": "ok", "key": "k1", "candidate": CAND},        # screen result row
        {"status": "infeasible", "key": "k2", "candidate": CAND},  # cycles-infeasible -> skip
        {"key": None, "candidate": CAND, "anchor": True},        # anchor row (no status)
    ]))
    got = load_candidates(f)
    assert len(got) == 3                                 # the infeasible row is dropped
    assert all(c["arch"]["d"] == [2] * 5 for c in got)


def test_done_keys_resume(tmp_path) -> None:
    out = tmp_path / "rows.jsonl"
    assert done_keys(out) == set()
    out.write_text('{"key": "abc", "proxy_map": 0.5}\n{"key": "def", "proxy_map": 0.6}\n')
    assert done_keys(out) == {"abc", "def"}


def test_proxy_one_pins_the_pruned_proxy_protocol() -> None:
    """Source pin: head UNFROZEN (DepGraph refuses frozen params — pruning slices through
    the head, so there is no intact donor head to freeze), l2 importance, rest_ratio from
    the spec, prebuilt short-FT path."""
    import inspect

    from eval.candidate_proxy import proxy_one

    src = inspect.getsource(proxy_one)
    assert "freeze_head=False" in src
    assert 'importance="l2"' in src
    assert 'ratio=float(c["spec"]["rest_ratio"])' in src
    assert "prebuilt=model" in src
