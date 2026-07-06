"""eval/graft_ablate.py — CP 5.2 pure logic + resumable orchestration (stubbed fine-tune)."""
import json

import pytest

from eval.graft_ablate import (
    CORE_VARIANTS,
    VARIANTS,
    assemble_report,
    gates_from_state_dict,
    run_ablation,
    summarize,
    v3_warranted,
)

ARCH = {"ks": [3] * 20, "e": [4] * 20, "d": [2, 2, 4, 3, 3]}


def test_variant_table_shape() -> None:
    assert VARIANTS["v0_control"] == {}
    assert VARIANTS["v1_net2wider"] == {"adapter_init": "net2wider"}
    assert VARIANTS["v2_topdown"]["neck"] == "topdown"
    assert VARIANTS["v3_pan"]["neck"] == "pan"
    assert CORE_VARIANTS == ("v0_control", "v1_net2wider", "v2_topdown")  # V3 is conditional


def test_summarize_and_v3_gate() -> None:
    s = summarize({"v1_net2wider": [0.60, 0.61, 0.62], "v2_topdown": [0.64, 0.65, 0.66]})
    assert s["v1_net2wider"]["mean"] == pytest.approx(0.61)
    assert v3_warranted(s)                                    # clear win: Δ=0.03 > 1σ≈0.008
    tie = summarize({"v1_net2wider": [0.60, 0.62], "v2_topdown": [0.61, 0.63]})
    assert not v3_warranted(tie)                              # Δ=0.01 == 1σ → not strict win
    with pytest.raises(ValueError, match="v2_topdown"):
        v3_warranted({"v1_net2wider": {"mean": 0.6, "std": 0.0}})
    with pytest.raises(ValueError, match="no mAPs"):
        summarize({"v0_control": []})


def test_gates_from_state_dict() -> None:
    import torch

    sd = {"model.2.g54": torch.tensor(0.31), "model.2.g43": torch.tensor(-0.02),
          "model.2.lat54.weight": torch.zeros(1), "model.0.first_conv.weight": torch.zeros(1)}
    assert gates_from_state_dict(sd) == {"g54": pytest.approx(0.31), "g43": pytest.approx(-0.02)}
    assert gates_from_state_dict({"model.0.w": torch.zeros(1)}) == {}


def test_run_ablation_resumes_from_cache(tmp_path) -> None:
    cache = tmp_path / "graft_ablate_e5_r640.jsonl"
    cache.write_text(json.dumps(
        {"variant": "v0_control", "seed": 1, "map": 0.55, "map50": 0.9}) + "\n")
    calls: list[tuple[str, int]] = []

    def stub(arch, *, seed, graft_kwargs, **kw):
        name = next(n for n, k in VARIANTS.items() if k == graft_kwargs)
        calls.append((name, seed))
        return {"map": 0.60 + 0.01 * seed, "map50": 0.9}

    results = run_ablation(ARCH, head_weights="donor.pt", variants=("v0_control",),
                           seeds=(1, 2), cache=cache, finetune_fn=stub)
    assert calls == [("v0_control", 2)]                       # seed 1 came from the cache
    assert results["v0_control"]["maps"] == [0.55, pytest.approx(0.62)]
    assert len(cache.read_text().splitlines()) == 2           # the new row was appended

    calls.clear()                                             # a full re-run costs nothing
    run_ablation(ARCH, head_weights="donor.pt", variants=("v0_control",),
                 seeds=(1, 2), cache=cache, finetune_fn=stub)
    assert calls == []


def test_run_ablation_passes_variant_kwargs_and_freezes_head() -> None:
    seen: list[dict] = []

    def stub(arch, **kw):
        seen.append(kw)
        return {"map": 0.6, "map50": 0.9}

    run_ablation(ARCH, head_weights="donor.pt", variants=("v2_topdown",), seeds=(1,),
                 finetune_fn=stub)
    kw = seen[0]
    assert kw["graft_kwargs"] == {"adapter_init": "net2wider", "neck": "topdown"}
    assert kw["freeze_head"] is True and kw["head_weights"] == "donor.pt"
    with pytest.raises(ValueError, match="unknown variant"):
        run_ablation(ARCH, head_weights="d.pt", variants=("v9",), finetune_fn=stub)


def test_assemble_report_deltas_and_v3_flag() -> None:
    results: dict[str, dict] = {
        "v0_control": {"maps": [0.58, 0.59, 0.60], "gates": {}},
        "v1_net2wider": {"maps": [0.60, 0.61, 0.62], "gates": {}},
        "v2_topdown": {"maps": [0.64, 0.65, 0.66], "gates": {"1": {"g54": 0.3, "g43": 0.1}}},
    }
    r = assemble_report(ARCH, results, seeds=(1, 2, 3), epochs=5, imgsz=640, batch=16)
    assert r["v1_vs_v0_delta"] == pytest.approx(0.02)
    assert r["v2_vs_v1_delta"] == pytest.approx(0.04)
    assert r["v3_warranted"] is True
    assert r["variants"]["v2_topdown"]["gates"]["1"]["g54"] == 0.3
    assert r["protocol"]["freeze_head"] is True
