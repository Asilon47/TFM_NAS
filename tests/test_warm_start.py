"""Tests for the head warm-start + freeze helpers (CP 2.4 proxy repair).

``warm_start_head`` / ``freeze_module`` are generic torch ``state_dict`` logic — no ultralytics —
so they run under ``.venv`` / CI. (The real Pose-head clone from a ``.pt`` via ``_donor_head_state``
is GPU/.venv-nas-gated and smoke-only.) Stub ``nn.Module``s stand in for the Pose head: the copy
rule is purely key+shape matching, so a small conv stack exercises it faithfully — including the
gate-vs-COCO keypoint-branch mismatch (cv4 is 8*3 vs 17*3 channels).
"""
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from detect.pose_model import freeze_module, warm_start_head  # noqa: E402


def _head(co_kpt: int) -> nn.Module:
    """A tiny stand-in 'head': box + cls branches (fixed shape) and a keypoint branch whose
    out-channels vary with the keypoint count — mirrors how a real Pose head's cv4 changes for
    8 (gate) vs 17 (COCO) keypoints."""
    return nn.Sequential(
        nn.Conv2d(16, 16, 1),      # '0' box branch
        nn.Conv2d(16, 1, 1),       # '1' cls branch (nc=1)
        nn.Conv2d(16, co_kpt, 1),  # '2' kpt branch (co_kpt = nkpt * 3)
    )


def test_warm_start_copies_all_when_shapes_match():
    target = _head(co_kpt=24)  # 8 keypoints (gate)
    donor = _head(co_kpt=24)   # same shape → whole head transfers
    with torch.no_grad():
        for p in donor.parameters():
            p.fill_(0.5)
    report = warm_start_head(target, donor.state_dict())
    assert report["skipped"] == []
    assert len(report["copied"]) == len(target.state_dict())
    for t in target.parameters():
        assert torch.allclose(t, torch.full_like(t, 0.5))  # now equals the donor


def test_warm_start_skips_mismatched_shapes():
    target = _head(co_kpt=24)  # gate: 8 kpts → cv4 out=24
    donor = _head(co_kpt=51)   # COCO: 17 kpts → cv4 out=51, mismatches at branch '2'
    report = warm_start_head(target, donor.state_dict())
    assert any(k.startswith("2.") for k in report["skipped"])      # kpt branch skipped
    assert all(not k.startswith("2.") for k in report["copied"])
    assert report["copied"]  # box + cls still transferred


def test_warm_start_leaves_unmatched_params_at_init():
    target = _head(co_kpt=24)
    before = target[2].weight.detach().clone()  # the kpt branch we won't overwrite
    warm_start_head(target, _head(co_kpt=51).state_dict())  # mismatched cv4
    assert torch.equal(target[2].weight, before)  # untouched (stayed at init)


def test_warm_start_raises_when_nothing_matches():
    target = _head(co_kpt=24)
    donor = nn.Conv2d(3, 7, 1)  # unrelated module → no key overlap at all
    with pytest.raises(ValueError, match="no head tensors matched"):
        warm_start_head(target, donor.state_dict())


def test_freeze_module_disables_grad_and_optimizer_filter():
    head = _head(co_kpt=24)
    freeze_module(head)
    assert all(not p.requires_grad for p in head.parameters())
    # the 'train only trainable params' filter short_finetune uses must exclude the frozen head.
    model = nn.Sequential(nn.Conv2d(3, 16, 1), head)  # conv0 trainable, head frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 2  # conv0 weight + bias only
