"""CP 8.2-early — distill/kd_loss.py structural KD loss (torch-only; teacher load needs
ultralytics and is exercised by the recovery smoke on .venv-nas / Kaggle)."""
import pytest

torch = pytest.importorskip("torch")

from distill.kd_loss import kd_map_loss  # noqa: E402


def _maps():
    return [torch.rand(2, 65, 8, 8), torch.rand(2, 65, 4, 4)], torch.rand(2, 24, 84)


def test_identical_structures_give_zero():
    x, k = _maps()
    assert float(kd_map_loss((x, k), (x, k))) == pytest.approx(0.0)


def test_mse_positive_and_teacher_detached():
    x, k = _maps()
    s0 = x[0].clone().requires_grad_(True)
    loss = kd_map_loss(([s0, x[1]], k), ([x[0] + 1.0, x[1]], k))
    assert float(loss) > 0.9  # MSE against a +1-shifted teacher ≈ 1
    loss.backward()
    assert s0.grad is not None  # grads flow to the student only


def test_structure_mismatches_refused():
    x, k = _maps()
    with pytest.raises(ValueError, match="mismatch"):
        kd_map_loss(x[0], torch.rand(2, 65, 4, 4))          # shape
    with pytest.raises(ValueError, match="mismatch"):
        kd_map_loss(([x[0]], k), (x, k))                    # length
    with pytest.raises(TypeError, match="unsupported"):
        kd_map_loss("boxes", "boxes")                       # node type


def test_head_dict_distills_protected_streams_and_skips_feats():
    # the real train-mode head emits {boxes, scores, feats, kpts}; feats channels differ
    # once the student is pruned → must be ignored, the other three must be matched.
    b, s, k = torch.rand(1, 64, 10), torch.rand(1, 1, 10), torch.rand(1, 24, 10)
    student = {"boxes": b, "scores": s, "kpts": k, "feats": [torch.rand(1, 32, 4, 4)]}
    teacher = {"boxes": b, "scores": s, "kpts": k, "feats": [torch.rand(1, 64, 4, 4)]}
    assert float(kd_map_loss(student, teacher)) == pytest.approx(0.0)  # feats ignored
    with pytest.raises(ValueError, match="missing stream"):
        kd_map_loss({"boxes": b}, {"boxes": b})
