"""Arm K — distill/kd_feat.py: FitNets feature-mimic at the head-input taps (torch-only)."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from distill.kd_feat import FeatureKD, build_feature_kd, head_input_channels  # noqa: E402


class _Head(nn.Module):
    """Enough of the Ultralytics Pose head for the hook + channel-read contracts."""

    def __init__(self, chs):
        super().__init__()
        self.cv2 = nn.ModuleList(nn.Sequential(nn.Conv2d(c, 8, 1)) for c in chs)

    def forward(self, x):
        return [seq(t) for seq, t in zip(self.cv2, x, strict=True)]


def _model(chs):
    from types import SimpleNamespace

    return SimpleNamespace(model=[_Head(chs)])


def _taps(chs, size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, c, size, size, generator=g) for c in chs]


def test_head_input_channels_reads_first_convs():
    assert head_input_channels(_model([48, 96, 160])) == [48, 96, 160]


def test_matched_widths_use_identity_and_zero_loss():
    student, teacher = _model([64, 128, 256]), _model([64, 128, 256])
    kd = build_feature_kd(student, teacher)
    assert all(isinstance(p, nn.Identity) for p in kd.proj)
    same = _taps([64, 128, 256])
    student.model[-1](same)          # pre-hooks capture the head INPUTS
    teacher.model[-1](same)
    assert float(kd.loss()) == pytest.approx(0.0)


def test_mismatched_widths_project_and_train():
    student, teacher = _model([32, 64, 96]), _model([64, 128, 256])
    kd = build_feature_kd(student, teacher)
    assert all(isinstance(p, nn.Conv2d) for p in kd.proj)
    student.model[-1]([t.requires_grad_(True) for t in _taps([32, 64, 96])])
    with torch.no_grad():
        teacher.model[-1](_taps([64, 128, 256], seed=1))
    loss = kd.loss()
    assert float(loss) > 0.0
    loss.backward()
    assert kd.proj[0].weight.grad is not None   # regressors train with the student


def test_loss_requires_both_forwards_and_clears_taps():
    student, teacher = _model([64, 128, 256]), _model([64, 128, 256])
    kd = build_feature_kd(student, teacher)
    with pytest.raises(RuntimeError, match="both forwards"):
        kd.loss()
    same = _taps([64, 128, 256])
    student.model[-1](same)
    teacher.model[-1](same)
    kd.loss()
    with pytest.raises(RuntimeError, match="both forwards"):
        kd.loss()                     # taps consumed — a stale pair must not be reused


def test_detach_hooks_stops_capture():
    student, teacher = _model([64, 128, 256]), _model([64, 128, 256])
    kd = build_feature_kd(student, teacher)
    kd.detach_hooks()
    student.model[-1](_taps([64, 128, 256]))
    teacher.model[-1](_taps([64, 128, 256]))
    with pytest.raises(RuntimeError, match="both forwards"):
        kd.loss()


def test_scale_count_mismatch_raises():
    with pytest.raises(ValueError, match="scale count"):
        FeatureKD([64, 128], [64, 128, 256])


def test_regressors_stay_out_of_the_student():
    """The deploy contract: KD regressors live in the side module only — the student's
    state_dict (what gets saved/exported) must not grow."""
    student, teacher = _model([32, 64, 96]), _model([64, 128, 256])
    before = set(student.model[-1].state_dict())
    build_feature_kd(student, teacher)
    assert set(student.model[-1].state_dict()) == before


def test_recovery_loop_wiring_source_pins():
    """The heavy loop needs .venv-nas — pin the wiring: regressor params join the
    optimizer, the loss adds the weighted feat term, and a feat_kd without teacher fails."""
    import inspect

    from prune.prune_baseline import recovery_finetune

    src = inspect.getsource(recovery_finetune)
    assert "train_params += list(feat_kd.parameters())" in src
    assert "kd_feat_alpha * feat" in src
    assert "feat_kd needs a teacher" in src

    from prune.recover_graft import graft_prune_train_ladder

    src = inspect.getsource(graft_prune_train_ladder)
    assert src.index("prune_graft(") < src.index("build_feature_kd(model, teacher)")
    assert src.index("build_feature_kd(model, teacher)") < src.index(
        "metrics = recovery_finetune(")
    assert "detach_hooks()" in src                 # saved/exported student carries no hooks
    assert 'tag += "_kdf" if kd_feat else "_kd"' in src
