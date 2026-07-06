"""CP 4.3 — net2net/bn.py: deepen + BN re-estimation preserves the function within 1e-3 (DoD)."""
import pytest
import torch
from torch import nn

from net2net.bn import bn_to_identity_, reestimate_bn
from net2net.deeper import identity_conv2d, inserted

torch.manual_seed(0)


def test_reestimate_matches_data_statistics() -> None:
    bn = nn.BatchNorm2d(4, momentum=0.1)
    model = nn.Sequential(bn)
    data = [3.0 + 2.0 * torch.randn(16, 4, 8, 8) for _ in range(20)]
    n = reestimate_bn(model, data)
    assert n == 20
    assert bn.running_mean is not None and bn.running_var is not None
    assert torch.allclose(bn.running_mean, torch.full((4,), 3.0), atol=0.2)
    assert torch.allclose(bn.running_var, torch.full((4,), 4.0), atol=0.4)
    assert bn.momentum == 0.1                 # momentum restored (mode restoration: next test)


def test_reestimate_restores_mode_and_rejects_empty() -> None:
    model = nn.Sequential(nn.BatchNorm2d(2))
    model.train()
    reestimate_bn(model, [torch.randn(4, 2, 3, 3)])
    assert model.training                     # original (train) mode restored
    model.eval()
    reestimate_bn(model, [torch.randn(4, 2, 3, 3)])
    assert not model.training                 # original (eval) mode restored
    with pytest.raises(ValueError, match="no batches"):
        reestimate_bn(model, [])
    with pytest.raises(ValueError, match="no BatchNorm"):
        reestimate_bn(nn.Sequential(nn.Conv2d(2, 2, 1)), [torch.randn(1, 2, 3, 3)])


def test_dod_deepen_with_bn_preserves_function_within_1e3() -> None:
    """The CP 4.3 DoD: insert identity-conv + fresh BN, re-estimate, invert → output preserved."""
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1)).eval()
    x = torch.randn(2, 3, 12, 12)
    ref = net(x)

    fresh_bn = nn.BatchNorm2d(8)
    deeper = inserted(net, 2, identity_conv2d(8), fresh_bn)
    reestimate_bn(deeper, [torch.randn(8, 3, 12, 12) for _ in range(10)])
    bn_to_identity_(fresh_bn)
    deeper.eval()
    assert torch.allclose(deeper(x), ref, atol=1e-3)      # the DoD bar
    assert torch.allclose(deeper(x), ref, atol=1e-5)      # eval-mode inversion is in fact exact


def test_bn_to_identity_requires_affine_and_stats() -> None:
    with pytest.raises(ValueError):
        bn_to_identity_(nn.BatchNorm2d(4, affine=False))
    with pytest.raises(ValueError):
        bn_to_identity_(nn.BatchNorm2d(4, track_running_stats=False))
