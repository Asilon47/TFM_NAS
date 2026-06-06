"""Sample subnets from the OFA-MBv3-w1.0 supernet with inherited weights.

Implements PROJECT_PLAN.md CP 1.3: turn a canonical OFA arch dict
(``{"ks": [...], "e": [...], "d": [...]}``) into a runnable PyTorch
``nn.Module`` whose weights come from CP 1.2's cached pretrained
checkpoint at ``~/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0``.

We bypass ``ofa.model_zoo`` and instantiate ``OFAMobileNetV3`` directly:

- ``ofa.model_zoo.ofa_net(pretrained=True)`` redownloads into
  ``.torch/ofa_nets/`` *relative to CWD*, which defeats CP 1.2's
  ``~/.cache/ofa/`` + SHA-pin contract.
- ``ofa/model_zoo.py:3`` top-imports ``gdown``; we don't otherwise need it.

Run as a script for the CP 1.3 DoD smoke test::

    python -m supernet.sampler
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3

from supernet.download_ofa import CHECKPOINT_PATH

# Constructor kwargs that match the pretrained
# ``ofa_mbv3_d234_e346_k357_w1.0`` checkpoint.
# Mirrors ``ofa/model_zoo.py:72-79``.
OFA_MBV3_W10_KWARGS: dict = {
    "dropout_rate": 0,
    "width_mult": 1.0,
    "ks_list": [3, 5, 7],
    "expand_ratio_list": [3, 4, 6],
    "depth_list": [2, 3, 4],
}

_supernet_cache: OFAMobileNetV3 | None = None


def load_supernet(checkpoint_path: Path = CHECKPOINT_PATH) -> OFAMobileNetV3:
    """Instantiate OFA-MBv3-w1.0 and load CP 1.2's pretrained weights."""
    supernet = OFAMobileNetV3(**OFA_MBV3_W10_KWARGS)
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["state_dict"]
    # OFAMobileNetV3.load_state_dict overrides nn.Module's default to
    # remap legacy ProxylessNAS-era keys (.mobile_inverted_conv. → .conv.
    # etc.) onto the elastic-NN module names.
    supernet.load_state_dict(state)
    supernet.eval()
    return supernet


def _get_or_load_supernet(
    supernet: OFAMobileNetV3 | None,
) -> OFAMobileNetV3:
    global _supernet_cache
    if supernet is not None:
        return supernet
    if _supernet_cache is None:
        _supernet_cache = load_supernet()
    return _supernet_cache


def sample(
    arch_dict: dict,
    supernet: OFAMobileNetV3 | None = None,
) -> nn.Module:
    """Materialise a subnet from ``{"ks": [...], "e": [...], "d": [...]}``.

    Returns a freshly deep-copied ``MobileNetV3`` that owns its weights;
    the caller may train it without touching the supernet's parameters.
    """
    sn = _get_or_load_supernet(supernet)
    sn.set_active_subnet(ks=arch_dict["ks"], e=arch_dict["e"], d=arch_dict["d"])
    return sn.get_active_subnet(preserve_weight=True)


def random_arch(supernet: OFAMobileNetV3 | None = None) -> dict:
    """Draw a uniform-random arch from the supernet's elasticity envelope."""
    return _get_or_load_supernet(supernet).sample_active_subnet()


if __name__ == "__main__":
    supernet = load_supernet()
    arch = random_arch(supernet)
    subnet = sample(arch, supernet)
    with torch.no_grad():
        out = subnet(torch.randn(1, 3, 224, 224))
    print(f"arch: {arch}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"params: {sum(p.numel() for p in subnet.parameters()):,}")
