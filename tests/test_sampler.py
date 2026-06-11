"""CP 1.3 smoke as pytest: sample a random subnet and forward it.

Needs the ``ofa`` package (installed only in .venv-nas) and the CP 1.2
pinned checkpoint; skips cleanly everywhere else (.venv, CI).
"""
import pytest

pytest.importorskip("ofa", reason="ofa is only installed in .venv-nas")

import torch  # noqa: E402

from supernet import sampler  # noqa: E402
from supernet.download_ofa import CHECKPOINT_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="OFA checkpoint not cached (run python -m supernet.download_ofa)",
)


def test_sampled_subnet_forwards():
    supernet = sampler.load_supernet()
    arch = sampler.random_arch(supernet)
    subnet = sampler.sample(arch, supernet)
    with torch.no_grad():
        out = subnet(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 1000)
