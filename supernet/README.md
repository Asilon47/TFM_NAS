# `supernet/`

Wraps the MIT-HAN-Lab Once-for-All supernet so the rest of the project
can sample subnets without depending on OFA's repo layout directly.

> **Note on naming:** This package is called `supernet/` (not `ofa/`) to avoid
> shadowing the pip-installed `ofa` library, which is a direct dependency.

## Status (CP 1.3)

The OFA dependency is pinned (CP 1.1), the pretrained MBv3 supernet
checkpoint is downloaded + hash-verified into `~/.cache/ofa/`
(CP 1.2), and `sampler.py` materialises subnets from canonical OFA
arch dicts with weights inherited from the cached checkpoint
(CP 1.3).

## Pinned dependencies

The NAS-side venv (`.venv-nas/`) installs:

| Package | Pin | Why |
|---|---|---|
| `ofa` | `==0.1.0.post202307202001` | Latest PyPI release of MIT-HAN-Lab Once-for-All. Pure-Python wheel; only declares `Requires-Dist: torch` with no upper bound, so it works with modern torch. Verified via `pip index versions ofa`. |
| `torch` | `>=2.3,<2.12` | GPU build (`+cu128` local segment when installed via `setup_laptop_nas.sh`). Lower bound matches the original LUT pipeline's torch; upper bound brackets the laptop's already-installed `torch==2.11.0+cu128`. |
| `torchvision` | `>=0.18` | Imported transitively by `ofa.utils.my_dataloader.my_random_resize_crop` (`import torchvision.transforms.functional as F`). Must be the matching `+cu128` build — see install note in `requirements-nas.txt`. |
| `Pillow` | `>=10.0` | Same import chain — `from PIL import Image` is the *first* failing import without it. |

The full pin list lives in `requirements-nas.txt` at the repo root.

**Why no `gdown`?** Only `ofa.model_zoo` top-imports `gdown`. The
sampler bypasses `ofa.model_zoo` (see "Why we bypass `ofa.model_zoo`"
below), so the dep is unnecessary.

## What this package holds

| File | Checkpoint | Purpose |
|---|---|---|
| `download_ofa.py` | CP 1.2 ✅ | Pulls a pretrained OFA-MBv3 checkpoint into `~/.cache/ofa/` and verifies SHA256. |
| `sampler.py` | CP 1.3 ✅ | `sample(arch_dict) -> nn.Module` — turns an OFA arch spec into a runnable subnet with inherited weights. |

### Sampler usage

```python
from supernet.sampler import load_supernet, sample, random_arch

supernet = load_supernet()                  # one-time, 31 MB load
arch = random_arch(supernet)                # {'ks': [...], 'e': [...], 'd': [...]}
subnet = sample(arch, supernet)             # nn.Module with pretrained weights
y = subnet(torch.randn(1, 3, 224, 224))     # ImageNet logits, shape (1, 1000)
```

The supernet argument is optional — when omitted, `sample()` and
`random_arch()` lazy-load it once into a module-level cache so the
Phase 3 BO loop's per-arch overhead is a `set_active_subnet` +
`deepcopy`, not a 31 MB re-load.

### Why we bypass `ofa.model_zoo`

CP 1.3 instantiates `OFAMobileNetV3` directly and loads CP 1.2's
cached `state_dict`, rather than calling
`ofa.model_zoo.ofa_net(pretrained=True)`. Two reasons:

1. **Cache contract.** `model_zoo.ofa_net` redownloads to
   `.torch/ofa_nets/` *relative to CWD*, defeating CP 1.2's
   `~/.cache/ofa/` + SHA-pin invariant.
2. **One fewer dep.** `ofa/model_zoo.py:3` top-imports `gdown` for
   its Google Drive code paths; we don't otherwise need that package.

The duplicated load logic is three lines (`OFAMobileNetV3(**kwargs)`
+ `torch.load(...)["state_dict"]` + `load_state_dict`), and decoupling
our cache contract from upstream's behaviour is a long-term win.

## Pinned OFA checkpoint

| Field | Value |
|---|---|
| Net ID | `ofa_mbv3_d234_e346_k357_w1.0` |
| URL | `https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0` |
| Cache path | `~/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0` |
| Size | 31.0 MB (31,011,816 bytes) |
| SHA256 | `a7def36bb4e4c688c16d37eb60d5d34b2e6dcf6438c05bc86dea918fda04c6c7` |
| File format | `torch.save` dict, top-level key `state_dict` (475 tensors) |
| Source upstream | `ofa.model_zoo.ofa_net("ofa_mbv3_d234_e346_k357_w1.0")` |

The pin lives in `supernet/download_ofa.py::PINNED_SHA256`. The
script is idempotent: it re-downloads only when the file is missing
or `--force` is passed, and a hash mismatch fails loudly so an
upstream rotation cannot silently reach a training run.

### Why w1.0 (not w1.2)?

The plan (`PROJECT_PLAN.md:77-79`) accepts either. w1.0 is the
canonical OFA-MBv3 baseline, has the most third-party validation, and
matches the MobileNetV3-large width factor we benchmark against. w1.2
becomes interesting once CP 5.1 widens the supernet anyway; starting
narrower keeps the pretrained-vs-search-space question simpler.

## Local install

```bash
bash scripts/setup_laptop_nas.sh
source .venv-nas/bin/activate
python -c "import ofa; print(ofa.__file__)"      # CP 1.1 DoD
python supernet/download_ofa.py                  # CP 1.2 DoD
python -m supernet.sampler                       # CP 1.3 DoD
```
