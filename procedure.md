# Procedure Journal

A growing, checkpoint-by-checkpoint record of **what was done and why**.
Sister document to:

- `PROJECT.md` — the vision (why this project exists at all).
- `PROJECT_PLAN.md` — the plan (the checkpoint catalog).
- `state/plan_state.yaml` — the state (which checkpoint we're at).

Each section here is the narrative for a single checkpoint: every file
created, every decision taken, every command run for verification.

---

## CP 1.1 — Skeleton repo + state file

**Date:** 2026-04-25
**Source spec:** `PROJECT_PLAN.md:66-73`
**DoD:** `python -c "import ofa; print(ofa.__file__)"` works.

### What was done

1. Created five empty Python packages:
   - `supernet/__init__.py`
   - `search/__init__.py`
   - `eval/__init__.py`
   - `net2net/__init__.py`
   - `expand/__init__.py`
2. Created the state directory and its first state file:
   - `state/plan_state.yaml` (with `current_checkpoint: "1.1"`,
     `last_completed: "1.1"`, and a list of completed checkpoints).
3. Created `requirements-nas.txt` at the repo root, pinning the
   NAS-side dependency set (GPU `torch`, `numpy`, `pyyaml`, `ofa`).
4. Created `supernet/README.md` — a stub that documents the
   pinned `ofa` version and reserves space for the OFA-checkpoint pin
   that CP 1.2 will fill in.
5. Appended `.venv-nas/` to `.gitignore`.
6. Created `scripts/setup_laptop_nas.sh` — a sibling of the existing
   `scripts/setup_laptop.sh` that builds the new NAS venv and runs
   the DoD import check.
7. Ran `bash scripts/setup_laptop_nas.sh` to create `.venv-nas/`,
   install the pinned deps, and confirm the DoD command succeeds.
8. Authored this entry in `procedure.md`.

### Why each piece

#### Why five empty packages, why not just one?

`PROJECT_PLAN.md` lists the five packages the project will grow into:

| Dir | Purpose | First populated at |
|---|---|---|
| `supernet/` | Wraps the OFA supernet; produces `nn.Module` from arch dict | CP 1.2 |
| `search/` | Search-loop code (encoder, BO, NSGA-II, cost) | CP 2.1 |
| `eval/` | Short fine-tune harness + final long-train | CP 2.4 |
| `net2net/` | Function-preserving widen/deepen + diff | CP 4.1 |
| `expand/` | Cross-family op injection + LUT pre-screen | CP 5.0 |

Creating all five up-front is cheap (5 empty `__init__.py` files), and
it lets future checkpoints land code without first having to
mkdir-and-touch. It also makes the layout legible at a glance —
`tree -L 1` immediately tells the reader the macro structure of the
NAS pipeline. `state/` is the only new directory **without** an
`__init__.py`, because it's a data directory (yaml + future model
manifests), not a Python package.

#### Why a separate `state/plan_state.yaml`?

`PROJECT_PLAN.md` opens with a "How to resume across sessions" note
that explicitly calls out `state/plan_state.yaml` as the third resume
input (after `PROJECT.md` and `PROJECT_PLAN.md`). It's the *only*
file in the repo that records *where we are*, vs. *what we're doing*
(plan) and *why we're doing it* (vision).

Schema kept minimal on purpose: `current_checkpoint`,
`last_completed`, a `completed:` list, and free-form `notes`. We'll
grow this only when later checkpoints need new fields (e.g. CP 1.2
will likely add a `cached_artifacts:` map listing the OFA checkpoint
path + SHA).

#### Why a separate `requirements-nas.txt`?

The existing `requirements.txt` pins `torch==2.3.1+cpu` because the
LUT-generation pipeline only needs CPU torch (it builds ONNX and
hands it off to the Jetson — see `README.md:7-12`). The NAS side
needs a **GPU build of torch** so subnets can be fine-tuned on the
laptop's dGPU.

Two options were considered:

- *Unify:* replace the LUT pipeline's CPU torch with a GPU build in
  `requirements.txt`. Rejected — it would silently change the
  contract documented in `setup_laptop.sh:18` ("CPU wheels for torch")
  and re-install the LUT pipeline's torch every time someone refreshes
  the venv.
- *Separate:* keep two requirement files, one venv per role.
  **Selected** because it isolates the two pipelines completely —
  each can be rebuilt independently, and a `pip install` mistake on
  one side can never break the other.

The user confirmed this preference via AskUserQuestion (option A:
"Separate `.venv-nas/` (Recommended)").

#### Why pin `ofa==0.1.0.post202307202001`?

This is the latest PyPI release of the MIT-HAN-Lab Once-for-All
package (verified via `pip index versions ofa`, which lists 28
versions; `0.1.0.post202307202001` from July 2023 is the newest).
The wheel only declares `Requires-Dist: torch` — no upper bound —
so it imports cleanly against modern torch (2.10 has been verified).

The user confirmed the latest-PyPI choice via AskUserQuestion. The
alternative — pinning to a specific git commit per `PROJECT_PLAN.md`'s
"fork-lock a known-good commit" risk note — is reserved for if/when
the PyPI wheel breaks against a future torch.

PROJECT_PLAN.md's risk callout about "Weight loader expects a specific
PyTorch version" is **not** load-bearing for CP 1.1 (which only
imports the package). It will become load-bearing at CP 1.2 (download
checkpoint) and especially CP 1.4 (verify the loaded weights match
the published accuracy).

#### Why pin `torch>=2.3,<2.12` rather than an exact version?

A range, not a pin, because:

- Lower bound `>=2.3` matches the LUT pipeline's `torch==2.3.1+cpu` —
  same major API.
- Upper bound `<2.12` brackets the version actually installed during
  CP 1.1 setup (`torch==2.11.0+cu130`, see "Verification" below).

The original pin was `<2.11`, picked to bracket the user's
already-installed `torch==2.10.0+cu128`. During the live install pip
resolved to `torch==2.11.0` (the new release on PyPI), so the upper
bound was bumped to `<2.12`. If torch 2.12+ ever breaks ofa, lower
it; if a future ofa release demands 2.12+, raise it. Range pins give
us that flexibility without giving up reproducibility.

#### Why a new `setup_laptop_nas.sh` instead of editing `setup_laptop.sh`?

`setup_laptop.sh` is a stable artifact for the LUT pipeline — it's
referenced from `README.md:48-51`. Editing it would couple the LUT
pipeline's setup to NAS work that may take weeks of additional
checkpoints to be useful.

A sibling script keeps the two pipelines decoupled: the user can run
`scripts/setup_laptop.sh` to refresh the LUT venv (`.venv/`) without
ever touching the NAS venv, and vice versa. The two scripts share
shape (set -euo pipefail, venv creation, `pip install -r ...`,
sanity import) so they're easy to read side-by-side.

The `TORCH_CUDA_INDEX` env var (defaulted to `cu128`) lets the user
override the CUDA wheel index without editing the script — useful
because different machines have different CUDA toolkits.

#### Why `procedure.md` at the repo root?

The user asked for "procedure.md where you explain absolutely
everything done in detail with justification". A single growing file
at the repo root (rather than one-file-per-checkpoint under
`docs/procedures/`) means:

- One file to grep when answering "why did we do X?".
- The narrative reads chronologically — each new entry is appended,
  preserving the reasoning trail.
- It sits next to `PROJECT.md` / `PROJECT_PLAN.md` / `state/`, which
  makes the four-file resume protocol visually obvious.

### Decisions taken (via AskUserQuestion)

| Question | Choice | Reason |
|---|---|---|
| Where should NAS deps live? | Separate `.venv-nas/` | Isolates from LUT pipeline; reproducible pin file. |
| Which OFA version? | Latest PyPI (`0.1.0.post202307202001`) | Most recent release; pure-Python wheel; no torch upper bound. |

### Verification (DoD)

The CP 1.1 setup ran on 2026-04-25. First attempt via
`scripts/setup_laptop_nas.sh` hit a transient `ReadTimeoutError`
against `files.pythonhosted.org` while pulling the cuda12.8 torch
wheel (~3 GB). Re-ran with a longer pip timeout and without
`--extra-index-url`; pip resolved to the PyPI-default torch 2.11.0
which ships with cu130 nvidia libs as separate packages — this
version is one minor release newer than the user's pre-existing
`torch==2.10.0+cu128`, prompting the `<2.11` → `<2.12` pin bump
documented above.

Final verified state:

```
$ source .venv-nas/bin/activate

$ python -c "import ofa; print(ofa.__file__)"
/home/asil/Desktop/lookup_table/.venv-nas/lib/python3.12/site-packages/ofa/__init__.py

$ python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
torch: 2.11.0+cu130 cuda: False

$ python -c "import yaml; print('yaml:', yaml.__version__); import numpy; print('numpy:', numpy.__version__)"
yaml: 6.0.3
numpy: 2.4.4

$ python -c "import supernet, search, eval, net2net, expand; print('all 5 NAS packages import OK')"
all 5 NAS packages import OK

$ python -c "import yaml; d=yaml.safe_load(open('state/plan_state.yaml')); print(d['current_checkpoint'])"
1.1
```

**DoD satisfied:** `import ofa` succeeds and prints a path inside
`.venv-nas/`.

**Note for CP 1.2+:** `torch.cuda.is_available()` returned `False` on
this machine despite the cu130 wheels being installed — no NVIDIA
driver / GPU was visible at setup time (`nvidia-smi` not on PATH).
This is **not** a CP 1.1 blocker (DoD is just the `import`), but
fine-tuning subnets at CP 2.4 / CP 3.x will require a working CUDA
driver. Resolve before CP 2.4.

### What's next

CP 1.2 — OFA checkpoint download + cache. Plan:

1. Add `supernet/download_ofa.py` that pulls the canonical
   `ofa_mbv3_d234_e346_k357_w1.0` checkpoint into `~/.cache/ofa/`.
2. Lock its SHA256 in `supernet/README.md` so future-us can
   detect upstream changes.
3. Update `state/plan_state.yaml` with the cached artifact's path.

---

## CP 1.2 — OFA checkpoint download + cache

**Date:** 2026-04-25
**Source spec:** `PROJECT_PLAN.md:75-82`
**DoD:** Checkpoint file exists on disk, hash matches the pin.

### What was done

1. Wrote `supernet/download_ofa.py` — a stdlib-only downloader
   that pulls `ofa_mbv3_d234_e346_k357_w1.0` from MIT-HAN-Lab's
   GitHub mirror into `~/.cache/ofa/` and verifies SHA256 against a
   constant pinned at the top of the file.
2. Ran the script once with the SHA pin set to a placeholder, copied
   the actual computed digest into `PINNED_SHA256`, re-ran to verify.
   Final pin:
   `a7def36bb4e4c688c16d37eb60d5d34b2e6dcf6438c05bc86dea918fda04c6c7`.
3. Smoke-tested the download with `torch.load(weights_only=False)` —
   the file unpickles to a dict with key `state_dict` containing 475
   tensors (consistent with OFA-MBv3's parameter count).
4. Re-ran the script a third time to confirm idempotency (skips
   re-download when the cached hash matches the pin).
5. Updated `supernet/README.md` with a "Pinned OFA checkpoint"
   table (URL, cache path, size, SHA256, format) and a justification
   for choosing w1.0 over w1.2.
6. Advanced `state/plan_state.yaml` to `current_checkpoint: "1.2"` and
   added a new `cached_artifacts:` section recording the checkpoint's
   path + bytes + hash.
7. Authored this entry.

### Why each piece

#### Why a stdlib-only download instead of `ofa.model_zoo.ofa_net()`?

`ofa.model_zoo` would download the same file but with three baggage
items we don't want at CP 1.2:

| Issue | Detail |
|---|---|
| Transitive deps (`gdown`, `PIL`) | Importing `ofa.model_zoo` triggers `import gdown` (for the resnet50 path that uses Google Drive) and `from ofa.utils import download_url`, where `ofa.utils.__init__` pulls in `PIL` via the bundled dataloader. None of those are needed for the MBv3 mirror download — adding them now would conflate CP 1.2 (download + verify) with CP 1.3 (sampler) dep work. |
| Cache location | Upstream defaults to `.torch/ofa_nets/` *relative to the current working directory*. The plan calls for `~/.cache/ofa/` — a stable per-user location that survives `cd`. |
| No hash verification | Upstream's `download_url` uses `urlretrieve` and never hashes the result. CP 1.2's whole point is the hash pin. |
| Side-effect on import | `ofa.model_zoo.ofa_net(..., pretrained=True)` instantiates the full `OFAMobileNetV3` module *and* loads weights — far more work than CP 1.2 needs. |

A 70-line stdlib script (`urllib.request` + `hashlib` + `argparse`)
covers all of CP 1.2's DoD with zero new dependencies, and is small
enough that a reader can audit the whole file in 30 seconds.

#### Why w1.0 and not w1.2?

The plan (`PROJECT_PLAN.md:77-79`) explicitly accepts either; the
deciding factors:

- **Validation density.** w1.0 is the OFA-MBv3 baseline used in the
  ICLR 2020 paper and most downstream NAS work. w1.2 is published but
  cited far less, so any deviation we measure at CP 1.4 is harder to
  explain.
- **Forward compatibility with Phase 5.** CP 5.1 explicitly widens
  *from* w1.2 to w1.4/w1.6. Starting at w1.0 means we'll exercise
  Net2Wider over a larger range (w1.0 → w1.2 → w1.4) instead of
  beginning halfway up. The widening tests get more interesting.
- **Latency.** w1.0 is the narrowest published OFA-MBv3 supernet, so
  Phase 3's first search runs against the tightest latency baseline.
  Pareto-dominating MobileNetV3-large at w1.0 is a more credible
  "we're better than the obvious baseline" claim than dominating at
  w1.2.

#### Why pin the SHA256 in code, not in a YAML file?

Pinning in `download_ofa.py` itself means the verification check and
the pin live in one file — a change to either is a change to one
file's diff. A YAML pin would either need its own loader (extra code)
or a string-substitution build step (extra build complexity). The
constant is one line; reviewing a future bump is a single-line diff.

The pin is *also* echoed in `supernet/README.md` and in
`state/plan_state.yaml::cached_artifacts`. Those are documentation
mirrors, not the source of truth — when in doubt, trust
`download_ofa.py::PINNED_SHA256`.

#### Why a download "lock" + atomic rename pattern?

`download()` writes to `<dest>.part` and renames to `<dest>` only on
clean completion. If the download is interrupted (laptop sleep,
network blip), the next run sees no `<dest>` and re-downloads — it
never sees a partial file with the wrong hash and report it as a
real mismatch. This matters because the OFA mirror is a public CDN
endpoint with no resumable-download protocol; partial bytes are
indistinguishable from a corrupted file by hash alone.

#### Why no `--bootstrap` mode in the script?

The first run had a placeholder SHA in the pin, the script printed
the actual hash, I copied it into the constant, and re-ran. That
two-step bootstrap happened *once*, in this checkpoint. A
`--bootstrap` flag would be code-debt for a workflow that, by
definition, runs zero times after CP 1.2 lands.

If a future checkpoint adds a second pinned artifact (e.g. CP 5.3
adds the FusedMBConv weights), the same two-step pattern can be
re-applied without script support — the script will simply print
the unmatched hash and exit non-zero, which *is* the bootstrap
signal.

### Verification (DoD)

```
$ source .venv-nas/bin/activate

$ python supernet/download_ofa.py
Downloading https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0
         -> /home/asil/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0
  31.0 / 31.0 MB (100.0%)
OK  /home/asil/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0  (31.0 MB, sha256=a7def36bb4e4…)

$ ls -l ~/.cache/ofa/
-rw-rw-r-- 1 asil asil 31011816 Apr 25 11:22 ofa_mbv3_d234_e346_k357_w1.0

$ sha256sum ~/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0
a7def36bb4e4c688c16d37eb60d5d34b2e6dcf6438c05bc86dea918fda04c6c7  ofa_mbv3_d234_e346_k357_w1.0

$ python -c "
import torch
from supernet.download_ofa import CHECKPOINT_PATH
ck = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
print('top-level keys:', list(ck.keys()))
print('num tensors:', len(ck['state_dict']))
print('first 3 keys:', list(ck['state_dict'].keys())[:3])
"
top-level keys: ['state_dict']
num tensors: 475
first 3 keys: ['first_conv.conv.weight', 'first_conv.bn.weight', 'first_conv.bn.bias']

$ python supernet/download_ofa.py   # idempotent re-run
OK  /home/asil/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0  (31.0 MB, sha256=a7def36bb4e4…)
```

**DoD satisfied:** The 31 MB checkpoint exists at
`~/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0` and its SHA256 matches
`PINNED_SHA256` in `supernet/download_ofa.py`. The script is
idempotent on re-run.

### Side note: ETag ≠ SHA256

The HEAD on the mirror returned `ETag:
"bb1d9115e01715e25e198e63e3eb8e455b573d96deb0cd87fb5a7d59bd6e73f7"`,
which is exactly 64 hex chars. I initially set `PINNED_SHA256` to
that value assuming it was a content hash; the first verification
correctly flagged a mismatch. The ETag for `raw.githubusercontent.com`
is a server-side opaque token — it correlates with content but is
*not* the SHA256. The lesson is recorded here so future-me doesn't
repeat it: **always compute the hash from the downloaded bytes**.

### What's next

CP 1.3 — `supernet/sampler.py`. Plan:

1. Add `gdown` and `Pillow` to `requirements-nas.txt` (transitive
   deps of `ofa.model_zoo` / `ofa.utils`). Re-run
   `scripts/setup_laptop_nas.sh` to refresh `.venv-nas/`.
2. Write `sampler.py` with:
   - `load_supernet() -> OFAMobileNetV3` (instantiates the
     architecture and loads `state_dict` from the cached checkpoint
     at the path exported by `download_ofa.py::CHECKPOINT_PATH`).
   - `sample(arch_dict) -> nn.Module` (calls the supernet's
     `set_active_subnet(...)` + `get_active_subnet(...)` and
     returns the materialised submodule).
3. DoD: forward `(1, 3, 224, 224)` through a randomly sampled subnet
   without error.

---

## CP 1.3 — Subnet sampler

**Date:** 2026-04-25
**Source spec:** `PROJECT_PLAN.md:83-87`
**DoD:** `sampler.sample(random_arch)` forwards a `(1, 3, 224, 224)`
tensor without error.

### What was done

1. Reconnoitred the installed `ofa` package (in `.venv-nas/`) to pin
   down the exact API surface for instantiation and elastic-subnet
   selection. Findings (file:line in the venv):
   - Class: `ofa.imagenet_classification.elastic_nn.networks.OFAMobileNetV3`
     (`ofa_mbv3.py:24`).
   - Canonical kwargs for the `_w1.0` checkpoint:
     `dropout_rate=0, width_mult=1.0, ks_list=[3,5,7],
      expand_ratio_list=[3,4,6], depth_list=[2,3,4]`
     (`model_zoo.py:72-79`).
   - `set_active_subnet(ks=, e=, d=)` (`ofa_mbv3.py:244-257`).
   - `get_active_subnet(preserve_weight=True)` deep-copies into a
     concrete `MobileNetV3` (`ofa_mbv3.py:325-355`).
   - `sample_active_subnet()` returns a `{"ks", "e", "d"}` dict and
     also sets the supernet's active state as a side effect
     (`ofa_mbv3.py:274-323`).
   - `OFAMobileNetV3.load_state_dict` overrides the default to remap
     legacy ProxylessNAS-era keys onto the elastic-NN module names
     (`ofa_mbv3.py:209-235`) — so `torch.load(...)["state_dict"]`
     can be passed straight in, as in `model_zoo.py:107`.
2. Verified the *transitive* import chain by trying the import in
   the CP 1.2 venv. It failed first on `from PIL import Image`
   (Pillow), then on `import torchvision.transforms.functional as F`
   (torchvision). Both come in via
   `ofa.utils.my_dataloader.my_random_resize_crop`. CP 1.2's "what's
   next" had guessed `gdown + Pillow`, but `gdown` is only imported
   by `ofa.model_zoo` — which we don't import.
3. Updated `requirements-nas.txt`:
   - Added `Pillow>=10.0` and `torchvision>=0.18`.
   - Did **not** add `gdown`.
   - Prepended a comment block warning that
     `pip install -r requirements-nas.txt` will pull torchvision from
     PyPI default (a cu13 build) and break at import; the canonical
     install path is `bash scripts/setup_laptop_nas.sh`, which passes
     `--extra-index-url https://download.pytorch.org/whl/cu128`.
4. Refreshed `.venv-nas/`. The first attempt — naïvely
   `pip install -r requirements-nas.txt` — pulled
   `torchvision==0.26.0` (PyPI default, built for CUDA 13) and crashed
   at import with `libcudart.so.13: not found` because torch is
   `2.11.0+cu128`. Fix: re-installed torchvision with
   `pip install --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchvision`,
   landing on `torchvision==0.26.0+cu128`. Documented this gotcha in
   `requirements-nas.txt`'s comment header and in
   `state/plan_state.yaml::notes`.
5. Wrote `supernet/sampler.py` (~70 lines). Public surface:
   - `OFA_MBV3_W10_KWARGS` — constructor kwargs constant.
   - `load_supernet(checkpoint_path=CHECKPOINT_PATH) -> OFAMobileNetV3`.
   - `sample(arch_dict, supernet=None) -> nn.Module`.
   - `random_arch(supernet=None) -> dict`.
   - Module-level `_supernet_cache` so callers that omit the
     `supernet` argument pay the 31 MB load **once**.
   - `__main__` block runs the CP 1.3 DoD smoke test.
6. Verified the DoD with `python -m supernet.sampler`. Output:
   `output shape: (1, 1000)`, `params: 4,636,232` (a random arch).
7. Ran extra sanity checks (max-arch vs. min-arch param counts,
   repeated random sampling, CP 1.2 hash check). All four pass.
8. Updated `supernet/README.md`: status header bumped to
   CP 1.3, sampler row marked ✅, added a usage example, added a
   "Why we bypass `ofa.model_zoo`" subsection, refreshed the deps
   table to include Pillow + torchvision, added the CP 1.3 DoD
   command to the local-install snippet.
9. Advanced `state/plan_state.yaml` to
   `current_checkpoint: "1.3"`, appended `"1.3"` to `completed:`,
   rewrote the `notes` block with the CP 1.3 status, the deps
   correction, and the venv-refresh gotcha.
10. Authored this entry.

### Why each piece

#### Why bypass `ofa.model_zoo` and re-implement the load?

Two structural reasons (both flagged at plan time, both confirmed
during implementation):

1. **Cache contract.** `ofa.model_zoo.ofa_net(pretrained=True)`
   redownloads via `download_url(url_base + net_id, model_dir=".torch/ofa_nets")`
   (`model_zoo.py:106`). The `model_dir` is **relative to the current
   working directory**, so any CWD change relocates the cache and a
   stale download leaks bytes into the repo working tree. CP 1.2's
   contract is `~/.cache/ofa/<net_id>` with a SHA pin — `model_zoo`
   silently breaks both halves of that contract.
2. **One fewer dep.** `ofa/model_zoo.py:3` does
   `import gdown` *unconditionally at module scope* — even if we'd
   only ever use the non-Google-Drive download path. Importing
   `ofa.model_zoo` therefore requires `gdown` to be installed, and
   `gdown` pulls `requests`, `tqdm`, etc. Re-implementing the 3-line
   load (`OFAMobileNetV3(**kwargs)` →
   `torch.load(...)["state_dict"]` → `load_state_dict`) eliminates
   that dep tree entirely.

The duplicated load logic is small and stable (the kwargs constant
lives next to it, so future-us only has to change one file if the
checkpoint's expected kwargs ever rotate).

#### Why Pillow + torchvision instead of gdown + Pillow?

CP 1.2's "what's next" note guessed at the deps based on
`ofa.model_zoo`'s top-level imports (`gdown` for Google Drive
downloads; `from PIL import Image` further down the chain). That
guess was anchored on the assumption we'd be importing
`ofa.model_zoo` — which CP 1.3 then chose **not** to do.

The actual transitive deps for
`from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3`:

| Import path | Module |
|---|---|
| `ofa.imagenet_classification.elastic_nn.networks.__init__` | → `.ofa_proxyless` |
| → `.ofa_proxyless` | → `ofa.utils` |
| → `ofa.utils.__init__` | → `.my_dataloader` |
| → `ofa.utils.my_dataloader.__init__` | → `.my_random_resize_crop` |
| → `.my_random_resize_crop` | `from PIL import Image` (Pillow) **and** `import torchvision.transforms.functional as F` (torchvision) |

Verified empirically: after installing Pillow alone, the import
fails at `import torchvision`; after installing both, it succeeds.

`gdown` is only ever imported by `ofa.model_zoo` (verified with
`grep -rln "import gdown" .venv-nas/.../ofa/`, exactly one hit:
`model_zoo.py`).

#### Why a module-level `_supernet_cache`?

The Phase 3 BO loop (CP 3.2 / 3.3) is going to call
`sample(arch_dict)` thousands of times. Each call's expensive part
is `set_active_subnet` (cheap — sets a few ints) followed by
`get_active_subnet(preserve_weight=True)` (deep-copies the active
subnet, ~10s of ms). The 31 MB `state_dict` load and the supernet
instantiation are one-time costs that should not recur per call.

The cache is module-level (a private `_supernet_cache`) rather than
a class attribute because there is exactly one supernet per CP 1.3 —
the `_w1.0` MBv3 — and a class-based wrapper would add ceremony for
no benefit. If Phase 5 ever loads a *second* supernet alongside the
first, this becomes a `dict[net_id -> OFAMobileNetV3]` and the
`load_supernet` signature gains a `net_id` argument. Until then,
YAGNI.

The cache is opt-out: callers who pass an explicit `supernet=`
argument bypass it. That matters because (a) tests want fresh
state, and (b) `random_arch()` mutates the supernet's active
subnet as a side effect — a caller that wants deterministic state
across `sample()` calls can manage their own supernet instance.

#### Why `weights_only=False` on `torch.load`?

The CP 1.2 verification used `weights_only=False`
(`procedure.md:354`). Matching that choice means the load path
behaves identically to what CP 1.2 verified by hand. The cached
file is a `torch.save` dict whose only top-level key is
`state_dict` (a dict of tensors), so `weights_only=True` should
also work — but the modest gain (avoid future-warning + tighter
unpickling) doesn't justify diverging from CP 1.2's verified path.

If a future torch version flips the `weights_only` default to
`True` and our checkpoint stops loading, we revisit. Until then,
the explicit `False` is a small annotation that says "CP 1.2's
load contract is what we're matching."

#### Why no `gdown` but yes `Pillow`?

Pillow is unavoidable on the elastic-NN import path (above). gdown
is avoidable because we don't import `ofa.model_zoo`. Avoiding deps
is cheap and removes a future failure mode (a `gdown` upgrade that
breaks against newer Python will not affect CP 1.3+).

#### Why a comment block on `requirements-nas.txt`?

The venv-refresh gotcha (a plain `pip install -r` lands on PyPI's
cu13 torchvision and breaks at import) is the kind of thing
future-me will rediscover painfully if it isn't documented at the
point of failure. The header comment now says explicitly: "Install
via `setup_laptop_nas.sh`." That keeps future-me from running the
seemingly-equivalent `pip install -r` command.

A more durable fix would be pinning
`torchvision==0.26.0+cu128` literally in the requirements file —
but that hardcodes the CUDA tag, which conflicts with
`setup_laptop_nas.sh`'s `TORCH_CUDA_INDEX` override. The comment
buys most of the safety with none of the rigidity.

### Verification (DoD)

Verbatim `python -m supernet.sampler` output (random seed not
fixed; arch will differ across runs):

```
arch: {'ks': [7, 7, 7, 7, 5, 5, 5, 7, 5, 3, 7, 3, 5, 7, 7, 7, 3, 3, 3, 3], 'e': [3, 6, 6, 3, 4, 4, 3, 6, 6, 6, 6, 3, 4, 3, 4, 3, 4, 3, 6, 3], 'd': [2, 2, 2, 3, 3]}
output shape: (1, 1000)
params: 4,636,232
```

Extra sanity (max-arch vs. min-arch, then a third random arch):

```
max-arch params: 7,664,760
min-arch params: 3,410,792
ratio max/min:   2.25x
max output shape: (1, 1000)
min output shape: (1, 1000)
random arch param count: 5,112,704
random arch output shape: (1, 1000)
OK: max > min, repeated sampling stable, output shape (1, 1000) every time
```

Idempotency of the underlying CP 1.2 cache:

```
$ python supernet/download_ofa.py
OK  /home/asil/.cache/ofa/ofa_mbv3_d234_e346_k357_w1.0  (31.0 MB, sha256=a7def36bb4e4…)
```

Interpretation:

- **DoD satisfied.** The `(1, 1000)` output shape is ImageNet's
  1000-class logits — the network ran end-to-end without error.
- **`set_active_subnet` actually mutates structure.** Param count
  ranges from 3.4 M (min: `k=3, e=3, d=2` everywhere) to 7.7 M (max:
  `k=7, e=6, d=4` everywhere) — a 2.25× spread. If the call were a
  no-op, both would land on the same value (~6.0 M, the default
  active subnet).
- **Cached checkpoint untouched.** The CP 1.2 hash still matches
  the pin, so the sampler's load path is read-only.

### Decisions taken (no AskUserQuestion in this CP)

| Question | Choice | Reason |
|---|---|---|
| `ofa.model_zoo` or direct? | Direct (re-implement 3-line load). | Cache contract + one fewer dep (gdown). |
| What deps to add? | Pillow + torchvision. | Empirically verified import chain. CP 1.2's "gdown + Pillow" guess was wrong (gdown only used by model_zoo). |
| Cache the supernet? | Module-level lazy cache. | Phase 3 BO loop will call `sample()` thousands of times; the 31 MB load must amortise. |
| `weights_only=`? | `False`, matching CP 1.2. | No reason to diverge from CP 1.2's verified load path yet. |

### What's next

CP 1.4 — ImageNet sanity: confirm a sampled subnet is within 1.5 %
top-1 of OFA's published number for that arch on a 2k-image
ImageNet-val subset.

Two preconditions to settle before CP 1.4 starts:

1. **D1 — target dataset.** The full Phase 2.4 fine-tune harness
   needs a target task (ImageNet vs. Cityscapes/ADE20K vs. COCO).
   CP 1.4's smoke test, however, can run against ImageNet-val
   independently of D1, because OFA's published numbers are
   ImageNet-val top-1.
2. **ImageNet-val 2k subset.** Need to assemble (or download) a 2k
   subset with labels and a known iteration order. Likely path:
   download ILSVRC2012 val, hash-pin a deterministic 2k subset
   (e.g. first 2k filenames after sorted ordering).

---

## CP 2.1 — Arch → block list translator

**Date:** 2026-06-06
**Source spec:** `PROJECT_PLAN.md` CP 2.1 (Phase 2)
**DoD:** 10 random archs — every emitted `(block, cfg, input_shape)`
tuple has a matching `row_key` in `data/lut.jsonl`.

### Ordering note: why 2.1 before 1.4

CP 1.4 (ImageNet sanity) is the plan's literal next step but is gated
on assembling an ImageNet-val 2k subset (a download + label plumbing)
and is only meaningful once a target dataset (D1) is in view. CP 2.1
needs nothing external — only the already-built sampler, catalog, and
LUT — so it was taken first. CP 1.4 remains open.

### What was done

1. **Discovered a grid/search-space mismatch.** A faithful translator
   must emit OFA-MBv3's *real* block configs, but the LUT's MBConv
   grid (`catalog/blocks.py`) used generic widths `{16,32,64,96,160}`
   at resolutions `{56,28,14,7}`. OFA-MBv3-w1.0 actually uses widths
   `{16,24,40,80,112,160}` at resolutions `{112,56,28,14,7}`, plus an
   `expand=1` first block. So **zero** emitted tuples would have
   matched the LUT as it stood — the DoD was unreachable without a
   grid change.
2. Confirmed `data/lut.jsonl` was **dummy/roofline** data (all 2619
   rows shared a single timestamp `2026-04-25T08:13:53Z`, the
   signature of `gen_dummy_lut.py`, not per-row `run_sweep.py`
   measurements), so regenerating it costs nothing real.
3. **Created `catalog/ofa_mbv3.py`** — the single source of truth for
   the OFA-MBv3-w1.0 macro-topology (stage widths/strides/SE/res, the
   fixed first block, the `KS/E/D` choice sets). Exposes
   `reachable_mbconv_configs()`, which enumerates the **91** unique
   MBConv configs the search space can produce (1 fixed first block +
   5 stages × {entry, repeat} × |KS|×|E|).
4. **Augmented the catalog grid** (`catalog/blocks.py`): unioned those
   91 configs into `_MBCONV_GRID` after the existing generic grid
   (de-duplicated). The generic rows are untouched; mbconv rows went
   2016 → 2107. (Per the user's "augment, don't replace" choice — the
   directed union adds only the 91 reachable configs rather than
   exploding the cartesian axes, which would have added ~11k rows
   unreachable by OFA.)
5. **Wrote `search/arch_to_blocks.py`** — the CP 2.1 deliverable:
   - `arch_to_blocks(arch_dict) -> list[("mbconv", cfg, input_shape)]`
     walks the shared topology, propagating channels/stride/resolution
     (entry block at the stage stride and `res_in`; repeat blocks at
     stride 1 and `res_in // stride`). Reuses `input_shape_for` so the
     `input_shape` matches the LUT's exactly.
   - `arch_to_keys()` maps each block through `catalog.sweep.row_key`.
   - `__main__` runs the DoD smoke test, reusing
     `lut.orchestrate.resume.completed_keys` to read the LUT's keys.
6. **Regenerated the dummy LUT** with
   `python -m lut.orchestrate.gen_dummy_lut --overwrite` (CPU roofline,
   no Jetson/CUDA) — 2710 rows, mbconv 2107.

### Why each piece

#### Why a shared `catalog/ofa_mbv3.py` instead of topology in `search/`?

Two consumers need the same fixed table: the **LUT grid** (catalog
layer) must enumerate the reachable configs so every searchable block
has a row, and the **translator** (search layer) must order them per
arch. Putting the table in `catalog/` keeps the dependency direction
clean (`search → catalog`, never the reverse) and guarantees the grid
and the translator can never drift — they read the same constants.
The translator owns only the *ordering* logic (arch_dict → sequence),
which is genuinely search-space knowledge.

#### Why emit only the MBConv backbone (not stem/head)?

The stem (`3→16` s2), `final_expand` (`160→960`), `feature_mix`
(`960→1280`), and the classifier are **identical for every arch** —
search never varies them. They contribute a constant latency offset,
not a per-arch lookup, so CP 2.2's cost function adds them once rather
than the translator emitting them per arch. This also keeps every
emitted tuple `block="mbconv"`, so the DoD is a clean LUT-coverage
check over one block type.

#### Why the `expand=1` first block is fine despite a structural quirk

OFA skips the inverted-bottleneck 1×1 when `expand=1`; the catalog's
`MBConv` builds a (redundant) 1×1 instead. For CP 2.1 this is
irrelevant — the DoD checks `row_key` membership, not module
structure — and the dummy LUT's roofline latency for that single
fixed block doesn't change search *ranking*. If real Jetson
measurement later cares, the first block is one row to special-case.

### Verification (DoD)

```
$ python -c "from catalog.ofa_mbv3 import reachable_mbconv_configs as f; print(len(f()))"
91

$ python -m lut.orchestrate.gen_dummy_lut --overwrite
Done. Wrote 2710 rows.   # mbconv 2107

$ python -m search.arch_to_blocks
LUT: 2710 row_keys in .../data/lut.jsonl
  arch 0: d=[3, 4, 3, 2, 4] -> 17 blocks  [OK]
  ...
  arch 7: d=[4, 4, 4, 4, 3] -> 20 blocks  [OK]
  arch 9: d=[4, 4, 2, 2, 4] -> 17 blocks  [OK]
DoD PASS
```

Block counts span 15 (all stages min depth: 1 + 5×2 + ... ) to 20
(near-max depth), matching `1 + Σ d[s]`.

**Extra check — real sampler integration.** Translated an arch from
the actual `supernet.sampler.random_arch(load_supernet())` (not just
synthetic dicts): `len(ks)=20, len(e)=20, len(d)=5`, 15 blocks, **0**
missing from the LUT. Confirms the translator consumes real OFA
sampler output, not just the smoke test's format.

**DoD satisfied:** every emitted tuple of 10 random archs (and one
real sampled arch) matches a LUT `row_key`.

### Decisions taken (via AskUserQuestion)

| Question | Choice | Reason |
|---|---|---|
| LUT doesn't cover OFA blocks — realign, augment, or defer? | **Augment** (keep generic grid + add OFA configs) | Preserves the generic blocks for any future non-OFA use; the directed union keeps the bloat to +91 rows. |
| CUDA missing (from prior session) | Note, defer | CP 2.1 needs no CUDA; resolve before CP 2.4. |

### What's next

CP 2.2 — LUT composite-cost function (`search/cost.py`):
`cost(arch) → {latency_ms, peak_mem_mib, params, flops}` as the sum of
`LUT[row_key]` over `arch_to_blocks(arch)`, plus the constant
stem/head offset. The measured-vs-summed additivity validation (DoD:
within 15% on 5 real subnets) needs a Jetson and is the one part of
CP 2.2 that can't run on this machine yet — the summing + cost API can
be built and unit-tested against the dummy LUT now.

---

## Plan amendment — Phase 8 Knowledge Distillation added (2026-06-10)

**Type:** Scope / roadmap change (not a checkpoint). No code shipped; no
checkpoint advance in `state/plan_state.yaml` (still at CP 2.1).
**Source request:** "add a final step at the end, a distillation process;
scan the whole project to modify the `.md`s and `state/` and anything needed."

### What changed

A new **Phase 8 — Knowledge Distillation** was inserted into the plan, and the
former **Phase 8 — Deployment Packaging** was renumbered to **Phase 9**. The
pipeline now reads: search → winner α* → **distill (teacher → student)** → TRT
export.

Phase 8 (CP 8.1–8.4): select & pin an external SOTA teacher (CP 8.1); implement
the KD loss + training harness, reusing `eval/`'s data pipeline (CP 8.2); run
the full-schedule distillation on the search winner, beating CP 7.3's plain
long-train baseline at the same latency (CP 8.3); serialize the distilled winner
to `state/winner_distilled/` as Phase 9's input (CP 8.4).

### Why

Every accuracy number the search produces is a 5-epoch **proxy** used only to
*rank* candidates (CP 2.4 / 3.2 / 7.2) — α* is never trained to convergence
during search. A dedicated final phase is the natural home for the project's one
full-schedule training run, and KD against a strong teacher is the standard,
highest-accuracy-per-epoch way to do it (OFA, BigNAS, AttentiveNAS all distill
the final model). KD is **latency-invariant** — it changes weights, not the
graph — so the entire LUT contract and Phase 9's ≤ 15 % export bar are
untouched; only accuracy moves. Placing it *before* deployment (rather than
literally last) is ML-correct: you export the distilled weights.

### Decisions taken (via AskUserQuestion)

| Question | Choice | Why |
|---|---|---|
| Placement of the distillation step | **New Phase 8; Deployment → Phase 9** | Distillation produces the model you deploy, so it must precede the TRT export. |
| Distillation teacher | **External SOTA pretrained model** | Higher accuracy ceiling than self-distillation; the concrete model is chosen at CP 8.1 to match the D1 dataset/task (no new open decision — the approach is pinned). |

### Files edited

- `PROJECT_PLAN.md` — new Phase 8 section (CP 8.1–8.4, refs, risks, latency
  note); pipeline diagram; Phase 8 → 9 deployment renumber (CP 9.1–9.3,
  distilled-winner input, `model_card` records teacher + KD hyperparams);
  timeline table (+Phase 8 row, total 18–28 → 20–31 sessions); D1 extended to
  note it also selects the teacher.
- `PROJECT.md` — one-line summary clause; new "Final stage — Knowledge
  distillation" subsection; Milestone M6; Hinton KD reference; `eval/`
  repository-status line clarified (its long-train is the *baseline*; KD is the
  final train).
- `README.md` — status table (KD = Phase 8, Deployment = Phase 9); module map
  (+`distill/`, `eval/` tightened); "all 8 phases" → "all 9 phases".
- `CLAUDE.md` — project paragraph (KD final stage); module-structure tree
  (+`distill/`); "Phases 2–8" → "Phases 2–9".
- `state/plan_state.yaml` — forward-looking note in `notes:` (no
  checkpoint-state change).
- `distill/` — new module stub (`__init__.py` + `README.md`) so the module map
  has an honest target; the teacher pin is TBD until D1 resolves.

### What's next (unchanged)

CP 2.2 — `search/cost.py` (LUT composite-cost). The distillation phase is future
work gated on the CUDA blocker (same as CP 2.4+); nothing in Phase 8 is
actionable until D1 is resolved and a GPU is available.

---

## Hardening pass — code quality & architecture (2026-06-11)

**Type:** Quality/infrastructure pass (not a checkpoint). No roadmap advance in
`state/plan_state.yaml` (still at CP 2.1).
**Source request:** "analyze the codebase and implement code quality and
architectural improvements" — robustness/scalability for the research-community
codebase.

### Decisions taken (via AskUserQuestion)

| Question | Choice | Why |
|---|---|---|
| Committed Jetson credentials (public repo) | **Untrack + rewrite history + force-push** | `git filter-repo --invert-paths` scrubbed `jetson credentials username.txt` and the stray `curl` from all 3 commits; mirror backup at `../TFM_NAS_backup_2026-06-11.git`. **Rotate the Jetson password** — it was public and GitHub may cache pre-rewrite objects. |
| `config.yaml` `precision: fp32` vs FP16-only docs | **Keep fp32; document + precision-aware resume** | Resume now filters by precision (`completed_keys(path, precision=...)`), so the fp16 dummy LUT can no longer mask a real fp32 sweep; the caveat (precision is NOT in `row_key`) is documented in `lut/docs/schema.md`. |
| `lut/loader.py` in scope? | **Yes (CP 2.2 groundwork)** | `load_lut(path, precision)` filters before keying and raises on duplicate keys — the validated input surface `search/cost.py` will consume. |
| GitHub Actions CI? | **Yes** | ruff + mypy + `pytest -m "not slow"` on the CPU venv; ofa/LUT-file tests skip by design. |

### What was done (chronological, one commit per phase)

1. **Git hygiene + history rewrite.** Both stray files scrubbed from history
   (first commit hash `793bb7b` unchanged; `34fddcd`→`5999341`, `a4c19a7`→`eac1715`),
   force-pushed. `.gitignore` gained `*credential*`/`*secret*`/`.env` and tool
   caches; duplicate `*.swo` removed. Credentials note survives untracked on disk.
2. **Dev tooling.** `pyproject.toml` (tool config only — deliberately no
   `[project]`: runtime stays `python -m` from repo root), root `conftest.py`,
   `requirements-dev.txt` (pytest/ruff/mypy) wired into both setup scripts,
   `scripts/check.sh` (uses `python -m`, unsets `PYTHONPATH` — ROS's setup.bash
   was crashing pytest via auto-loaded `launch` plugins).
3. **Safety-net tests (before any refactor).** `tests/` froze: 5 golden
   `row_key` hashes + the bool-vs-int JSON tripwire (`se=True` vs `se=0` hash
   differently — load-bearing!), catalog counts (2710/2107/91) + schema
   uniformity + CP 2.1 reachable⊆grid invariant, arch_to_blocks structure
   (chaining/strides/resolutions/depth truncation) + in-memory key coverage,
   hand-computed FLOPs goldens, slow end-to-end `gen_dummy_lut` regeneration
   identity, `resume.py` corruption semantics, sampler smoke (skips sans ofa).
4. **Refactor (contract-frozen).** `catalog/flops.py` extracted the FLOPs hook
   counter that lived verbatim in BOTH `run_sweep.py` and `gen_dummy_lut.py`;
   `catalog/contracts.py` added TypedDicts (`MBConvCfg`, `ArchDict`, `LutRow`,
   `LatencyStats`, `Block`) — TypedDict not dataclass so the runtime wire format
   (and hence hashes) cannot drift. Dead code removed (duplicate docker `cmd`
   in `run_remote_bench`, unused `pending`); deprecated `utcnow()` replaced.
5. **Robustness.** `_parse_bench_stdout` (empty/garbage container output →
   diagnosable ValueError); per-row failure accounting + end-of-run summary +
   exit 1; `load_config` aggregate validation naming every missing key;
   `validate_arch_dict` at the search boundary (lengths, membership, exact-int
   types — rejects `bool`/`np.int64` that would corrupt `row_key` JSON);
   `build_block` unknown-name ValueError listing known blocks.
6. **Loader + precision-aware resume.** `lut/loader.py` (`iter_lut_rows` owns
   tolerant line-parsing + malformed-count warning; `load_lut` filters
   precision before keying, raises on collisions). `completed_keys` gained
   `precision=None` (legacy default — DoD smoke test untouched); `run_sweep`
   passes its configured precision. Dummy rows now carry
   `"source": "roofline_dummy"`; dummy LUT regenerated (keys identical).
7. **Lint/type.** ruff (E,F,W,I,B,UP @ line-length 100) and mypy clean across
   36 files; `lut/bench/` + `nas-course/` excluded (Jetson-side files are
   deployed separately and untestable locally — left untouched on purpose).
8. **Docs.** `lut/docs/schema.md`: precision/`source` caveats, `res` added to
   the cfg example, stale `python -m orchestrate.probe_device` path fixed;
   CLAUDE.md: hardening state + "Tests & tooling" conventions;
   `requirements-nas.txt`: stale `ofa_extractor/` reference fixed.

### Erratum (CP 2.1 entry)

CP 2.1's narrative (and a comment in `catalog/ofa_mbv3.py`) claimed the catalog
MBConv represents the first block's `expand=1` "as a (redundant) 1x1". Wrong:
`catalog/mbconv.py` skips the expansion conv when `expand == 1` (`if expand !=
1`), exactly like OFA — the structures match. No behavioral impact (the CP 2.1
DoD checked key membership only); comment fixed, CP 2.1's entry left as
written per journal discipline.

### Verification (final)

```
bash scripts/check.sh            # ruff clean, mypy clean (36 files), 102 passed / 1 skipped
python -m search.arch_to_blocks  # DoD PASS (10 archs, 0 missing keys)
git log --all -- "jetson credentials username.txt"   # empty (scrubbed)
```

### Pending (user contribution)

`load_device_info` in `lut/orchestrate/run_sweep.py` carries a TODO(user): the
fail-fast policy for missing/corrupt `device_info.json` (rows must never
silently get `power_mode: None`). Scaffolded with acceptance criteria;
recommended shape is fail-fast + `--allow-missing-device-info` escape hatch.

### What's next (unchanged)

CP 2.2 — `search/cost.py`, now consuming `lut.loader.load_lut` and covered by
the existing test scaffolding. The CUDA blocker (CP 2.4+) is unchanged.

---

## Measurement audit — LUT collection methodology (2026-06-12)

Not a checkpoint. The first REAL Jetson rows landed today (3 conv3x3 rows,
fp32, TRT 10.3.0, container `l4t-tensorrt:r10.3.0-devel`, ~7 s/row); this
entry records the audit of the collection path and the hardening that
followed. Trigger: user request to verify "the way data is collected /
measured is correct" before committing to the full 2710-row sweep.

### Verdict

The measurement core was confirmed sound:

- **CUDA-event timing, per-iteration, queue depth 1** (`lut/bench/run_bench.py`)
  — correct semantic for blocks that execute sequentially in a net. Evidence
  it is live: every p50 in the measured rows is an exact multiple of 32 ns
  (Orin's 31.25 MHz globaltimer tick).
- 50 warmup + 200 timed iters; sorted samples; p50/p95/mean/std/n persisted;
  H2D/D2H excluded (input uploaded once before the loop).
- Engines built on-device by trtexec; `trt_version`/`power_mode`/`jetpack`
  stamped per row; `nvpmodel` before `jetson_clocks` in setup (correct order);
  `--store`/`--restore` pairing with the new teardown script.
- The user's `peak_mem_mib` rework (TRT `device_memory_size_v2` + IO bytes,
  replacing the `cuda.mem_get_info()` free-delta) is correct: the free-delta
  produced 0.0 / 86.7 MiB garbage on unified memory (preserved in
  `data/lut.jsonl.stale3.bak`); the TRT-reported scratch is deterministic.
- Physical sanity: 339 GFLOPS achieved on conv3x3@res112 (~26% of fp32 peak
  at 612 MHz); probed DRAM BW 62.8 GB/s vs 68 theoretical.

### Decisions (user, via AskUserQuestion)

| Decision | Choice | Rationale |
|---|---|---|
| fp32 rows are TF32-allowed (TRT default on Ampere; no `--noTF32`) | **Keep TF32, document** | LUT should predict what a default TRT deployment does; documented in `lut/docs/schema.md` + `config.yaml` |
| Jetson public endpoint (host/port/user) was about to be committed in `config.yaml` | **`config.local.yaml` overlay** | Real endpoint now lives in gitignored `config.local.yaml`, merged over the committed placeholder template by `load_config` and both jetson scripts |

### Gaps found and fixed

1. **`tests/test_lut_keydrift.py` failed** once `lut.jsonl` became a partial
   real LUT (it asserted catalog ⊆ file, valid only for the complete dummy
   artifact). Split into: hard orphan check (file ⊆ catalog, always) +
   completeness gate that skips with a coverage count until collection
   finishes.
2. **No device-state verification at sweep time** (the structural risk):
   `jetson_clocks` does not survive reboots and `device_info.json` could be
   stale, so a post-reboot sweep would silently measure with DVFS active.
   `run_sweep` now re-probes at start (shared `probe_device.probe()`),
   rewrites `data/device_info.json`, and aborts on unlocked clocks or
   power-mode mismatch (`preflight_verdict`; `--skip-preflight` bypasses).
   `device_probe.sh` now reports `gpu_clock_mhz_cur` + `clocks_locked`
   (devfreq `min_freq == max_freq`). Rows stamp `clocks_locked` and
   `source: "jetson_trt"`.
3. **~9% inter-run p50 drift on ~40 µs blocks** (evidence: 15:49 vs 15:57
   runs of the same 3 rows). `run_bench.py` now samples until BOTH
   `n >= timed_iters` AND the timed window spans `min_window_s` wall time
   (default 0.5 s; trtexec uses 3 s duration-based sampling for the same
   reason). `latency_ms.n` records the actual count.
4. **No TRT timing cache** across 2710 builds: added persistent
   `--timingCacheFile` at `{remote_workdir}/cache/trt_timing.cache` — faster
   builds and identical layers resolve to identical tactics across rows.
5. **`peak_mem_mib` semantics were stale in `schema.md`** (still described
   the abandoned free-delta). Rewritten: TRT scratch + IO buffers, excludes
   weights (reconstruct via `params`); explicitly documented as
   NON-additive across blocks (inter-block tensors double-count) — the
   whole-net memory model is a CP 2.2+ decision.
6. Small script fixes: clock sync now uses UTC on both ends (`date -u`,
   `sudo date -u -s` — a TZ mismatch skewed the Jetson clock otherwise);
   teardown restores clocks BEFORE switching power mode and reads
   `jetson.idle_power_mode` (default 1) instead of hardcoding; both scripts'
   minimal YAML reader now strips inline `#` comments (the old reader broke
   on commented values — the reason config.yaml's comments had been deleted).

### Contracts kept

- `row_key` untouched; golden hashes untouched; all new row fields
  (`source`, `clocks_locked`) are additive payload. The 3 measured rows
  remain valid (they predate the new fields; schema documents that).
- `JetsonConfig` gained `power_mode`/`lock_clocks` — the old "awk-only"
  rationale lapsed because the Python preflight now consumes them too.

### Pending (user contribution)

`preflight_verdict` in `lut/orchestrate/run_sweep.py` ships the recommended
fail-fast policy (abort on unlocked clocks / power-mode mismatch, warn on a
failed bandwidth probe) but the policy is TODO(user)-owned: open calls are
whether bandwidth-0 should abort, and whether a TRT/JetPack version change
vs. the previous device_info.json should abort to keep one LUT per software
stack. The older `load_device_info` TODO(user) (now only the
`--skip-preflight` path) still stands.

### What's next

Run the full sweep: `bash scripts/setup_jetson.sh`, then
`python -m lut.orchestrate.run_sweep` (resumable; ~6 h optimistic at the
observed 7 s/row — mbconv engine builds will dominate; the timing cache
amortizes them), then `bash scripts/teardown_jetson.sh`. CP 2.2 unchanged.

**Addendum (same day):** `search/arch_to_blocks.py`'s `_dod_smoke_test` had the
same partial-file assumption as the keydrift test — it resolved emitted keys
against `data/lut.jsonl`. Retargeted at the catalog key set (`iter_sweep`),
which is what the dummy artifact materialized anyway; the translation DoD is
unchanged and `python -m search.arch_to_blocks` prints `DoD PASS` again.

---

## CP 2.2 — LUT composite-cost function (2026-06-13)

`search/cost.py`: `cost(arch_dict, lut) -> {latency_ms, peak_mem_mib, params,
flops}`. Composes a sampled subnet's predicted cost from the per-block LUT rows
(`arch_to_blocks` → `row_key` → `lut[key]`), so Phase-3 search can rank
thousands of candidates without touching the Jetson. Built TDD (tests RED before
code); `tests/test_cost.py` is the contract.

### Decisions taken (via AskUserQuestion)

- **Stem/head offset → parameterized, default no-op.** `arch_to_blocks` emits
  only the searchable MBConv backbone; the fixed stem (3→16) + head
  (final-expand 160→960, feature-mix 960→1280) convs are a constant offset.
  They are **not** in the catalog grid (conv3x3 starts at in_c=16/res≤112;
  conv1x1 caps out_c at 320; no linear/GEMM block) and were **deliberately not
  added** (the rejected option: grid edit + dummy regen + 3 sweep rows). Instead
  `cost(..., stem_head: CostOffset | None = None)` takes an optional offset,
  default `ZERO_OFFSET`. Rationale: the offset is identical for every arch in
  the OFA-MBv3-w1.0 space (last stage always → 160 ch, input fixed at 224), so
  it **never changes ranking** — only absolute cost, which only the deferred
  additivity DoD needs. The measured numbers slot in later without touching the
  interface.

### Whole-net memory model (resolves the CP 2.2+ decision flagged in the
Measurement audit §5)

Aggregation is **heterogeneous**: `latency_ms` / `params` / `flops` are
**summed** (sequential runtime, resident weights, total compute are additive);
`peak_mem_mib` is the **max** over blocks (and over the offset), never the sum.
The LUT's `peak_mem_mib` is per-block scratch+IO measured in isolation; blocks
execute one at a time and free that scratch, so the resident working set is the
largest single block's, not the running total — summing would overestimate
~20×. The offset folds into the `max` (not added): the stem/head run
sequentially too, and folding-in also keeps the empty-`rows` case well-defined
(`max([])` would raise). This is the documented additivity assumption; its error
bound is the DoD below.

### Files

- `search/cost.py` (new): `cost`, `cost_from_path` (load-then-cost convenience),
  `_aggregate` (the reduce), `CostError` (loud on a missing key — a partial real
  LUT legitimately misses keys; silently undercounting would corrupt the search
  ranking), `ZERO_OFFSET`, and a `__main__` smoke demo.
- `catalog/contracts.py`: added `CostDict` + `CostOffset` TypedDicts.
- `tests/test_cost.py` (new, 9 tests): `_aggregate` reduce on inline synthetic
  rows (sum vs **max**, offset fold-in, empty case), `cost()` missing-key →
  `CostError`, precision filtering via `cost_from_path`, and an on-disk
  round-trip that **skips while the real LUT is partial** (like keydrift).

### DoD status

- **Code + unit tests: DONE** — 8 pass, 1 skips (round-trip, gated on a complete
  LUT). `bash scripts/check.sh` green (ruff + mypy + 122 pytest).
- **Additivity gate DEFERRED:** "measured vs. summed latency for 5 random full
  subnets within 15 %" (PROJECT_PLAN.md:130) needs the full real fp32 LUT + a
  full-subnet Jetson measurement. Run after the sweep; if it fails, CP 2.3
  (residual-GP correction) opens. `state/plan_state.yaml` keeps `last_completed:
  2.1` until this passes.

### Contracts kept

`row_key` untouched; golden hashes untouched; no catalog grid change (the dummy
LUT and all 3 measured rows stay valid). `CostDict`/`CostOffset` are new,
additive type contracts.

### What's next

Either run the full sweep (completes Phase 0 + unblocks the additivity DoD) or
proceed to CP 3.1 (search-loop scaffold) consuming `cost()`. The CUDA blocker
(CP 2.4+ fine-tuning) is unchanged.

---

## Review-response pass — peer-review findings on the done work (2026-06-14)

Not a checkpoint (stays CP 2.2 / `last_completed: 2.1`). `peer_review_simulation.md`
(a simulated 5-reviewer panel, 2026-06-13) critiques the **thesis**; its LaTeX
source is **not in this repo** (no `*.tex`/`*.bib` — only code + planning md). So
this pass actions only the findings that land on **code/design already built here**
(Phase 0 LUT, CP 2.1, CP 2.2) plus the P0 "close-before-compute" plan gates. The
thesis-prose findings (soften the 32 ns claim to *timer provenance, not value
correctness*; §2.8 contribution-delta table; §5.10 limitations; named external
baselines; MCUNet/HAT) are **out of scope** here — they belong to the writeup tree.
Scope was confirmed with the user via AskUserQuestion.

### The sharpest finding: cost.py contradicted our own schema (Reviewer 4.4)

`lut/docs/schema.md:53-62` says the whole-net memory estimate is
`sum(weights) + max_i(scratch_i + io_i)`, "to be decided at cost-model time
(CP 2.2+)". But CP 2.2's `cost()` shipped `peak_mem_mib = max_i(scratch_i+io_i)` and
reported `params` *separately* — the **resident weights were never in the memory
figure** (~10–16 MiB at fp16 for a 5–8 M-param subnet, i.e. comparable to or larger
than the scratch term). A consumer reading `cost(arch)["peak_mem_mib"]` as "memory
this net needs" would undercount badly, and Phase-3's `μ·max(0, m−budget)²`
constraint would be wrong.

Resolution (A1): keep `CostDict.peak_mem_mib` meaning **exactly** the measured
working set (same as `LutRow.peak_mem_mib`), and add
`search.cost.resident_mem_mib(cost, bytes_per_param)` =
`params·bytes/2²⁰ + peak_mem_mib`. `bytes_per_param` is explicit (fp16→2, fp32→4)
so no precision assumption hides inside the reduce. Documented the exclusion on
`CostDict.peak_mem_mib`. (Two fields named `peak_mem_mib` meaning different things
would have been the trap; the weights term lives in a named helper instead.)

### Code/doc fixes (Part A — laptop-only, TDD)

- **A1 — memory model.** `resident_mem_mib` helper (above) + `CostDict` doc.
  Tests: `test_resident_mem_adds_weights_to_peak_working_set`,
  `test_resident_mem_scales_with_precision_bytes`. Measured-vs-composed `m`
  validation stays **deferred** with the latency additivity check (needs Jetson).
- **A2 — additivity DoD designed to expose fusion, not hide it (R4.2).** New
  `search/validate_additivity.py`: reports `(summed−measured)/measured` **binned by
  depth** + aggregate, and flags any bin breaching the bar (the CP 2.3 trigger).
  TensorRT fuses across block seams isolated rows can't see, so the summed LUT
  over-predicts and the error grows with depth; a single aggregate would average
  that away. Tests (synthetic — real inputs need the sweep + on-device runs):
  `test_depth_growing_residual_trips_bin_but_not_aggregate` is the headline (mean
  < 15 % yet the deepest bin breaches).
- **A3 — precision is a validity boundary, not just a filter (R4.3).** `cost.py`
  docstring limitation: block latency *rankings* are not precision-invariant, so a
  result searched on the fp32/TF32 LUT is faithful **at the searched precision**
  only — re-targeting FP16/INT8 is a re-sweep *and* a re-search.

### Plan gates (Part B — PROJECT_PLAN.md, pre-compute P0)

- **B1 — proxy-rank-fidelity DoD beside CP 2.4 (R2.1 / P0.2, the panel's #1
  consensus gap).** CP 2.4's "twice within 0.5 %" is *reproducibility*, not *rank
  correctness*. Added a DoD: fully train ≈8–12 archs, require **Kendall-τ ≥ 0.7**
  of the 5-epoch proxy vs full-train ranking, gate the search on it. CUDA/D1-dep.
- **B2 — statistical protocol for Phase 3 (R2.2 / P0.3).** ≥5 seeds for **both** BO
  and the random-search control; Pareto **hypervolume** + across-seed dispersion +
  dominance-across-seeds (replaces the single-run anecdote). GP seeded with the
  random evals; batch-EI needs explicit diversification; D2 budget justified, not
  assumed. CP 3.3 DoD rewritten accordingly.
- **B3 — CP 2.2 DoD is now depth-binned; CP 2.3 trigger pre-registered.** Pass =
  *no depth bin* exceeds 15 %; CP 2.3 fires on any bin breach **or** an
  upward-with-depth residual — not on an aggregate miss.

### Deferred / out of scope / needs the user

- **Deferred (needs Jetson):** measured-vs-summed additivity run (now via
  `validate_additivity`), measured-vs-composed peak-memory validation, FP16
  additivity spot-check.
- **Needs a user conversation:** **D1** (dataset/task/teacher/latency-budget/metric)
  — gates B1/B2 and CP 2.4+; only flagged, not resolved.
- **Out of scope (no LaTeX here):** all thesis-prose softening/table/limitations.

### Contracts kept & verification

`row_key` and golden hashes untouched; no catalog grid change; dummy LUT and all 3
measured rows stay valid. `resident_mem_mib`/`validate_additivity` are additive.
`bash scripts/check.sh -m "not slow"` green: ruff + mypy clean, **128 passed /
3 skipped** (+6 vs CP 2.2: 2 memory, 4 additivity). `python -m search.cost` smoke
unchanged (loud `CostError`s on the partial real LUT).
