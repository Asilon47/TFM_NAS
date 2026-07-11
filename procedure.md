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

## Cache relocation — OFA checkpoint moved into the repo (2026-06-16)

Not a checkpoint — infra-only. CP 1.2's `~/.cache/ofa/` (a stable per-user
location outside the repo) is replaced with `<project_root>/.cache/ofa/`, at
the user's request.

### Why this is safe

`~/.cache/ofa/` was chosen back at CP 1.2 specifically to survive `cd` and to
avoid colliding with `ofa.model_zoo`'s own CWD-relative `.torch/ofa_nets/`
(see procedure.md CP 1.2, "Cache location" row). Moving the cache *into* the
repo reintroduces a CWD-adjacent path, but not the CWD-relative failure mode
that motivated avoiding it: `CACHE_DIR` is now derived from
`Path(__file__).resolve().parent.parent` in `supernet/download_ofa.py`, not
from `Path.cwd()`, so the contract holds regardless of where the script is
invoked from — only *which checkout* you're in matters, which is the
intended, correct sensitivity for a per-project cache.

### Changes

- `supernet/download_ofa.py`: `CACHE_DIR = PROJECT_ROOT / ".cache" / "ofa"`
  where `PROJECT_ROOT = Path(__file__).resolve().parent.parent`.
- `.gitignore`: added `.cache/` — the 31 MB checkpoint must never be
  tracked (mirrors the existing `data/` rule).
- Docs updated to say `<project_root>/.cache/ofa/` instead of
  `~/.cache/ofa/`: `supernet/sampler.py` docstring, `supernet/README.md`,
  `PROJECT_PLAN.md` CP 1.2, `state/plan_state.yaml::cached_artifacts`.
- Migrated the already-downloaded checkpoint from `~/.cache/ofa/` to
  `<project_root>/.cache/ofa/` (`mv`, hash re-verified post-move) instead of
  letting `download_ofa.py` re-fetch 31 MB on the next run. The stale
  `~/.cache/ofa/` copy was left in place rather than deleted — it's outside
  the repo and harmless to leave; nothing reads it anymore.

### Contracts kept

`PINNED_SHA256` unchanged. `tests/test_sampler.py` keys off
`CHECKPOINT_PATH` (re-exported, not hardcoded), so it needed no change —
confirmed by re-running `bash scripts/check.sh -m "not slow"` (green) and,
in `.venv-nas`, `python -m supernet.download_ofa` (no-op, hash matches) +
`python -m supernet.sampler` (CP 1.3 smoke still forwards a sampled subnet).

## `.venv` drift repair — onnxscript export crash (2026-06-16)

Not a checkpoint; an environment-integrity fix. Resuming the Phase-0 sweep
(`python -m lut.orchestrate.run_sweep`) crashed every one of the 1610 unmeasured
rows at the ONNX export step with `ModuleNotFoundError: No module named
'onnxscript'` (0 added). Root cause was **`.venv` drift, not a missing package**:
`.venv` is pinned to `torch==2.3.1+cpu` (`requirements.txt`) but had drifted to
`torch==2.11.0+cu128` + `torchvision==0.26.0+cu128` — the NAS GPU stack, inside the
LUT venv. torch 2.11's `torch.onnx.export` defaults to `dynamo=True`, which
hard-requires `onnxscript`; torch 2.3.1's default is the legacy TorchScript
exporter, which does not. `lut/export/to_onnx.py` was correct for 2.3.1 — code
untouched.

**Decision: restore the pin, do NOT `pip install onnxscript`.** The 1100 rows
already in `data/lut.jsonl` were exported by the 2.3.1 legacy exporter (they
predate the drift). Installing onnxscript would have finished the remaining 1610
rows via the *dynamo* exporter on torch 2.11 → structurally different ONNX →
different TRT engines → latencies not comparable *within the same LUT*. Keeping one
export→TRT path across all rows is load-bearing for LUT validity, so the fix was
`rm -rf .venv && bash scripts/setup_laptop.sh` (clean rebuild to 2.3.1+cpu; the
`rm -rf` clears the +cu128 cruft a plain reinstall leaves). Verified: torch
2.3.1+cpu, `export` has no `dynamo` kwarg, first catalog block (`conv3x3`) exports
to valid ONNX with `onnxscript` never imported. The sweep then resumes idempotently
(fills only the 1610 missing rows; the 1100 are untouched). Why the venv drifted
(a stray install / wrong setup script) was not diagnosed. No checkpoint advance.

## CP 2.2 offline cost preview + additivity wiring (2026-06-16)

Not a checkpoint advance — the offline groundwork the now-complete real sweep
unblocks. **Phase 0 is DONE**: `data/lut.jsonl` holds all 2710 rows, every one
`source=jetson_trt`, `precision=fp32`, `clocks_locked=true` (the `.venv`-drift
repair above cleared the last blocker; the idempotent sweep then filled the
remainder). With a complete per-block LUT, the whole search space's *cost* side is
computable on the laptop — no Jetson, no CUDA, no dataset decision (D1).

The user chose to extract that value now and pre-wire CP 2.2's deferred DoD rather
than re-run the Jetson immediately. Two new offline tools (both `.venv`, CPU,
numpy+pandas only):

- **`search/cost_preview.py`** — samples N archs, composes each cost from the LUT
  (`search.cost.cost`), and reports the cost geometry. Headline: the rank agreement
  between a *free* FLOPs/params proxy and *measured* latency. On 2000 archs:
  FLOPs~latency Spearman 0.95 / Kendall-τ 0.81; params~latency 0.54 / 0.37; and at
  near-equal FLOPs, measured latency still spans up to 1.45×. Reading: FLOPs is a
  decent-but-imperfect proxy, params a poor one, and the LUT captures intra-FLOPs
  ordering a proxy is blind to — it earns its keep near the frontier where BO
  discriminates. Per-arch cost ranges (latency 2.1–4.8 ms sampled; resident mem
  ≤24 MiB fp32, far under the 8 GB budget — so D4's μ-penalty won't bind for these
  subnets) dump to `data/cost_preview.csv`. Promoted
  `arch_to_blocks._random_arch_dict` → public `random_arch_dict` (kept the underscore
  alias) since the preview reuses it.

- **`search/additivity_preview.py`** — wires CP 2.2's deferred DoD across the
  laptop/Jetson boundary. `manifest` picks one subnet per depth spanning 11→21 LUT
  blocks, computes each `summed_ms` from the LUT now, and writes
  `data/additivity_subnets.json` with `measured_ms: null` placeholders (pinning
  *which* whole subnets the Jetson must benchmark, so summed and measured refer to
  the same archs). `report` ingests the filled manifest and prints the depth-binned
  `AdditivityReport` (PASS / BREACH→CP 2.3). The binning is load-bearing (peer-review
  R4.2): a demo where fusion error ramps 0→+20% with depth yields a +10% *aggregate*
  that a single-number DoD would PASS while depths 19–21 correctly breach. The pure
  error-binning logic stays in `search/validate_additivity.py`; this adds only the
  LUT-driven selection + manifest I/O.

Tests: `tests/test_cost_preview.py` (16, all `.venv`/CI-safe — rank & skyline
helpers on hand-built arrays, LUT paths on a synthetic unit LUT, one on-disk smoke
that skips while partial). `pandas.*` added to the mypy `ignore_missing_imports`
override (no stubs ship; numpy ships its own). `check.sh` green (146 passed,
1 skipped).

**State unchanged: `current_checkpoint` stays 2.2, `last_completed` stays 2.1.**
This does NOT close the DoD — that still needs whole-subnet Jetson measurements (the
`measured_ms` side). It makes everything *around* the gate ready: once those numbers
exist, `report` closes (or escalates to CP 2.3) in one command. The on-device half
reuses the generic `run_sweep.run_remote_bench` path; only a "export a sampled
subnet (not a single block) to ONNX" helper is still missing for it.

## CP 2.2 closed — additivity DoD PASS + predictor calibration (2026-06-17)

The deferred half landed: whole subnets were benchmarked on the Jetson and the
measured-vs-summed DoD **PASSES**, so CP 2.2 is complete (`current_checkpoint`
2.2→2.4, `last_completed` 2.1→2.2, `completed += "2.2"`). The pre-registered CP 2.3
(residual correction) is **not** triggered — it was conditional on a depth-bin breach,
and none occurred.

**What was measured.** `lut/orchestrate/measure_additivity.py` drove 33 whole subnets
(3 per depth, spanning 11→21 LUT blocks; `data/additivity_subnets.json`) as single
TensorRT engines, reusing `run_sweep.run_remote_bench` verbatim under the same preflight
(locked clocks, power mode 0, fp32). The methodological crux: each subnet is assembled
(`search/export_subnet.py`) from the **same `catalog` block implementations the per-block
LUT timed**, chained in an `nn.Sequential` (verified channel/resolution continuity:
FIRST_BLOCK 16ch@112 → stages → 160ch@7) — *not* the real OFA modules — so
`measured − summed` isolates only cross-seam TensorRT fusion (peer-review R4.2) and not
implementation drift. The fixed stem (3→16) and head (final-expand/feature-mix/
classifier) were measured too (`--with-stem-head` → `data/stem_head_offset.json`:
stem+head = 0.388 ms, 2.67 M params, peak 2.1 MiB). Whole-net latencies never enter
`data/lut.jsonl` (no valid per-block `row_key`); they live only in the manifest + offset
JSON (both gitignored).

**DoD result — PASS, and the fusion signature is mild + flat.** Depth-binned mean signed
error `(summed − measured)/measured` (`search/validate_additivity.py`):

| depth | err | depth | err | depth | err |
|---|---|---|---|---|---|
| 11 | +6.8% | 15 | +7.1% | 19 | +7.8% |
| 12 | +7.4% | 16 | +8.5% | 20 | +7.5% |
| 13 | +8.2% | 17 | +9.2% | 21 | +8.0% |
| 14 | +8.1% | 18 | +7.8% | **agg** | **+7.9%** |

Every bin is positive (the summed LUT over-predicts — fusion shaves real latency) and
no bin nears the 15% bar; worst single arch ≈ +12%. Critically the bias is **flat in
depth**, so fusion behaves like a near-constant multiplicative discount, not the
depth-exploding error R4.2 warned could hide behind an aggregate.

**Predictor fidelity + calibration (new this session).** The user asked to go beyond
pass/fail and quantify/calibrate the predictor. `search/predictor_stats.py` (scipy-
backed) computes, over the 33 (summed, measured) pairs:

- **Ranking** (what search relies on): Spearman ρ = **0.991**, Kendall τ-b = 0.943,
  Pearson r = 0.998 (all p ≤ 1e-25). The summed-LUT predictor orders archs essentially
  exactly as the device does → BO search is faithful with the raw sum, no calibration
  needed for *ranking*.
- **Calibration** (absolute latency): OLS `measured ≈ 0.9343·summed − 0.0225 ms`
  (R² = 0.996; slope stderr ±0.011); through-origin "fusion discount" factor 0.928
  (device runs ~7.2% faster than the per-block sum). A single affine fit cuts MAPE from
  **7.85% → 1.04%** (RMSE 0.249 → 0.039 ms).

Reading: a high ρ with a *removable* (coherent multiplicative) bias is the best possible
outcome — ranking already faithful, absolute error fixed by two parameters.

**Wiring.** The fit is opt-in in `search.cost.cost(arch, lut, calibration=…)` /
`cost_from_path` (new `LatencyCalibration` contract; default `IDENTITY_CALIBRATION` is
ranking-neutral). It is applied to the **backbone sum only**, before the stem/head offset
(`latency = slope·Σblocks + intercept + offset`), matching how the fit was derived
(both manifest sides are backbone-only). A slope>0 affine map is monotonic → search
ranking is untouched whether or not calibration is on; it matters only for absolute
latency (the Phase-3 objective's `λ·latency` term + the latency budget). Persisted to
`data/latency_calibration.json` (fit + provenance; `load_latency_calibration` reads back
`slope`/`intercept`); per-subnet pairs to `data/additivity_pairs.csv` for thesis plots.
Surfaced in `additivity_preview report` (now prints the stats block; `--write-calibration`
/ `--csv` persist the artifacts).

**Dependency.** `scipy>=1.15` added to `requirements.txt` (runtime dep of
`predictor_stats`) and to the `pyproject.toml` mypy `ignore_missing_imports` override.
Installed into `.venv` (scipy 1.17.1) **without** moving the `torch==2.3.1+cpu` /
numpy 2.4.6 pin — scipy's only runtime dep is numpy, already satisfied. (User explicitly
authorized adding scipy, relaxing the earlier numpy-only stance.)

**Tests / state.** +14 tests (`tests/test_predictor_stats.py` 10; calibration in
`test_cost.py` +4; report/calibration in `test_cost_preview.py` +3; built TDD,
RED→GREEN). `check.sh` green (177 passed, 1 skipped). The offline calibration path needs
no Jetson — the measurements already exist in the manifest. CP 2.4 (eval/fine-tune) is
next but remains **blocked on CUDA + dataset decision D1**.

## CP 1.4 CLOSED — ImageNet sanity via rank fidelity (2026-06-18)

The deferred Phase-1 sanity checkpoint is **complete** (`completed += "1.4"`;
`current_checkpoint`/`last_completed` unchanged — CP 1.4 was always an out-of-order
backfill, not the critical-path head, which stays at CP 2.4). DoD **PASS**.

**What CP 1.4 verifies, and the re-frame.** The original DoD wording was "a sampled
subnet is within 1.5% top-1 of OFA's published number." Building the harness
(`eval/imagenet_sanity.py`) surfaced that that wording rests on a false premise:
OFA publishes only *fine-tuned specialized-net* accuracies (`note10_lat@…_finetune@75`,
25–75 extra epochs), so a directly-extracted subnet legitimately scores several points
lower — there is no clean absolute external anchor. What the checkpoint actually needs to
prove is that the **inherited weights + BatchNorm recalibration are intact** — a bad load
or skipped `set_running_statistics` would poison every accuracy number from CP 2.4 on.

**The first real run (2026-06-17, Kaggle GPU, full 50k val) made the right test obvious.**
Measured top-1 vs OFA's released *accuracy predictor* (the artifact OFA uses to *rank*
candidates in evolutionary search) showed a clean ~6.3pp constant offset: `max` 77.3% vs
predicted 83.6%, `min` 70.5% vs 77.7%, a random interior 75.8% vs 81.5% — and a single
offset anchored on `max` reconciled all three to <1pp with **identical rank order**. That
is the signature of a ranking model trained on a higher absolute scale (a train-holdout
subset), harmless for OFA's use and ours since search needs only the *order*. The `max`
subnet (all weights, no slicing) landing at 77.3% — exactly OFA's biggest-direct-net
ballpark — is the load-integrity proof; a broken load would be tens of points low.

**Decision (via AskUserQuestion): gate on rank fidelity (Spearman), not an absolute bar.**
This is the predictor's intended use and the strongest claim for the committee. The DoD
became: *measured and predicted top-1 rank-correlate across a spread of archs — Spearman
ρ ≥ 0.85 (p < 0.05)*, with the OLS affine fit reported as scale evidence. The harness was
pivoted to compute the statistic over a *set* of archs (the two space corners + N random
interior), **reusing `search.predictor_stats.predictor_stats`** — the exact
Spearman/Kendall/affine tooling CP 2.2 built for the latency predictor (x=predictor,
y=device convention). Added pure functions `random_archs` / `rank_pass` / `rank_summary`
(TDD, `.venv`-pure via lazy scipy import); removed the superseded absolute-bar gate
(`verdict`/`is_diagnostic`/`overall_pass`). `scipy>=1.15` added to `requirements-nas.txt`
(lazy-imported, so the pure layer and `import eval.imagenet_sanity` stay scipy-free under
`.venv`).

**DoD result — PASS.** Re-run 2026-06-18 (Kaggle GPU, full 50k val, 20 archs = max/min +
18 random; `data/results/imagenet_sanity_report.json` + `.csv`):

- **Ranking (the gate):** Spearman ρ = **0.919** (p = 1.1e-08), Kendall τ = 0.800
  (p = 1.7e-08) — measured and predicted order archs the same way. **PASS** (≥ 0.85).
- **Scale (supporting):** OLS `measured ≈ 1.105·predicted − 14.469` (r² = 0.918). The
  slope > 1 means a mild compression on top of the offset, but it is affine and monotone
  → never reorders. Calibration collapses MAPE **7.89% → 0.38%**, and **all 20 archs** sit
  within the 1.5pp band on the *calibrated* gap (`measured − affine(predicted)`; worst
  −0.81pp at `min`, +0.74pp at `rand2`).
- **External anchor:** `max` = 77.34%, `min` = 70.66% — the OFA-w1.0 direct-extraction
  corners, confirming the weight load + BN recalibration are intact.

Reading: high ρ with a *removable* (coherent affine) absolute bias is the best outcome —
the supernet ranks faithfully and the offset is fully explained by two parameters. The
"FAIL" the first run printed was an artifact of the old absolute gate, not a supernet
defect.

**State.** `tests/test_imagenet_sanity.py` swapped its 8 abs-bar tests for 6 rank tests
(21 total in-file); `check.sh` fast lane green (198 passed, 1 skipped, 3 deselected).
Reports saved under `data/results/` (gitignored). No CUDA was used on this machine — the
run is GPU-only and ran on Kaggle. CP 1.4 no longer gates anything; the critical path
remains CP 2.4 (fine-tune harness), **blocked on CUDA + dataset decision D1**.

---

## D1 resolved — pose pivot (2026-06-18)

Not a checkpoint — a **decision** (D1) plus a CPU-verified prototype. `current_checkpoint`
stays **2.4**; D1 was the non-CUDA half of CP 2.4's block, now cleared.

### The decision

The user supplied the target dataset (`dataset/`) and asked to adopt it "from now on." It is
an **Ultralytics YOLO-pose** dataset — 1 class `gate`, **8 keypoints**, 2842 train / 140 val
synthetic A2RL drone-racing renders (see `dataset/SCHEMA.md`). Their existing stack
(`yolo-ros2-inference/`) already deploys **yolo11n-pose** on the Jetson Orin Nano
(`yolo11n-pose-jetson-fp16.engine`, 640, FP16 TRT). So D1 resolves to **gate detection +
8-keypoint pose** — a re-frame from 1000-class ImageNet classification to detection/pose
(metric: **pose mAP / OKS**).

This was an open decision (D1, "do not resolve unilaterally"). Three sub-decisions taken via
AskUserQuestion:

1. **NAS strategy = OFA backbone + YOLO-pose head.** Each OFA-MBv3 subnet is searched as a
   *backbone*; a YOLO11-pose neck/head is grafted on top. This keeps the entire investment —
   supernet, sampler, latency LUT, Net2Net, BO — and makes the ImageNet pretrain (CP 1.2/1.4)
   the backbone *warm-start*, which is exactly its right role for a detection backbone.
2. **Baseline + teacher = yolo11-pose.** The baseline-to-beat becomes the deployed
   **yolo11n-pose** (on the Jetson accuracy/latency frontier); the Phase-8 distillation teacher
   becomes a bigger **yolo11s/m/l-pose** (reuse `yolo-ros2-inference/scripts/yolo_distillation.py`).
   This replaces the old "Pareto-dominate MobileNetV3 on ImageNet" headline.
3. **Accuracy metric = pose mAP (OKS).** Reused from Ultralytics' validator (`metrics.pose.map`),
   not re-implemented.

### What carries over vs. changes

Keep: OFA supernet/sampler/Net2Net/BO (backbone now), the ImageNet pretrain + CP 1.4 sanity
(warm-start), the Jetson LUT pipeline + TRT methodology (latency is task-agnostic). Change:
the accuracy harness (top-1 → pose mAP), the `J(α)` accuracy term, the baseline/teacher.

### The backbone-tap design (the technical crux)

OFA-MBv3-w1.0 makes only ks/e/d elastic; the **stage output widths are fixed**
(24/40/80/112/160). Stages 1/3/4 end at strides 8/16/32, so a subnet's `(P3, P4, P5)` taps are
the **last-block indices of those stages** — pure cumulative sums of the active depths
(`stage_tap_indices(d) = (sum(d[:2]), sum(d[:4]), sum(d[:5]))`; `blocks[0]` is OFA's fixed first
block). Because the widths are fixed, the tap channels are **invariant across the whole search
space**: `(40, 112, 160)` for *every* arch. So **one fixed neck/head/adapter serves every
sampled backbone** — what makes "search the backbone, freeze the head" tractable. At 640×640
the taps are 80²/40²/20² — the canonical YOLO P3/P4/P5 scales.

### Prototype built + CPU-verified (`.venv-nas`, no CUDA needed for a forward)

- `supernet/pose_backbone.py` — `stage_tap_indices` (pure, TDD'd under `.venv`) +
  `PoseBackbone(subnet, depths)` (wraps a sampled subnet, drops the classifier, returns the
  three taps). Real-OFA `__main__` smoke: a random subnet forwards `(1,3,640,640) →
  P3(1,40,80,80) P4(1,112,40,40) P5(1,160,20,20)` — confirming the tap math against the actual
  sampled block list (`len(blocks) == 1 + sum(d)`), with pretrained weights.
- `detect/` — `ChannelAdapter` (1×1 convs (40,112,160)→(64,128,256); torch-only, TDD'd under
  `.venv`); `pose_model.build_pose_model` grafts a **real `ultralytics.nn.modules.head.Pose`**
  onto backbone+adapter; `evaluate.pose_map` wraps Ultralytics' pose validator and rewrites
  `dataset.yaml`'s stale absolute `path:` at run time. Real-OFA `__main__` smoke: forwards
  `(1,3,640,640)` → Pose head `boxes(1,64,8400) scores(1,1,8400) kpts(1,24,8400)` (24 = 8×3
  keypoints, 8400 = 80²+40²+20² anchors, scores ch = nc = 1). The graft works end-to-end.
- Tests: `tests/test_pose_backbone.py` (8, incl. a stub-backbone forward), `tests/test_pose_adapter.py`
  (3). `check.sh` fast lane green (**209 passed**, 1 skipped, 3 deselected).

### Infra fixes (discovered while installing ultralytics)

- `requirements-nas.txt` += `ultralytics>=8.3` (installed 8.4.70; torch 2.11.0+cu128 / tv
  0.26.0+cu128 / ofa pins all intact; its opencv/matplotlib/pandas deps are CPU/pure).
- `scripts/setup_laptop_nas.sh` was **broken on this machine**: the repo was moved
  (`…/lookup_table` → `…/TFM_NAS`), so the checked-in `.venv-nas/bin/activate` exports a stale
  `VIRTUAL_ENV` (and even references `cygpath`); `source activate` left bare `python` pointing
  at the system, externally-managed interpreter → pip aborted with **PEP 668** on the first
  command. Fixed: invoke `.venv-nas/bin/python` by absolute path (not `activate`) and clear
  ROS's leaked `PYTHONPATH` (same guard `scripts/check.sh` already uses). NB the user's *manual*
  `source .venv-nas/bin/activate` (per CLAUDE.md) is still stale — regenerate the venv
  scaffolding or invoke the interpreter directly.
- `.gitignore` += `dataset/*` (with `!dataset/SCHEMA.md`): the 1.6 GB payload is no longer
  committable, the schema doc stays tracked. `pyproject.toml`: `detect` added to mypy `files`,
  `ultralytics.*` to the no-stubs override, `yolo-ros2-inference` to ruff `extend-exclude` (it
  is the user's separate ROS2 repo, not NAS source).

### Consequence owed (not done here)

The LUT rows are keyed by per-block `input_shape` derived from **224**. Pose runs at **640**, so
every block's feature-map shape differs → new `row_key`s → the measured 224-LUT does not cover
them. The append-only, `input_shape`-keyed schema absorbs this natively: a **second LUT sweep at
the deployment resolution** (recommend 640) plus a resolution parameter on
`search/arch_to_blocks` + the catalog grid. `search/cost.py`'s constant offset generalizes
(stem + pose neck/head instead of the classifier). Also owed: anchor the baseline (yolo11n-pose
Jetson latency + val pose mAP), and wrap the grafted backbone as an Ultralytics model for
end-to-end pose train/val (the fresh head needs a short fine-tune to be meaningful). All
CUDA/Jetson-gated.

### State

`current_checkpoint` 2.4 (unchanged), `last_completed` 2.2, `completed` unchanged. CP 2.4's
metric is now pose mAP (`PROJECT_PLAN.md` CP 2.4 + the D1 entry re-scoped; CP 8.1 teacher →
yolo11-pose). The actual fine-tune remains **CUDA-gated**. `CLAUDE.md` is agent-write-protected,
so its one-paragraph summary, the CP 1.4 line, and the D1 row need a manual update by the user.

## CP 2.4 — CPU slice (trainable graft + harness + DoD gates) (2026-06-18)

Built the **CPU-buildable slice** of CP 2.4 (the rest is GPU-gated). The forward-only prototype
from the D1 pivot could only run inference; the graft is now **trainable and eval-able
end-to-end**, and both DoD checks are coded + unit-tested. No checkpoint advance — the DoDs
themselves need the GPU fine-tunes.

**The trainable graft (`detect/pose_model.py`).** Added `GraftedPoseModel`, a subclass of
Ultralytics' `ultralytics.nn.tasks.PoseModel`, plus `build_grafted_pose_model(arch)` (factory:
sample OFA subnet → `PoseBackbone` → `ChannelAdapter` → fresh `Pose` head w/ `bias_init`). The
subclass overrides **only** (a) construction — it skips `DetectionModel.__init__` (yaml build +
stride-inference forward) via `nn.Module.__init__`, holding the assembled parts in
`self.model = Sequential(backbone, adapter, head)` so `self.model[-1]` is the Pose head; and
(b) `_predict_once` — runs `backbone → adapter → head` directly, because the OFA/adapter modules
lack Ultralytics' per-layer `.f`/`.i` routing the inherited loop assumes. Everything else
(`loss()`, `init_criterion()` → `v8PoseLoss`, `_apply` stride-moving) is **inherited unchanged**
— they only ever reach the head through `self.model[-1]`. The wrapper exposes the attributes the
loss/validator read: `.args` (loss gains box/cls/dfl/pose/kobj via `get_cfg(DEFAULT_CFG)`),
`.names` (`{0: gate}`), `.nc`, `.kpt_shape`, `.stride`, `.yaml`, `.task`. The subclass is built
lazily through module `__getattr__` (PEP 562) so `import detect.pose_model` stays
ultralytics-free under `.venv` (the contract; only touching `.GraftedPoseModel` pulls ultralytics
in).

**Grafted eval (`detect/evaluate.py`).** Added `pose_map_model(model, …)`: the existing `pose_map`
drives the high-level `YOLO` wrapper (baseline/teacher anchoring) which a bare graft lacks, so
this runs `PoseValidator` directly against the graft (AutoBackend reads `.stride`/`.names`/
`.kpt_shape`; dataloader from the path-rewritten yaml). **CPU-verified** on the real 140-img val
split (random head → mAP≈9e-8, as expected; the point is the OKS pipeline *runs*).

**The harness (`eval/shortft.py`).** `short_finetune(arch, …)` = seed → `build_grafted_pose_model`
→ Ultralytics pose dataloader (`_build_pose_loader`, **`cfg.task='pose'`** so keypoints load) →
AdamW loop (`loss.sum().backward()`) → `pose_map_model`. Train path CPU-smoked: real pose batch
(keys `img/batch_idx/cls/bboxes/keypoints`) → grafted model → finite v8PoseLoss → backward/step.
Plus the two **DoD gates**: `rank_fidelity(proxy, full)` → `RankFidelity{kendall_tau, spearman,
passes}` (the search gate, τ ≥ 0.7, scipy) and `reproducible(a, b)` (twice within 0.5 mAP pts =
0.005 absolute, since pose mAP is a [0,1] fraction). `v8PoseLoss.loss` returns `(5-vector ×
batch, detached)` (box, pose, kobj, cls, dfl) → the loop sums before backward.

**Why this design (vs a thin `v8PoseLoss` loop).** Subclassing the real `PoseModel` keeps the
graft a first-class Ultralytics model → the user's `yolo_training.py` / `yolo_distillation.py` /
`.val()` can drive it later unchanged, and it sets up the Phase-8 distillation graft. Cost: the
two small overrides above. The cls-BCE term flows gradient to the backbone on *every* anchor, so
the grad-path test is robust even when no anchor matches the synthetic box.

**Tests / verification.** `tests/test_shortft.py` (10, scipy-only → `.venv`/CI: the two DoD
gates incl. the one-swap τ=2/3<0.7 boundary). `tests/test_grafted_pose_model.py` (6,
`.venv-nas`: head-is-`model[-1]`, `.args`/metadata contract, train-mode predict dict, **loss
finite + grad reaches the backbone stem**, and a slow 1-image overfit that drives loss down with
a stub backbone — *no OFA checkpoint needed*). `python -m detect.pose_model` overfit smoke:
loss 24.93 → 8.97 / 15 steps. Built TDD (RED→GREEN for both helper + wrapper). `check.sh` fast
lane green: **219 passed**, 2 skipped (grafted test skips without ultralytics; sampler without
ofa), 3 deselected; ruff + mypy clean.

**Remains (GPU-gated):** the real ~5-epoch fine-tune + both DoDs — reproducibility (same arch
twice within 0.5 %) and proxy-rank Kendall-τ ≥ 0.7 over ~8–12 archs vs a full-train ranking
(peer-review R2.1 / P0.2; gates the whole search). Run on Kaggle / Jetson. Parallel CPU-runnable
item: anchor the baseline yolo11n-pose **mAP** via `pose_map` (its Jetson **latency** stays gated).

**Protocol driver (`eval/proxy_rank.py`, the one-command GPU run).**
`python -m eval.proxy_rank --archs 10 --proxy-epochs 5 --full-epochs 100 --device cuda` runs the
whole DoD: `sample_archs` picks N archs spanning the space (min/max **corners** + seeded randoms —
uniform sampling clusters mid-depth and weakens τ), scores each under proxy + full via
`short_finetune`, reruns arch 0 at **seed+1** for the reproducibility floor (same seed twice is
bit-identical → a meaningless pass), and `assemble_verdict` → Kendall-τ/Spearman + repro +
PASS/FAIL (process exit 0 ⇔ DoD pass). **Resumable:** every per-arch result is flushed to
`data/cp24_proxy_rank.json` (gitignored), so a Kaggle timeout (a full train of 10 archs can exceed
the ~9–12 h cap) continues, not restarts; verdict → `<out>.verdict.json`. Pure logic
(verdict/corners/JSON-resume) TDD'd in `tests/test_proxy_rank.py` (7, `.venv`); the loop is
CPU-smoked via `--max-steps` (2 corner archs, proxy+full+repro, resume verified, no repo
`runs/` pollution). +7 tests; `check.sh` fast lane **226 passed**.

### State

`current_checkpoint` 2.4 (unchanged), `last_completed` 2.2, `completed` unchanged — CP 2.4 stays
**open** until the GPU DoDs pass. `CLAUDE.md` updated (guard lifted earlier this session): the
CP 2.4 line, blockers, and "lowest-friction next build" now reflect the built CPU slice.

---

## CP 2.4 — GPU run FAILED both DoDs → diagnose-first (2026-06-21)

The Kaggle GPU run of `python -m eval.proxy_rank` landed (`data/cp24_proxy_rank.json` +
`.verdict.json`). **Both DoD gates failed:**

| Gate | Result | Threshold |
|---|---|---|
| Proxy-rank fidelity | Kendall-τ = **0.20** | ≥ 0.70 |
| Reproducibility | Δ = **0.0149** (1.5 mAP pts, arch 0 at seed vs seed+1) | ≤ 0.005 |

### What the 10-arch data says

Per-arch (proxy → full pose-mAP): the **min corner** (idx 0) is correctly lowest in both
(0.50 → 0.778); the **max corner** (idx 1) highest full (0.850). But the 8 random archs' full-train
mAPs **cluster in 0.823–0.850** (spread ≈ 0.024), and proxy vs full are **uncorrelated** among them
(e.g. idx 8 is 2nd-best by full, dead last by proxy; idx 9 best by proxy, mid by full). So the only
signal the 5-epoch proxy captured is *"the smallest net is worst"* — that single arch contributes
~all of τ=0.20 (net ≈ 9/45 concordant pairs). The proxy is trying to resolve accuracy gaps smaller
than its own noise. The reproducibility and rank failures share a root: `build_grafted_pose_model`
fine-tunes a **randomly-initialized** YOLO-pose head for only 5 epochs, so the proxy partly measures
head-init luck (eval is deterministic given weights → the 1.5-pt gap is training-trajectory noise).

### Decisions (AskUserQuestion)

- **Q1 → diagnose first.** The data can't tell us whether the *full-train* ranking of the clustered
  archs is itself reliable. If full-train noise ≈ the cluster spread, the synthetic gate task does
  **not** separate archs on accuracy and *no* proxy can pass — reframe. If full-train is stable, the
  proxy is the (repairable) problem. Measure the full-train noise floor before spending repair compute
  (PROJECT_PLAN CP 2.4 "below threshold → repair the proxy first" branch).
- **Q2 → decide head warm-start after the diagnostic.** So no warm-start was built this pass.

### Built — the full-train noise diagnostic (`eval/proxy_rank.py`)

Extends the existing driver (reuses `ArchResult`, the resumable JSON, `short_finetune`); the 10
existing seed-0 full maps are **reused** as ground truth — the diagnostic only adds ~3 *new*
full-trains at `seed+1`.

- **`full_noise_verdict(reseed, cluster_maps)` (pure, TDD'd):** `noise_floor = median |seed1−seed0|`,
  `cluster_spread = max−min` of the clustered full maps (the global-min corner is dropped — it's the
  one trivially-separable outlier), `snr = spread / noise_floor` → `discriminates` (≥ 2) / `flat`
  (≤ 1) / `ambiguous`; plus per-arch deltas and the seed0↔seed1 Kendall-τ.
- **`ArchResult.full_map_reseed`** — new optional field (back-compatible; old JSON loads on the
  default `None`).
- **`run_full_diagnostic(indices=…)` + `--diagnose-full`/`--indices` CLI:** reads the prior results,
  reruns the chosen archs' full-train at `seed+1` into `full_map_reseed` (resumable per-arch flush),
  writes the verdict to `<out>.diagnostic.json`. Default `--indices 7,4,8` (spans the cluster).
- **Tests:** `full_noise_verdict` (discriminates / flat / ambiguous / deltas / ≥1-reseed guard) +
  `ArchResult` round-trip & old-record back-compat. Because **`.venv-nas` is not built on this
  laptop** (only `.venv`), `run_full_diagnostic`'s resume/guard/verdict-writing path is covered by a
  **stubbed-fine-tune** test (monkeypatch `eval.shortft.short_finetune`, `supernet=object()`) — the
  real fine-tune is the Kaggle step. CLI dispatch smoked via the missing-prior guard. `check.sh` fast
  lane **236 passed**, 2 skipped, 3 deselected; ruff + mypy clean.

### The decision the diagnostic feeds (next pass)

- **`discriminates`** → ground truth real → repair the **proxy** (revisit Q2 head warm-start +
  epochs 5→10–15 + LR schedule; re-run **proxy-only** and re-correlate against the existing full maps
  — cheap). Target τ ≥ 0.7.
- **`flat`** → task doesn't separate archs → **reframe** (accuracy as a constraint, latency the
  objective; adjacent to open decision **D4** → escalate, don't resolve unilaterally).
- **`ambiguous`** → tighten the full-train reference too (epochs / eval protocol) *and* the proxy.

Kaggle command (prior `data/cp24_proxy_rank.json` present for resume):
`python -m eval.proxy_rank --diagnose-full --indices 7,4,8 --full-epochs 100 --device cuda`.

### State

`current_checkpoint` **2.4** (unchanged), `last_completed` 2.2, `completed` unchanged — CP 2.4 stays
**open** (failed; diagnosing). Existing seed-0 `full_map`s preserved (the diagnostic writes only
`full_map_reseed` + a separate `.diagnostic.json`).

## CP 2.4 — repair: head warm-start + freeze (2026-06-21)

Before running the 300-epoch diagnostic, a **read-only investigation** (no GPU; correlated the 10
existing archs against zero-cost LUT descriptors via `search.cost.cost_from_path`) updated the picture
enough to change the plan.

### Investigation — the proxy is the noise, not the task

| ranker vs `full_map` | Kendall τ (all 10) | τ (8 randoms, no corners) |
|---|---|---|
| depth `sum(d)` | **+0.767** (passes gate) | +0.596 |
| Jetson latency | **+0.733** (passes gate) | +0.571 |
| FLOPs | +0.689 | +0.500 |
| params | +0.556 | +0.286 |
| **5-epoch fine-tune proxy** | **+0.200** | **0.000** |

- The full-train mAP **tracks size strongly and stays ordered even inside the "cluster"** — so the
  ground truth is real, not flat (the original diagnose-first worry). A free layer-count out-ranks the
  GPU fine-tune.
- The proxy correlates with **nothing** (τ=0.20; **−0.08 once the min corner is dropped**; τ=0.07 vs
  FLOPs). Regressing `full_map` on depth (r²=0.71) leaves an architectural residual of only
  **stdev ≈ 0.010 mAP** — the signal a *good* proxy must resolve.
- **Root cause = the randomly-initialized Pose head.** Smoking gun: **idx8 = 2nd-best full-train
  (0.846) but the worst proxy (0.5705)** — a good backbone sabotaged by 5 epochs of learning a head
  from scratch. Same root as the 1.5-pt reproducibility gap (re-seed → re-roll the head).
- (Full numbers + residual table in the plan file `ticklish-popping-mountain.md`, "Investigation
  addendum".)

### Decision (AskUserQuestion → "fix the head first")

Given the ground truth is clearly size-structured (the "flat" scenario that motivated diagnose-first
is now disfavored) and the random head is the proven culprit, **fix the head and re-test cheaply**
before spending the 300-epoch diagnostic. The diagnostic stays the **fallback** if the warm re-test
still misses. (`detect/pose_model.py:9-12` already named the trained-head clone as the intended next
step; the adapter was built to feed the head's `(64,128,256)` inputs.)

### Built — head warm-start + freeze

- **`detect.pose_model`:** `warm_start_head(head, donor_state)` — **shape-aware partial `state_dict`
  copy** (copy where key+shape match, leave the rest at init, raise if *nothing* matches); serves both
  the gate donor (nc=1/8-kpt → whole head transfers) and COCO (17-kpt → keypoint branch reinitialized).
  `freeze_module(m)` (`requires_grad_(False)`), `_donor_head_state(pt)` (lazy ultralytics —
  `YOLO(pt).model.model[-1].state_dict()`). `build_grafted_pose_model` gains `head_weights=`/
  `freeze_head=` (after `bias_init()`).
- **`eval.shortft.short_finetune`** gains `head_weights=`/`freeze_head=`; optimizer now over
  `[p for p in model.parameters() if p.requires_grad]` — a frozen head is excluded, so the short
  fine-tune adapts only backbone+adapter to a fixed, competent head (the proxy becomes a
  **backbone-quality probe**, not a head-init lottery).
- **`eval.proxy_rank`:** `run_protocol` gains `head_weights`/`freeze_head`/`reset_proxy`; the
  `load_supernet` import made **lazy** (parity with `run_full_diagnostic`, so a supplied supernet skips
  `ofa`). `--reset-proxy` nulls loaded `proxy_map`s (keeps `full_map`) for a warm-head re-test; CLI
  `--head-weights/--freeze-head/--reset-proxy` added.
- **Tests (TDD, `.venv`/CI — torch CPU, no ultralytics):** new `tests/test_warm_start.py` (copy-all on
  shape-match / skip-mismatch / leave-unmatched-at-init / raise-when-nothing-matches / freeze+optimizer
  filter) + a `run_protocol` reset-proxy orchestration test (stubbed fine-tune; asserts proxy
  recomputed with the warm head, full maps preserved, kwargs threaded). `check.sh` fast lane
  **242 passed**, 2 skipped, 3 deselected; ruff + mypy clean. CLI `--help` smoked.

### GPU re-test owed (Kaggle) + criteria

Work on a **copy** (never overwrite the only expensive ground truth):
```
cp data/cp24_proxy_rank.json /kaggle/working/cp24_warmstart.json
python -m eval.proxy_rank --reset-proxy --head-weights <gate-yolo11n-pose.pt> --freeze-head \
    --no-full --device cuda --imgsz 640 --batch 16 --out /kaggle/working/cp24_warmstart.json
```
Read `…cp24_warmstart.json.verdict.json`: **τ ≥ 0.7 & Δ ≤ 0.005 → CP 2.4 closes** (advance state); a
miss → run the (already-built) `--diagnose-full` to decide repair-more vs reframe (D4 → user). Donor:
the **gate** checkpoint freezes cleanly; with only COCO `yolo11n-pose.pt`, drop `--freeze-head` (its
reinitialized keypoint branch must train).

### State

`current_checkpoint` **2.4** (unchanged) — CP 2.4 stays **open** until the warm re-test clears the gate.
Original `data/cp24_proxy_rank.json` untouched (the re-test runs on a copy). No golden-hash / LUT
changes.

## CP 2.4 — deep-research + zero-cost ranker (Tier-1A, no GPU) (2026-06-22)

Not a checkpoint advance. While the donor trained on Kaggle, ran a literature pass (deep-research) for
alternatives to the failed 5-epoch proxy, then built + validated the cheapest one. Full report:
`~/.claude/plans/mode-full-research-piped-sunrise.md`.

**Diagnosis (literature-confirmed).** The proxy failure is two named effects, not a tuning miss:
(1) **random-head distortion** — a random task head emits large gradients that distort the pretrained
backbone during short fine-tuning, so the score is head-init luck not backbone quality (Kumar et al.,
"Fine-Tuning can Distort Pretrained Features"/LP-FT, ICLR 2022; "How to prepare your task head", ICLR
2023). Their stated ranking consequence — *backbones that fine-tune poorly may actually be superior* —
is idx8 exactly. (2) **top-k/cluster collapse** — every proxy collapses to ~random within a
similar-size cluster while #params/#FLOPs dominate on wide ranges (Zero-Shot NAS survey, arXiv
2307.01998; NAS-Bench-Suite-Zero, NeurIPS 2022). The Δ reproducibility half is score variance
("Variation Matters", arXiv 2502.19657 — average over seeds/batches).

**Built (TDD, `.venv`/CI, no GPU).** `eval/zerocost.py` — zero-cost descriptors (`depth_sum`, plus
`params`/`flops`/`latency_ms` from `search.cost.cost`), `zerocost_score`, `rank_report`, and a
reproducible `__main__`. `eval/shortft.py` gained `precision_at_k` + `top1_regret` (pure, additive; the
τ gate + `KENDALL_TAU_GATE` untouched). New tests `tests/test_zerocost.py` + `tests/test_rankmetrics.py`
(12); `check.sh` fast lane **254 passed**, 2 skipped, ruff + mypy clean. Committed `f184842`.

**Validated vs the seed-0 ground truth** (`data/cp24_proxy_rank.json`, read-only):

| ranker | Kendall τ | Spearman | precision@3 | top1_regret | gate |
|---|---|---|---|---|---|
| 5-epoch proxy (failed) | 0.200 | 0.212 | 0.33 | 0.0195 | fail |
| **depth_sum** | **0.767** | 0.843 | 0.67* | **0.000** | **PASS** |
| latency_ms (Jetson) | **0.733** | 0.855 | 0.67 | 0.000 | PASS |
| flops | 0.689 | 0.842 | 0.67 | 0.000 | fail |
| params | 0.556 | 0.685 | 0.67 | 0.000 | fail |

*precision@3 is tie-sensitive (depth_sum integer ties); τ + regret are tie-robust. **Every** descriptor
picks the true-best arch (regret 0); the failed proxy is the only ranker that doesn't. The zero-cost
ranker dominates the failed proxy on every metric — and the DoD's τ-on-10 gate mis-measures (params/flops
"fail" τ yet have regret 0), evidence for switching the gate to Spearman + precision@k.

**Owed / pending user decision (D4-adjacent, do not resolve unilaterally).** Two CP 2.4 paths now exist:
*repair* (warm-head re-test, GPU — donor `runs/pose/.../best.pt` ready) vs *reframe* (adopt the zero-cost
ranker, no GPU). Plus the DoD-gate change. GPU upgrade if reframing: a ZiCo/jacob_cov gradient proxy on
the backbone (`eval/zerocost.py` is the CPU descriptor half). State stays **2.4**.

### Both paths implemented (2026-06-25 — user: "do your recommendation, both")

- **Reframe (no GPU) — DONE.** Search-relevant DoD gate `rank_verdict`/`RankVerdict` in
  `eval/shortft.py`: passes iff **Spearman ρ ≥ 0.70 AND top1_regret ≤ 0.01** (τ + precision@k carried
  as diagnostics; proposed thresholds, tunable). Wired into `eval/zerocost.rank_report`. Under it
  (`python -m eval.zerocost`): depth_sum / latency_ms / flops **PASS**, the 5-epoch proxy + params
  (ρ=0.685) fail — a cleaner separation than τ. Commit `4dc3fc5`.
- **Repair (GPU-run) — code DONE.** `eval/proxy_rank.py` `--proxy-seeds N` averages each arch's proxy
  mAP over N seeds (per-seed flushed to `ArchResult.proxy_seed_maps` for mid-arch resume; repro rerun
  compares two independent averaged estimates) — the Δ fix per "Variation Matters". Commit `bf177c2`.
- **NOT building full LP-FT (deliberate, refines my own recommendation).** For *ranking* backbones the
  head must be **identical** across archs to isolate backbone quality; full LP-FT lets the head
  fine-tune per-arch, re-introducing head variance. So **warm-start + freeze-head** (already built) is
  the better ranking variant than LP-FT here — the repair cross-check is `--reset-proxy --head-weights
  best.pt --freeze-head --proxy-seeds 3`.
- **Objective `J(α)` integration intentionally deferred** — λ/μ + normalizing the zero-cost score into
  an accuracy term *is* D4/CP 3.3; building it now would bake that decision. `zerocost_score` already
  is the adopted accuracy signal.
- `check.sh` fast lane **261 passed**, 2 skipped, ruff + mypy clean. State stays **2.4** (the reframe
  gate + zero-cost ranker are the proposed close; the warm-head re-test is the GPU cross-check).

## CP 2.4 — donor trained, warm-head re-test scheduled on Colab (2026-06-22)

Not a checkpoint advance (state stays **2.4**). The blocker for the warm-head re-test was the donor:
the team had only the **deployed model as a fused ONNX** — unusable (Ultralytics export folds BN into
Conv, so the un-fused head `state_dict` keys `warm_start_head` matches on are gone; it is also a
flattened, non-differentiable graph). The repo-root `yolo11n-pose.pt` is the stock COCO release
(`train_args.data=coco-pose.yaml`, 17-kpt, `epoch:-1`) — the `--pretrained` seed, not a donor. So we
**trained our own gate donor** (user decision, AskUserQuestion). See [[cp24-donor-must-be-trained]].

**Donor run — `runs/pose/experiments/gate_baseline/` (1396 epochs, D1 gate dataset, exp15 recipe:
nc=1/8-kpt, `pose=50`, `kobj=10`, SGD, imgsz 640, `multi_scale`).** Trajectory (read from
`results.csv`, no GPU):

- **Keypoint branch converged ~epoch 300** — pose mAP50-95 peaked **0.886 @1139**, flat since (Δ +0.0015
  over the last 400 ep); pose mAP50 peaked 0.945 @201. For a *frozen* pose donor this is the metric that
  matters, and it's been done for ~1000 epochs.
- **Box branch still creeping** (+0.04 over the last 400 ep, decelerating) — kept Ultralytics' fitness
  rising, which is why `best.pt` landed late, but it's irrelevant to donor competence (the **deployed
  ONNX is the true baseline anchor**, not this model).
- **Donor = `best.pt` (epoch 1359, fitness 1.608):** best box mAP50-95 0.728 + near-peak pose 0.878 →
  dominates the custom `best_img640.pt` (@321: same pose 0.878 but box ~0.646). **Conclusion: stop
  training; `best.pt` is a strong converged donor.**

**Execution = Google Colab free T4** (user decision, AskUserQuestion). Kaggle weekly GPU quota is
exhausted and **TPU cannot run the PyTorch/OFA/Ultralytics stack** (no `torch_xla` path in the code,
Ultralytics has no TPU device, OFA's dynamic elastic ops force XLA recompiles). Colab's local disk is
ephemeral, so inputs + outputs sit on **Google Drive**; combined with `proxy_rank`'s per-arch flush, a
dropped session resumes on re-run. Re-test (on a COPY of the seed-0 maps, never the original):
`python -m eval.proxy_rank --reset-proxy --head-weights best.pt --freeze-head --proxy-seeds 3 --no-full
--device cuda --imgsz 640 --batch 16 --out <drive>/cp24_warmstart.json`. Verify the printed
`warm_start_head` copied/skipped is ~all-copied (donor is nc=1/8-kpt → whole head shape-matches the
graft). Gate: **τ≥0.70 & Δ≤0.005 → CP 2.4 closes**; miss → the built `--diagnose-full`. `check.sh`
fast lane re-confirmed **261 passed** before handoff. See plan `ticklish-popping-mountain.md`.

## CP 2.4 CLOSED — warm-head re-test + reframe gate (2026-06-27)

**`current_checkpoint` 2.4 → 3.1, `last_completed` 2.2 → 2.4, `completed += "2.4"`.** The warm-head
re-test ran on Colab (the file landed `data/cp24_warmstart.json`, 3 proxy seeds/arch ⇒ `--proxy-seeds 3`,
full maps byte-identical to the seed-0 originals ⇒ `--reset-proxy` kept the expensive ground truth). The
docs that called it "owed" were stale.

### The result — the repair worked

Warm proxy (warm-started + frozen gate head, 3-seed mean) vs the original seed-0 full maps, 10 archs:

| metric | original (random head, 1 seed) | warm-head (frozen, 3-seed) | old gate | reframe gate |
|---|---|---|---|---|
| Kendall-τ | 0.20 | **0.60** | ≥0.7 ✗ | diagnostic |
| Spearman ρ | 0.21 | **0.77** | — | ≥0.70 ✓ |
| top-1 regret | 0.0195 | **0.00** | — | ≤0.01 ✓ |
| precision@3 | 0.33 | 0.67 | — | diagnostic |
| reproducibility Δ | 0.0149 | 0.0145 | ≤0.005 ✗ | diagnostic |

Freezing a competent head tripled τ (0.20→0.60), nearly 4×'d ρ (0.21→0.77), and the proxy now picks the
**true-best arch** (idx1, the max corner; top-1 regret 0). Direct empirical confirmation of the LP-FT
random-head-distortion root cause — the original proxy scored head-init luck, not backbone quality.

### Decision (D4-adjacent, AskUserQuestion → "Close on reframe gate")

CP 2.4's DoD is **reframed**: **Spearman ρ ≥ 0.70 AND top-1 regret ≤ 0.01** (`eval.shortft.rank_verdict`),
superseding the Kendall-τ-on-10 + Δ≤0.005 gate. Rationale (literature pass + the data): τ-on-10 has very
wide CIs at n=10 and punishes mid-rank disagreements the search ignores — it *mis-measures* (size
descriptors fail τ yet have regret 0 / pick the true best). Both the warm-head proxy (ρ=0.77, regret 0)
**and** the zero-cost ranker (depth_sum ρ=0.843, latency_ms ρ=0.855, regret 0) pass the reframe gate ⇒
CP 2.4 closes. **Only the DoD-gate sub-decision is resolved; D4 proper (λ/μ in J(α)) stays open** (CP 3.3).

### Reproducibility — re-characterized as a diagnostic, not a gate

Δ=0.0145 is two independent 3-seed-block means of **idx0, the min corner** — the smallest / most
under-trained arch at 5 epochs, the worst case. Seed-averaging (N=3) did *not* reduce it vs the 1-seed
run (0.0149→0.0145): within a block the spread is ~0.002 (std), but two blocks differ by 0.0145 — i.e. the
noise is **non-i.i.d.** (a correlated/systematic component averaging can't shrink; σ/√N only kills i.i.d.
noise). It is therefore not cheaply fixable by more seeds, and — crucially — it does not affect *rank*
quality (the proxy's ordering is stable; it picks the true best). Under the reframe it is reported, not
gated. Honest caveat for the thesis: report Δ on the min corner as run-to-run noise; the rank robustness
(ρ=0.77, regret 0) is the search-relevant claim.

### Carries into Phase 3

- **Accuracy signal = the warm-head 5-epoch proxy** (warm-start + freeze the gate head). It is a *real*
  accuracy estimate, partially independent of size — what a latency↔accuracy Pareto search needs. (A
  zero-cost ranker that is a monotone function of latency would collapse the Pareto front to a line.)
- **Zero-cost descriptors = free cold-start prefilter** (depth_sum / Jetson latency_ms; `eval/zerocost.py`,
  no GPU) — they agree with the proxy (both pick the true best), a cheap robustness cross-check and a
  warm-start for BO before any fine-tune is spent.
- **J(α) λ/μ integration deferred to CP 3.3** (D4 proper) — `zerocost_score`/`rank_verdict` are the
  building blocks; normalizing into an accuracy term + choosing λ/μ is the next user decision.

### Built (CPU-only close — no further GPU)

`eval/proxy_rank.assemble_verdict` now gates on `rank_verdict` (emits `spearman`/`top1_regret` as the
gate + `kendall_tau`/`precision_at_k`/`reproducibility` as diagnostics; `dod_passes = rank_passes`;
precision@k clamped to k≤n; verdict JSON carries `spearman_gate`/`regret_tol` ⇒ self-describing). New
`reverdict()` + `--reverdict` re-stamp an existing results file's verdict under the current gate with no
fine-tune (scipy+json, runs in `.venv`), preserving the prior reproducibility block. Re-stamped
`data/cp24_warmstart.json.verdict.json` → `dod_passes: true`. Tests updated to the reframe semantics
(`test_verdict_reproducibility_is_diagnostic_not_gate` is the flipped behavioral spec) + `reverdict`
round-trip/guard. `check.sh` green: **266 passed, 2 skipped** (ofa/ultralytics → `.venv-nas`), ruff +
mypy clean. Commit `53ff58a`. The owed 640-res LUT re-sweep + baseline yolo11n-pose anchor remain
Jetson-gated (Phase-3-adjacent, not a CP 2.4 blocker).

## CP 3.1 CLOSED — search-space encoder (2026-06-27)

First Phase-3 checkpoint, and the only one with zero blockers (pure / CPU-only / no GPU / no Jetson).
`search/space.py` encodes an OFA `arch_dict` to a length-45 flat vector of **category indices**
(`[ ks(20) | e(20) | d(5) ]`) and back — the input surface the CP 3.2/3.3 surrogate searches over.
Lengths derive from `catalog/ofa_mbv3` (`KS/E/D/MAX_DEPTH/STAGES`), never hardcoded, so CP 7.1 extends
this same file for new op choices. **DoD PASS:** `decode(encode(arch)) == arch` for 100 random archs
(`python -m search.space` → 100/100; `random_arch_dict` is the documented torch-free equivalent of
`supernet.sampler.random_arch`, so the DoD runs in `.venv`). `check.sh` green: **271 passed, 2 skipped**
(ofa/ultralytics → `.venv-nas`), ruff + mypy clean. Commit `1cbe574`.

### The load-bearing design call: lossless bijection vs. canonical encoding

OFA's `sample_active_subnet` fills **all 20** `ks` and **all 20** `e` slots with random values, even the
trailing slots a stage's depth `d` switches off — and `arch_to_blocks` only ever reads `range(d[s])`, so
those inactive slots are **don't-cares** (proven by `test_arch_to_blocks.test_depth_truncation_ignores_
inactive_slots`). Consequence: the DoD's exact-equality round-trip over `random_arch` output **forces a
lossless 45-slot bijection** — `encode`/`decode` must preserve the don't-cares verbatim or any `d<4` arch
fails to round-trip. But that lossless vector is the *wrong* input for the CP 3.3 GP: two archs differing
only in inactive slots are the *same* network (same blocks, same latency, same mAP) yet sit at different
points → **phantom dimensions** that inflate the Hamming distance and waste the surrogate's ≤20-dim
budget. Resolution: keep `encode`/`decode` lossless (no masking) **and** add a separate `canonical()` that
masks inactive ks/e slots to `INACTIVE=-1`, so functionally-identical archs collapse to one point. Masking
lives only in `canonical()` — that is what lets the bijection (DoD) and the surrogate's distance metric
coexist instead of fighting. `AXIS_TYPES`/`AXIS_CARDINALITIES` expose the categorical (ks,e) vs ordinal
(d) split for the CP 3.3 Hamming+Matérn kernel. Pure Python (no torch/numpy); `decode` emits plain `int`
so its output passes `validate_arch_dict` (which rejects `np.int64`/`bool` to keep the LUT `row_key`
stable).

### Phase-3 audit decisions taken entering CP 3.1 (recorded so 3.2/3.3 inherit them)

- **Scope:** Phase 3 is **five** checkpoints (3.1–3.5), not three — 3.4 (TPE/Optuna fallback) and 3.5
  (winner export to `state/winner_v1/`) are part of it.
- **Accuracy signal = cheap NSGA-II + expensive BO** (user decision, *not* the OFA-predictor path).
  CP 3.2 NSGA-II runs on zero-cost `depth_sum` + LUT latency (free / CPU) — a *structural baseline* that
  warm-starts BO, **not** the headline accuracy frontier (its two axes are correlated, so expect a thin
  front that just clears "≥10 non-dominated points"). CP 3.3 BO spends the warm-head 5-epoch proxy on a
  small candidate budget. Do **not** use `latency_ms` as the accuracy axis (monotone-in-latency → Pareto
  front collapses to a line); `depth_sum` is the defensible cheap proxy (ρ≈0.84 vs full mAP, CP 2.4).
- **D2 budget multiplier (the binding Phase-3 GPU cost):** the ≥5-seed protocol makes CP 3.3 cost
  ≈ `5 × (2B − n_init)` warm-head proxy fine-tunes (5 seeds × {random control + BO}, GP seeded from the
  shared random evals). On Colab-only T4 that means `B≈40–50` → ~400–500 evals (feasible); `B=100` →
  ~1000 (likely too much). Size D2 against this; use `eval/zerocost.py` to prefilter before any proxy
  eval. **D2 stays open — bring the chosen `B` to the user.**
- **Still owed, Jetson-gated, not blocking the encoder:** 640-res LUT re-sweep (rows keyed @224; pose
  @640 — fine for *relative* ranking in CP 3.2, needed for the *absolute* `λ·latency` term in CP 3.3) +
  baseline yolo11n-pose anchor. **D4 (λ/μ) stays open → CP 3.3.**

## D2 RESOLVED — Phase-3 search budget B=50 (2026-06-27)

D2 ("Search-budget target") closed in a user conversation. The plan's "100 candidates" default
predated the locked 5-seed statistical protocol and was infeasible under it; the chosen "cheap NSGA-II
+ expensive BO" design also means a single "candidate count" no longer describes the budget — the two
search stages have completely different costs.

### What the budget actually is

- **NSGA-II (CP 3.2) — free.** Scored on `depth_sum` (zero-cost) + LUT latency; both CPU-only (verified:
  `search/cost.py` and `eval/zerocost.py` import no torch). The 100 gen × 50 pop ≈ 5,000 evals cost **$0
  GPU** and are not budget-constrained. (Plan text at PROJECT_PLAN.md:217 corrected from "short-FT
  accuracy" → depth_sum+LUT to match this.)
- **BO (CP 3.3) — the real budget.** Each eval = one warm-head 5-epoch proxy fine-tune on Colab T4. The
  protocol multiplies it: **total = `5 × (2B − n_init)`** (5 seeds × {random-search control + BO}, the
  GP's `n_init` initial design *shared* with the control's evals, counted once).

### The decision: B = 50, n_init = 20

→ `5 × (2·50 − 20)` = **400 warm-head fine-tunes** for CP 3.3 (covers BO **and** its same-budget random
control). Lands in the band procedure.md already flagged "feasible"; B=100 would have been ~900 ("likely
too much"). Estimated ~20–40 GPU-hours, but **no per-eval wall-clock was ever recorded** (CP 2.4 logged
only mAP) — the figure is inferred from config (5 ep × ~178 steps/ep + ~9 val batches @ ~3–6 it/s, T4,
`workers=0` ≈ ~3–6 min/eval). On *free* Colab that is multiples longer in calendar time
(sessions + quotas); resumable per-arch flush (`eval/proxy_rank.save_results`) makes it survivable.

### Fixed knobs (recorded so CP 3.3 inherits them, not re-litigated)

- **1 seed per eval**, not 3. A GP models observation noise natively via its nugget term, so feeding it
  1-seed noisy proxy mAPs is principled; CP 2.4 showed single-eval noise (Δ=0.0145 on the worst arch)
  doesn't reorder ranks. 3-seed averaging would triple cost to denoise what the surrogate already
  handles — reserved for the CP 3.5 winner verification (1 arch, cheap).
- **qEI batch-of-4 = diversification, not parallelism.** On one free T4 the 4 picks evaluate
  *sequentially*; batching only cuts GP refits / near-duplicate picks. It does **not** reduce eval count,
  so it does not change the `5·(2B−n_init)` formula.
- **NSGA-II frontier + `eval/zerocost.py` prefilter warm-start BO's init** (free) so B is spent near the
  frontier, not on blind random draws.
- **M3 "≥50 BO rounds" = ≥50 evals** (round = candidate). The real CP 3.3 gate is the
  hypervolume-dominance test, not a round count.
- **Step 0 of CP 3.3 = one timed calibration eval** on Colab to replace the ~3–6 min estimate before
  spending the 400-eval budget (lever: dataloader `workers` 0→2).

### Scope of the close

D2's **Phase-7** budget (was "200") is deliberately *not* set here — it's re-decided at CP 7.2 against
the same protocol. Recorded in PROJECT_PLAN.md (D2 entry + CP 3.2/3.3), CLAUDE.md (open-decisions table),
and plan_state.yaml. **CP 3.2 (NSGA-II, `search/evolution.py`) is now the next buildable checkpoint —
CPU-only / local / no Colab.** D4 (λ/μ) stays open → CP 3.3.

## CP 3.2 CLOSED — NSGA-II evolutionary baseline (2026-06-27)

`search/evolution.py`: NSGA-II over `(maximize depth_sum, minimize latency_ms)`, producing the Phase-3
Pareto frontier. **CPU-only / local** — reads the fp32 `data/lut.jsonl`, no GPU/Colab/Jetson. **DoD
PASS:** `python -m search.evolution` yields **11 non-dominated points** (≥10), the **true global front**
(every point at min ks/e — see the convergence note below), in ~5 s (~20k unique archs, memoized).
`check.sh`: **280 passed, 2 skipped**, ruff + mypy clean. Commits `c83d22a` (build) + the convergence
follow-up.

### Implementation: pymoo (user decision)

The user chose **pymoo** (the standard library) over a hand-rolled NSGA-II — it reads well in a methods
section, at the cost of a new dependency. `pymoo>=0.6.1` added to `requirements.txt` (CPU `.venv`; pulls
numpy/scipy — already pinned — + matplotlib/autograd/cma) and `pymoo.*` to the mypy
`ignore_missing_imports` overrides. **Verified no venv drift** ([[venv-drift-onnxscript]]): `torch`
stayed `2.3.1+cpu`, numpy/scipy unchanged, all 274 prior tests still green. pymoo is **lazy-imported
inside `run_search`** (mirrors `eval/shortft.py`'s lazy torch), so the module + its pure helpers
(`evaluate_objectives`, `_nondominated_dedup`) import and unit-test in `.venv`/CI without pymoo.

The GA searches the CP 3.1 length-45 **integer category-index** vector (uniform box `[0,2]`, all axes
cardinality 3) via the documented pymoo integer recipe (`IntegerRandomSampling` + `SBX`/`PM` with
`RoundingRepair`, `eliminate_duplicates=True`). `evaluate_objectives` returns `(-depth_sum, latency_ms)`
to minimize; objectives are memoized per genotype. The final frontier is deduped by objective value
(collapsing depth-inactive don't-care twins) and written to a **gitignored** `data/phase3_nsga2_frontier.json`
for the CP 3.3 BO warm-start.

### Result: the analytic depth staircase (as expected)

The front is exactly the **11-point depth staircase** — `depth_sum` 10→20, each at its min-latency config
(`ks`/`e` driven to their smallest), latency rising monotonically **1.73 → 3.26 ms**. This is the
analytic Pareto front of `(depth_sum, latency)`: at a fixed depth, varying `ks/e` only moves latency
(same "accuracy"), so those points are dominated; across depths, more blocks ⇒ strictly more latency ⇒ 11
mutually non-dominated steps. It is **intentionally thin** — the documented structural-baseline role.
`depth_sum`'s ρ≈0.84 vs real mAP makes it a defensible cheap axis, but it can't reward `ks/e`, so the
accuracy-richness comes from the **CP 3.3 BO over the warm-head proxy** (where mAP responds to `ks/e`).
CP 3.2's lasting value is the **reusable NSGA-II machinery**, re-run on the enriched op-space at CP 7.2.

### Convergence: population size, not generations (smoke-test follow-up)

A post-close smoke test caught that the first run (`pop=50, gen=100`) returned a front that was
*self-consistently* non-dominated but **not globally optimal** — only **2/11** points sat at the true
min ks/e, the other 9 ~1.5 % above optimal latency (a faithful approximation, not the Pareto front). A
budget sweep found the lever is **population, not generations**: `gen` 100→300 at `pop=50` changed
nothing (2/11), but `pop` 50→100→150 converged 2→7→**11/11**, robust across 5 seeds, by `gen=200`. Cause:
a 50-individual pool is too small to *hold* the all-min-ks/e config at every depth, so selection can't
fix what mutation rarely generates; a larger pool does. **Defaults bumped to `pop_size=150, n_gen=200`**
(~5 s, still trivial/CPU) and locked by `test_default_budget_reaches_true_pareto_front` (asserts every
frontier point is at min ks/e). The lesson carries to CP 7.2: size the population to the space, don't
just add generations.

### Tests (`tests/test_evolution.py`, TDD)

Pure (always run, no pymoo/LUT): `_nondominated_dedup` skyline + dedup, cross-checked against
`search.cost_preview.nondominated_indices`. LUT-only (no pymoo): `evaluate_objectives` depth-sign +
latency monotonicity. pymoo+LUT (`importorskip` + `lut_path` fixture + `slow`): reduced run
(`pop=40,gen=40`) → ≥10 non-dominated points, frontier internally non-dominated; seed reproducibility;
and the full-budget true-front check (every frontier point at min ks/e). `CostError`→skip guards keep
them green on a partial LUT. Next: **CP 3.3 BO**
(`search/bo.py`) — needs the D4 **numbers** (λ/μ, calibrated here — the formulation is now fixed, below)
+ the warm-head proxy budget (B=50) + the Jetson 640-res LUT/baseline for the absolute objective.

---

## D4 RESOLVED — J(α) = Pareto search + hard latency ceiling (2026-06-27)

D4 (the λ, μ in `J(α) = acc − λ·latency − μ·max(0, mem−budget)²`) was the last open decision blocking
CP 3.3. Settled by AskUserQuestion (full briefing → user choice), mirroring D1/D2. The user also asked
explicitly for "a method to select a maximum latency" — answered by the **ε-constraint hard ceiling**
(OFA's own "best accuracy under a latency budget" method), now part of the resolution.

### Three findings that reshaped the choice
1. **The memory term never binds in v1.** OFA-MBv3-w1.0 subnets are ≤24 MiB fp32 (tens of MiB fp16) vs
   the 8 GB device, so `μ·max(0, mem−budget)²` is identically 0 for every v1 subnet — it only matters
   after Phase-5 expansion. The μ/budget half of D4 is a near-non-decision for v1.
2. **The numeric λ can't be honestly pinned yet — and needn't be.** λ has units of mAP-per-ms,
   meaningless without the *deploy* (@640) latency scale. That scale is owed/Jetson-gated (the 640-res
   LUT re-sweep + the yolo11n-pose baseline anchor — no measured baseline latency exists yet). The
   NSGA-II frontier numbers (1.73→3.26 ms) are backbone-only @224, not what λ multiplies. So we fix the
   *method* now; the *number* lands at CP 3.3.
3. **CP 3.3's DoD is already Pareto hypervolume** (PROJECT_PLAN.md), explicitly *"not a single-run
   accuracy − λ·latency comparison"* — so λ is a sampled / selection knob, not a fixed search constant.

### The decision (user-selected)
- **Objective form = Pareto + hard latency ceiling.** CP 3.3 runs *multi-objective* BO over
  `(acc_eff, latency_ms)` bounded by `latency ≤ T_max`. The soft μ² penalty is **retained** (user
  choice) and folded into the accuracy axis — `acc_eff = acc − μ·max(0, resident_mem_mib − budget)²` —
  so the front stays 2-D while honouring the penalty (≡ acc for all v1 subnets). The scalar
  `J = acc_eff − λ·latency` is both the **ParEGO random-weight scalarization** (traces the front;
  reconciles the EI acquisition with the hypervolume DoD) and the **final-winner selector** (CP 3.5).
- **λ — sampled during search, calibrated at selection.** ParEGO samples the weight while searching (no
  fixed λ committed up front). The deploy winner is picked by calibrating λ from two reference models on
  a common iso-J contour (MobileNetV3-large vs EfficientNet-B0: `λ = Δacc/Δlat`), reported as a
  **sensitivity sweep**, not one magic value.
- **Memory — soft μ² retained, budget = 512 MiB resident (fp16).** A conservative model reservation on
  the shared 8 GB; μ calibrated with λ at CP 3.3. Keeping it (vs a hard filter) preserves one uniform
  `J(α)` across Phase 3 and Phase 7.

### Maximum latency T_max (the user's explicit second question)
`T_max = min(baseline_latency, fps_cap)` — the tighter of two anchors (user chose "both"):
- **baseline** = measured yolo11n-pose latency @640, FP16 TRT, Orin Nano — the literal "dominate the
  deployed baseline" bar (Jetson-gated, owed).
- **fps_cap** = the perception-node frame budget; provisional **60 FPS → 16.7 ms** (`fps_to_ms`),
  decidable now without the Jetson (confirm/adjust the FPS target).

The ceiling is a hard box constraint on the search — interpretable, and it stops the 50-eval budget
chasing accurate-but-slow models that can't dominate the baseline.

### Built — `search/objective.py` (pure, CPU, TDD; commit 335c4c4)
Locks the formula as a tested contract CP 3.3 just calls: `mem_penalty`, `effective_accuracy`,
`scalarize` (the scalar J), `within_ceiling` + `fps_to_ms` (the hard ceiling), `lambda_from_anchors`
(two-anchor iso-J λ — signed slope; raises on equal latency). λ/μ stay caller args (no deferred number
hard-coded); `DEFAULT_BUDGET_MIB = 512.0`. 13 tests (`tests/test_objective.py`); check.sh green (293
passed). This is decision-recording + a formula lock, **not** CP 3.3: `search/bo.py` stays gated on the
@640 sweep, the baseline anchor, and the timed Colab calibration eval.

## CP 3.3 — buildable slice BUILT (2026-06-28)

The whole CPU-buildable half of CP 3.3 plus the two remote-run artifacts that produce its numbers.
**CP 3.3 stays OPEN** — its DoD (5-seed Pareto hypervolume beating same-budget random search on the
*real* warm-head proxy) closes only after the Jetson @640 latencies + the Kaggle GPU runs land.
`current_checkpoint`/`last_completed`/`completed` are unchanged (still 3.3 / 3.2). User chose, against
the recommendations (the [[decision-briefing-then-choose]] pattern): **BoTorch+GPyTorch** for the
surrogate, and **git-clone + a data-only Kaggle Dataset** for delivery.

### @640 LUT re-key — the sanctioned count-pin bump (the decision test_catalog points to)
The pose backbone deploys at 640, not the OFA ImageNet 224, so every per-block input resolution
re-keys (`catalog.ofa_mbv3.stages_for_resolution`: stem 320; stage res_in `[112,56,28,14,14] →
[320,160,80,40,40]`; taps 80/40/20, confirmed by `supernet/pose_backbone.py`). Made the catalog
resolution-aware and **unioned the @640-reachable MBConv configs into the grid** (`catalog/blocks.py`),
threading `res:int=224` through `search.arch_to_blocks`/`search.cost` (640 for pose; 224 default
preserves CP 3.1/3.2). **Append-only**: the @640 res values `{320,160,80,40,20}` are disjoint from @224,
so the 91 new configs add 91 new `row_key`s; every measured @224 row + the golden hashes in
`tests/test_row_key.py` are untouched. The deliberate count-pin moves (per `test_catalog.py`'s own
"conscious act" rule): `sweep_size` **2710 → 2801**, `mbconv` grid **2107 → 2198**;
`test_lut_keydrift` correctly flips to SKIP at 2710/2801 until the @640 sweep fills the rows. TDD:
`tests/test_resolution.py` (8) + @640 cases in `test_arch_to_blocks`/`test_cost`. Commit `988e543`.

### `search/bo.py` — the BO loop (commits, BoTorch)
Split like `search/evolution.py`: pure numpy/stdlib helpers (unit-tested in `.venv`/CI without botorch
or a GPU) + a lazy-imported driver. **Pure** (`tests/test_bo.py`, commit `458a993`): `parego_weights`
(uniform-simplex), `tchebycheff_scalarize` (augmented Tchebycheff — recovers concave front regions),
`nondominated_indices` + `hypervolume_2d` + `pareto_hypervolume` (the DoD metric over `(acc_eff↑,
latency↓)`), `feasible`/`mutate_arch`/`candidate_pool` (discrete candidates under the hard ceiling,
canonical-deduped), `bo_verdict` (dominance-across-seeds: BO HV band entirely above random's).
**Driver** (`run_bo`, commit `331741e`): **classic ParEGO with BoTorch as the GP+EI engine** — each step
draws a random simplex weight, re-scalarizes the observed objectives (observed accuracy + the *exact*
LUT latency) via augmented Tchebycheff, fits a `MixedSingleTaskGP` (CategoricalKernel≈Hamming on the 40
ks/e dims, Matérn on the 5 ordinal depths) to the scalar values, and maximizes `qLogEI` over the
feasible pool. Latency is deterministic, so only accuracy is GP-modeled and the ceiling pre-filters.
Resumable (JSONL cache, skips done evals). CLI: `--structural` (no-GPU depth_sum smoke), `--calibrate N`
(per-eval wall-clock + 5-seed GPU-h estimate), real (`--device cuda --head-weights <gate best.pt>
--freeze-head --imgsz 640`); warm-starts from the CP 3.2 NSGA-II frontier. **CPU structural smoke @224
(t_max 2.5 ms binding): BO HV 9.69±0.06 vs random 3.66±0.35 → DoD PASS over 3 seeds.** Surrogate stack
`botorch>=0.11`/`gpytorch>=1.12` added to `requirements.txt` under the `torch==2.3.1+cpu` pin (that exact
pin is the constraint; tested botorch 0.17.2 / gpytorch 1.15.2 — torch/numpy unchanged,
[[venv-drift-onnxscript]]). 19 tests (+2 botorch-gated integration: run + resume).

### Jetson + Kaggle artifacts (the owed-numbers producers)
**Jetson** (commit `e05a86e`, `lut/orchestrate/bench_model.py` + `detect/export_baseline_onnx.py` +
`lut/docs/jetson_640_runbook.md`): export yolo11n-pose → static ONNX @640, then benchmark any whole
model on-device by reusing `run_sweep.run_remote_bench` verbatim → `data/baseline_anchor.json` (NOT a LUT
row). Sets `T_max = min(baseline, 16.7 ms)`. Precision defaults to `sweep.precision` (fp32) so the
ceiling is like-for-like with the fp32 LUT (the fp16 deploy figure is a separate Phase-8/9 number). The
runbook ties setup → idempotent @640 re-sweep (`run_sweep` skips the 2710 @224 rows, measures the 91 @640)
→ baseline → teardown. **Kaggle** (commit `02c194d`, `kaggle/`): a script kernel (`run.py`) clones the public repo, pins
Kaggle's torch via a constraint, wires a data-only Kaggle Dataset (dataset/ + LUT + NSGA-II seeds + frozen
gate head), re-downloads the SHA-pinned OFA ckpt, and runs `--calibrate` then the search; `push.sh`
automates dataset create/version (hardlink-staged, no 1.6 GB copy) + kernel push, token from the gitignored
`secrets/access_token` (new-style `KGAT_`; username in `secrets/kaggle_username` — legacy `secrets/kaggle.json`
still works). OFA ckpt is never uploaded (re-fetched in-kernel).

### Still owed to CLOSE CP 3.3 (unchanged gates, now runnable)
1. Jetson **@640 LUT re-sweep** + **yolo11n-pose baseline** (run the runbook).
2. Kaggle **5-seed warm-head BO + random control** → `cp33_bo.json` verdict (the real DoD).
3. The λ/μ **numbers** (need the @640 baseline scale; calibrated at selection via the iso-J anchors).
`check.sh` green throughout (324 passed, 3 skipped). See CLAUDE.md "Current state".

## CP 3.3 CLOSED — warm-head BO-vs-random DoD PASS (2026-07-02)

The DoD landed: the 5-seed warm-head Bayesian-Optimization Pareto **hypervolume beats the
same-budget random-search control**, decisively and on every seed. `current_checkpoint`
3.3→3.4, `last_completed` 3.2→3.3, `completed += "3.3"`. Verdict in
`data/cp33_kaggle_out/cp33_bo.json` (`passes=true`, `complete=true`, `res=640`,
`t_max_ms=12.75`, `budget=50`, `n_seeds=5`):

| seed | BO HV | random HV | BO/RS |
|---|---|---|---|
| 0 | 3.482 | 2.392 | 1.46× |
| 1 | 3.430 | 1.956 | 1.75× |
| 2 | 3.435 | 1.867 | 1.84× |
| 3 | 3.438 | 1.935 | 1.78× |
| 4 | 3.418 | 2.287 | 1.49× |
| **mean** | **3.441 ± 0.022** | **2.088 ± 0.211** | **1.65×** |

BO wins on every seed (never a tie/loss) and is ~10× more consistent (std 0.022 vs 0.211) — the
surrogate reliably steers to the good region while random swings on luck; the BO mean sits
~6.4 random-σ above random's, far outside the noise. The campaign ran **across both backends**
(Kaggle @640 seeds 0–1, then the AGX Jetson Orin for seeds 2–4 when the weekly Kaggle quota ran
out mid-run — identical cache format, machine-agnostic verdict; see `CP33_BACKENDS.md`). The
union BO frontier = 12 non-dominated points/seed, **all feasible** (6.85–11.74 ms under the
12.75 ms ceiling), proxy-acc 0.475→0.650; the top point is depth-15, *not* the deepest — the
search exploits `ks`/`e`, not just depth.

**What the DoD certifies — scope.** This is a *search-method* gate (BO ≫ random at navigating
the space), **not** the thesis headline. The frontier `acc` values are 5-epoch **warm-head
proxy** mAPs (a ranking signal; the CP 2.4 reframe), while the deployed-yolo11n-pose baseline's
**0.877** (`data/baseline_anchor_map.json`, full-train) is *not* comparable to them. Whether a
found arch Pareto-dominates yolo11n-pose in *deployable* accuracy is a CP 3.5 (winner export) +
Phase-8 (distill) question — answered by full-training the selected α* and measuring its real
mAP at its measured Orin-Nano latency, not by this checkpoint.

**λ/μ status (the owed "numbers").** `acc_eff == acc` at every frontier point → the μ² memory
penalty never binds (every subnet fits the 512 MiB fp16 budget) → **μ is moot for v1**. **λ is
deferred to CP 3.5**: it does not enter the hypervolume DoD (ParEGO samples λ internally; the
verdict is λ-free), only the single-winner *selection*. User chose the **two-anchor iso-J**
method (`search/objective.py:lambda_from_anchors`): the second reference (a bigger yolo11-pose
@640) is measured at CP 3.5 to set the accuracy/ms exchange rate, then α* = argmax
`scalarize(...)` over the union frontier. Recording the *method* — not a fabricated number — is
the honest close.

**Both baseline anchor coordinates are on disk:** `data/baseline_anchor.json` (latency 12.755 ±
0.012 ms, n=200, fp32/TRT-10.3, MAXN, clocks locked, JP7 R39.2) + `data/baseline_anchor_map.json`
(mAP 0.8774). **Next = CP 3.4 (TPE fallback, Optuna)** — same dominance test, reusing the
method-agnostic `search/bo.py` machinery + the 302-arch acc-memo + the cached CP 3.3 random
control (a much lighter GPU pass).

## CP 3.5 refinement — ceiling-first winner; two-anchor λ demoted to a robustness check (2026-07-02)

**Not a checkpoint close; a D4 method refinement (user-approved via AskUserQuestion).** While
staging CP 3.5 in parallel with CP 3.4, the user challenged the two-anchor λ:
`λ = (acc_A − acc_B)/(lat_A − lat_B)` **is a secant — it assumes the accuracy/latency trade is
linear between two off-the-shelf models**. Correct, and it cuts deeper: yolo11n/yolo11s aren't even
points *on* our search frontier (our archs may dominate both), so the chord between them is a
questionable exchange-rate source, and a true λ is a *local derivative* (tangent of the frontier),
not a wide secant across a 70 %-latency gap.

**Resolution — "ceiling-first, λ as check" (AskUserQuestion, option A of 3; alternatives were
local-frontier-slope λ and a λ-free knee-point).** The hard latency ceiling (`T_max = 12.75 ms`,
D4) is the real decision rule; among feasible archs more accuracy is strictly better, so:

- **Headline pick is λ-free:** α* = the most-accurate frontier point under `T_max`
  (`search/select_winner.ceiling_first_winner`, tie-break → lower latency). No linearity
  assumption enters the thesis.
- **The two-anchor λ survives only as a robustness check** (`winner_is_lambda_stable`): does the
  λ-scalar argmax-J agree with the ceiling-first winner across a whole log-λ grid? A fully-agreeing
  grid *proves* the latency term never flips the pick — the quantitative, assumption-free substitute
  for trusting one λ. On this saturated gate task λ ≈ 0.001–0.002 acc/ms (≪ the frontier's own
  ~0.03 slope), so agreement is expected; any flip is reported as the exact λ where it would matter.

**Consequence:** α* needs **neither anchor**, so the winner is fully determined *before* anchor B's
gate fine-tune finishes. Anchor B (a bigger yolo11-pose) drops from selection-critical to (i) the
robustness check and (ii) a Phase-8 distillation-teacher scout — which is why the anchor-B CPU
fine-tune can be stopped early with no effect on α*. Code: `search/select_winner.py` +
`tests/test_select_winner.py` (19 tests, TDD), commit `a1325e1`; `winner_record` now leads with
`selection_rule` and treats λ / anchor B / sweep / `robustness_check` as optional (null pre-anchor-B).

**Anchor-B latency curve on disk** (@640, fp32/TRT-10.3, 612 MHz mode 0, n=200; the slow/accurate
end of the line, all ≫ anchor A's 12.755 ms): yolo11s 21.69 ms / 43.6 MiB, yolo11m 43.43 ms /
79.2 MiB, yolo11l 55.79 ms / 81.5 MiB (`data/anchor_yolo11{s,m,l}_pose_640.json`). Anchor-B
*accuracy* (yolo11s gate fine-tune, CPU) is the only remaining input, and only for the check.

## CP 3.4 CLOSED — TPE fallback reproduces BO; the warm-start (not the acquisition) drives it (2026-07-04)

**DoD met — literal scope** (`PROJECT_PLAN.md:246`, "same dominance test as CP 3.3"; the
*interpretation* is corrected in Finding 1 — the pass is the warm-start, not the Bayesian
layer). `search/tpe.py`
(Optuna MOTPE) re-ran the warm-head BO-vs-random hypervolume test @640, reusing the
method-agnostic `search/bo.py` machinery (`pareto_hypervolume` / `feasible` /
`random_search_control`), the 302-arch acc-memo, and the **cached CP 3.3 random control**.
Verdict (`data/cp33_kaggle_out/cp34_tpe.json`, `passes:true`, `complete:true`, 5 seeds,
budget 50, res 640, `T_max=12.75 ms`):

| Metric | TPE (CP 3.4) | BO (CP 3.3) | Random (shared control) |
|---|---|---|---|
| Hypervolume | 3.414 ± 0.023 | 3.441 ± 0.022 | 2.088 ± 0.211 |
| vs random | 1.64× | 1.65× | — |
| Seeds won | 5/5 (1.42–1.84×) | 5/5 | — |

Provenance: the 5-seed run completed (`cp34_tpe.part0.json` already carried all 5 seeds
`complete:true`); the multi-backend Kaggle→Colab resume merged clean into the authoritative
`cp34_tpe.json`.

**Finding 1 — the win is the NSGA-II warm-start, NOT the Bayesian acquisition (corrected
2026-07-04).** *An earlier draft of this entry claimed TPE≈BO proved "the search is guided,
not a BoTorch quirk." That is wrong, caught by the question "couldn't TPE≈BO just be the
shared NSGA-II pre-run?"* Both `run_bo` and `run_tpe` seed their initial design from the
**same** CP 3.2 NSGA-II frontier (`search/tpe.py:139`: "BO uses these as seeds too";
`search/bo.py:516`), while `random_search_control` (`search/bo.py:571`) gets **no** seeds.
So the DoD compares warm-started search against a **cold** control, and TPE≈BO is largely
forced by the shared seeds. A free ablation — rebuild the control **with** the warm-start
from the cached `*.rs.jsonl` evals + `data/phase3_nsga2_frontier.json`, common ref
(0, 12.75 ms), 5 seeds (HV method reproduces every stored per-seed HV exactly) — decomposes it:

| Configuration | Hypervolume | isolates |
|---|---|---|
| cold random (the DoD control) | 2.088 ± 0.211 | no warm-start, no guidance |
| NSGA-II seeds alone (11 pts) | 3.357 | the warm-start only |
| **warm random** = 11 seeds + 39 *random* (matched 50) | **3.403 ± 0.013** | warm-start + dumb fill |
| TPE = 11 seeds + 39 TPE-picked | 3.414 ± 0.023 | warm-start + tree-Parzen |
| BO = ~20 init + 30 BO-picked | 3.441 ± 0.022 | warm-start + GP/qLogEI |

Of the 1.353 BO−cold-random gap, the **warm-start is +1.315 (97 %)** and the **acquisition
is BO +0.038 / TPE +0.011 (1–3 %)** over a *budget-matched warm-random* control. Under the
DoD's own band rule (`mean−std > mean+std`), TPE (lower band 3.391) does **not** clear
warm-random (upper band 3.416); BO clears it by 0.003 (noise). The "~10× tighter std" is
also just the fixed seed set — warm-random's std (0.013) is *tighter* than either optimizer.
34–38 % of each "converged" frontier is literally unchanged NSGA-II seeds (though α*,
d=[2,2,4,4,3], is a genuine BO discovery, not a seed).

**Threat to validity (recorded, not re-run — the ablation above IS the fair-control result).**
The CP 3.3/3.4 DoD control is **cold** random, so "search ≫ random" conflates the structural
warm-start with the Bayesian layer. The **fair** control is warm-started random, and it ≈
BO/TPE. So CP 3.3/3.4 honestly certify only that *the warm-started pipeline (NSGA-II → BO/TPE)
beats cold random* and that *TPE is a valid drop-in for BO* — they do **not** show the
acquisition function is the driver; on this task it isn't. The checkpoints stay CLOSED under
that corrected, narrower scope. *Why* the acquisition has no room is **Finding 2**: a near-flat
frontier is already traced by a good structural depth-staircase, so BO/TPE find almost nothing
NSGA-II's spread didn't already cover — the warm-start and saturation findings are one story.

**Finding 2 — the gate task is accuracy-saturated (anchor B landed).** yolo11s-pose
full-train mAP **0.8819** @ 21.69 ms vs yolo11n-pose 0.8774 @ 12.75 ms: **+70 % latency
buys +0.5 % mAP** → two-anchor λ ≈ **0.0005 acc/ms**, an order of magnitude under the search
frontier's own ~0.03 slope. This vindicates the CP 3.5 ceiling-first refinement: when
accuracy saturates, the fastest arch on the plateau wins, and the `λ·latency` term cannot
flip the pick. (Heads-up for Phase-8 teacher choice: a bigger teacher may offer little
accuracy headroom on this task; yolo11m/l mAP is still unmeasured.)

**Finding 3 — the λ-robustness check now runs and passes.** With anchor B's accuracy on
disk, `search.select_winner` over the BO∪TPE union (130/130 feasible under `T_max=12.75 ms`)
picks α* = `[bo, seed 0]`: proxy acc **0.650**, latency **11.744 ms** (< 12.75 → faster
than yolo11n), d=[2,2,4,4,3]. `winner_is_lambda_stable`: **stable=True, agree 1.00** across
the 7-point log-λ grid (0.00025…0.0010 acc/ms) — the ceiling-first winner is J-optimal at
every λ, so the linearising two-anchor secant never changes the decision.

**Scope caveat (carries to CP 3.5 / Phase 8).** Frontier accs (0.47–0.65) are the **5-epoch
warm-head PROXY mAPs** (the CP 2.4 ranking signal), NOT comparable to the full-train 0.877
baseline. α*'s *faster-than-yolo11n* is real (LUT-exact latency); the *deployable-accuracy*
dominance claim is Phase 8. **Next = CP 3.5** (winner-v1 export; DoD = reload α* in a clean
session and reproduce its cached proxy acc within noise → needs a Colab fine-tune).

## CP 3.5 CLOSED — winner-v1 = the de-noised knee; the reproduce-DoD caught (and corrected) a single-seed winner's curse (2026-07-04)

**Phase 3 closes.** `current_checkpoint` 3.5→4.1, `last_completed` 3.4→3.5, `completed += "3.5"`.
The winner-v1 export (`state/winner_v1/winner.json`) is **`d=[2,2,4,3,3]` @ 11.208 ms**, de-noised
warm-head proxy mAP **0.6101 ± 0.0049**, **12.1 % faster than yolo11n-pose** (12.755 ms) — *not*
the ceiling-first α* that CP 3.3/3.4 predicted. The reproduce-within-noise DoD did its job: it
falsified the naive winner and forced an honest re-selection.

**The DoD ran and FAILED for α*** (`eval/verify_winner.py`, Kaggle T4, 3 **fresh** seeds 1/2/3,
`data/cp33_kaggle_out/repro.json`). Reloading α* = `[bo,seed0]` `d=[2,2,4,4,3]` in a clean session
and re-deriving its warm-head proxy mAP gave fresh seeds `[0.575, 0.634, 0.621]`, **mean 0.610 vs
cached 0.650 → Δ = −0.040** (band 0.020) → **passes=false**. Not a pipeline bug — a **selection**
bug.

**Root cause — the single-seed winner's curse.** The search oracle scored *every* frontier arch at
a single fine-tune seed (`seed=0`); the top of the feasible frontier is a **statistical tie** (top-12
cached span 0.027 < a single arch's fresh-seed σ ≈ 0.031). The ceiling-first `argmax` over 130
single-seed draws therefore selects whichever arch's seed-0 draw was *luckiest* — an upward-biased
estimator that regresses hard on re-eval. α*'s own σ (0.025, the largest in the set) was the tell.

**The fix — de-noise the contenders, then re-select** (`search/denoise.py`, Kaggle
`kaggle/run.py MODE="denoise"`, 12 archs × 3 seeds = 36 warm-head fine-tunes, resumable per-(arch,seed);
result `data/cp33_kaggle_out/denoise.json`). Re-scoring the top-12 feasible frontier at fresh seeds and
averaging **scrambled the ranking completely** — every arch regressed:

| arch (depth) | cached (seed-0) | de-noised mean ± σ | Δ | de-noised rank |
|---|---|---|---|---|
| `[2,2,4,4,2]` (tpe) | 0.638 | 0.624 ± 0.003 | −0.014 | 1st |
| **`[2,2,4,3,3]` (bo) — winner** | 0.624 | **0.610 ± 0.005** | −0.014 | 5th |
| `[2,2,4,4,3]` — old α* | 0.650 | 0.610 ± 0.025 | **−0.040** | 4th |
| `[2,4,3,4,4]` — fastest-cached | 0.634 | **0.571 ± 0.002** | −0.063 | **12th (last)** |

**Finding — the de-noise averted a *second* curse.** The fastest-cached arch `[2,4,3,4,4]` @ 10.15 ms
(cached 0.634) collapsed to **dead last** (0.571) on de-noising — reliably the *worst* arch (σ=0.002),
its 0.634 pure seed-0 luck. A naive "re-pick the fastest of the cached scores" would have walked
straight into a second winner's curse. Averaging first is what caught it. **Lesson recorded**: low σ
means *reliably whatever it is*, not *good*; argmax-over-noisy-estimates systematically selects for
upward error, so the headline winner is disproportionately the one whose noise pointed up.

**The honest frontier is shallow, so the pick is a real trade.** Averaged, the feasible set is *not* a
flat plateau: near-ceiling archs (~12.6 ms) reach ~0.624, the fastest (10.7 ms) ~0.606 — **~0.018
proxy-mAP for ~2 ms**. Which winner is "honest" then hinges on the `select_denoised` tie-band, and three
*principled* band choices give three different winners:

| tie-band basis | winner | de-noised mAP | latency | vs yolo11n |
|---|---|---|---|---|
| top arch's own σ (0.003, strict) | `[2,2,4,4,2]` | 0.624 ± 0.003 | 12.65 ms | 0.8 % faster |
| **typical σ (~0.013) — chosen** | **`[2,2,4,3,3]`** | **0.610 ± 0.005** | **11.21 ms** | **12 % faster** |
| max σ (0.025, α*'s outlier) | `[2,2,4,3,2]` | 0.606 ± 0.012 | 10.73 ms | 16 % faster |

**Winner selection (user decision, AskUserQuestion).** The **knee** `[2,2,4,3,3]` was chosen over the
accuracy-first and latency-first extremes because it is the best *all-rounder* on a proxy, saturated
task: (1) a strong **12 %-faster-than-yolo11n** latency claim (LUT-exact, real), vs the accuracy-first
pick's thin 0.8 %; (2) the **tightest-but-one σ (0.005)**, so it is the most *reproducible* winner — it
clears the reproduce DoD **directly**: `|cached 0.6238 − mean 0.6101| = 0.0137 < 0.020`, where α* failed
at −0.040; (3) only ~0.014 proxy-mAP below the saturated top, a gap that (per anchor B: +70 % latency →
+0.5 % mAP) is expected to wash out under Phase-8 distillation. The latency-first pick was rejected
precisely because it needs α*'s *outlier* σ as its tie-band **and** has the noisiest σ of the three (the
weakest reproduction). Not a λ decision — λ≈0.0005 acc/ms is an order of magnitude too small to move any
of this; the choice is a documented engineering trade among de-noised means.

**DoD honesty — how the reproduce clause is satisfied.** winner-v1's reference accuracy is now the
**3-seed de-noised mean (0.610)**, not a single seed. The de-noise run *is* the "reload in a clean
session and reproduce" evidence — three independent clean-session warm-head fine-tunes — and the
single-seed search value lands within band of their mean. `winner.json` records the full provenance: the
de-noised maps/σ, the rejected single-seed α*, the averted-second-curse arch, the `vs_yolo11n` speedup,
and a `reproduction` verdict (`passes=true`).

**Code (all CPU, `.venv`/CI-tested).** `search/denoise.py`: `top_candidates` (pinned top-K →
`state/winner_v1/denoise_candidates.json`), `denoise_archs` (GPU, resumable — the only GPU step),
`select_denoised` (fastest-within-tie), `denoised_winner_record` + `--serialize` CLI (writes
`winner.json` from `denoise.json`, round-trip-guards the encode vector). `eval/verify_winner.py`: the
α* verifier that caught the curse (`ReproVerdict`, mean-of-3 band rule). +19 tests
(`tests/test_denoise.py`, `tests/test_verify_winner.py`); `check.sh` fast lane green.

**Deferred (non-blocking).** The knee's concrete *proxy* `weights.pt` was not persisted (the de-noise
fine-tunes weren't saved). It is regenerable from (arch + frozen gate head + seed) — the passing
reproduce-DoD *is* the proof that regeneration is deterministic within noise — and it is **superseded by
Phase 8**, which distills the *deployable* weights from the arch and discards proxy weights. A concrete
`weights.pt` can be dropped in later via a 1-seed `eval.verify_winner --save-weights` run if a tangible
artifact is wanted.

**Method scope (unchanged, carries to Phase 8).** winner-v1's accuracy is still the **warm-head PROXY**
(0.610), NOT comparable to yolo11n's full-train 0.877 — the *latency* dominance (12 % faster) is real
and LUT-exact; the *accuracy* dominance is Phase 8's to earn via distillation. **Phase 3 is complete.
Next = CP 4.1** (Net2Wider — `net2net/wider.py`, function-preserving widen, unit-test to 1e-5).

## Plan pivot — Phases 5–7 re-scoped to winner refinement; D3 RESOLVED → descoped (2026-07-05)

**Not a checkpoint** (`current_checkpoint` stays 4.1; `completed` unchanged). A structural plan
decision taken with the user at the Phase-3 → Phase-4 boundary, triggered by a full audit of the
completed work plus the question "are the expansion phases still the right next step, or should we
pivot to Net2Net head↔backbone compatibility / skip connections / pruning?".

**The audit (what prompted this).** Two exploration passes over state/procedure/artifacts/code plus
targeted verification re-reads found Phases 0–3 methodologically sound (measured LUT + additivity
DoD; the CP 2.4 proxy failure → root-cause → repair → re-gate; the honest CP 3.4 warm-random
correction; the CP 3.5 winner's-curse de-noise) and **two material findings**:

1. **The "12 % faster than yolo11n" headline is not yet an end-to-end claim.** Winner-v1's
   11.208 ms is the backbone-blocks-only LUT sum — every search call site passes `cost(arch, lut,
   res=640)` with no `stem_head` offset and no calibration — while the baseline's 12.755 ms is a
   full-network TensorRT measurement (backbone + PAN neck + pose head). `data/stem_head_offset.json`
   (0.388 ms) is the OFA *classifier* stem+head at res 224, not the pose stem / ChannelAdapter /
   Pose head at 640, which have **never been measured**. Two known corrections pull in opposite
   directions — the CP 2.2 calibration says the raw sum *over*-predicts the backbone ~7 % (TRT
   cross-seam fusion, slope 0.934, fitted @224), while the missing adapter+head is plausibly
   1.5–3 ms — so the true margin is UNKNOWN and could be negative. Search *ranking* is unaffected
   (the offset is arch-invariant; `search/cost.py` documents exactly this), but the absolute claim,
   the T_max feasibility margin, and the thesis headline all rest on one unmeasured number.
   **Consequence → Stage 0, owed before any further headline use:** export the grafted winner +
   a backbone-only ONNX (`detect/export_grafted_onnx.py`, to be built — `head.export=True`,
   opset 17, static shapes), bench end-to-end on the Nano (`lut/orchestrate/bench_model.py`,
   mode 0), derive `data/pose_stem_head_offset.json` (= e2e − backbone; the stem rides inside the
   backbone measurement — `arch_to_blocks` excludes `first_conv`, `PoseBackbone` includes it),
   additively re-stamp `state/winner_v1/winner.json` with an `e2e` block + the honest speedup, and
   bench the two de-noised fallbacks (`[2,2,4,3,2]` @10.73 sum, `[2,4,3,4,4]` @10.15 sum) in the
   same session so a documented re-pick is data-ready if the margin collapses.

2. **Accuracy dominance is still unearned** — known and flagged everywhere, but it *shapes* the
   pivot: proxy 0.610; the e2bfc17 side experiment's full fine-tune reached **0.841 vs the
   baseline's 0.877**, single-seed AND bare-AdamW vs the anchors' full Ultralytics recipe
   (confounded). Meanwhile the graft is **neck-less** — three independently *random-initialized*
   1×1 convs (never warm-started) bridge (40,112,160)→(64,128,256) straight into the Pose head,
   with no cross-scale fusion anywhere — while the baseline carries a full PAN-FPN. CP 2.4 already
   proved the head interface dominates the proxy signal (random head τ=0.20 → warm+frozen head
   ρ=0.77). That seam is the accuracy lever the refinement phases now attack.

Minor audit items: (a) three docs call the measurement regime "MAXN" — it is **mode 0 / 15 W /
612 MHz locked** (on the Orin Nano *Super*, real MAXN is 25 W / 918 MHz and was never measured;
the measured 62.5 GB/s DRAM confirms the non-Super regime). All latency artifacts are consistent
612 MHz numbers, so every *relative* claim stands; wording fix lands with Stage 0. (b) The CP 3.5
winner depends on a hand-picked tie-band (~0.015) — documented at the close; the new
`search/denoise_report.py` (this pass) makes the sensitivity reproducible on demand:
argmax / strict / typical / loose bands pick four different archs on the real `denoise.json`, and
`--tie-band 0.015` reproduces the committed knee exactly. (c) Hygiene: stray 0-byte `a.out`
removed; root scratch `evaluate_denoised.py` homed into `search/denoise_report.py` (+4 tests);
`tests/test_runner_consistency.py` ast-pins RES=640 / T_MAX_MS=12.75 across the three runner kits
(the triplication is deliberate — standalone remotes + RES-namespaced caches — the gate makes it
safe); `net2net` + `expand` added to mypy coverage.

**Why expansion (old Phases 5–7) lost.** The user's initial worry — "expansion would increase
latency" — is the one failure mode the old plan already guarded against (options-not-capacity;
CP 5.0 LUT screen; the T_max ceiling; "Net2Net never independently grows a model's latency"). The
real reasons are: (1) **pre-pivot DoDs** — the old Phase 5/6 gates are ImageNet-framed ("OFA's
published w=1.4 number within 2 %", forwards at 224², "day-0 accuracy of the expanded supernet");
honest expansion post-D1 means rewriting both phases AND re-validating CP 2.4-style proxy fidelity
for every injected block family. (2) **Compute** — the plan's own words: "fine-tune budget
(Phase 6) is the bottleneck, not supernet size" and "OFA's full PS is 180+ GPU-days. We don't have
that" (original OFA supernet training ≈ 1,200 V100-hours; free-tier Kaggle/Colab + a single AGX
cannot amortize even a "light" version, and quota exhaustion already interrupted CP 3.3 once).
(3) **Decisive — oracle saturation:** the warm-head proxy's per-arch σ (0.005–0.025) exceeds the
de-noised frontier's top-cluster gaps (~0.014), and anchor B says the *task* saturates (+70 %
latency → +0.5 mAP full-train). A richer space adds hypotheses the evaluator cannot distinguish —
CP 3.5's winner's curse was precisely this saturation biting. D3's own recorded rule of thumb
("run Phase 3 first, look where α* lands, don't inject speculatively"), evaluated on the Phase-3
outcome, recommends against paying Phase 6. **D3 → RESOLVED (descoped)**; FusedMBConv survives as
the optional evidence-only CP 5.0 LUT screen (Pareto report, no training).

**The decision (user, AskUserQuestion, four answers).** Direction = **pivot to winner refinement**
(the recommended option). Research = **ars-3w quick scans** before locking Stage-2 design details
(graft-interface/neck literature; pruning+KD on Jetson-class GPUs; expansion-cost evidence for the
thesis's descope defense). Timeline = **no hard deadline** (scope by value). Usable compute =
**Colab free T4 + Kaggle (quota restored) + AGX Orin** (training via the `jetson/` kit); the Orin
Nano 8 GB stays measurement-only (mode 0, locked clocks — the board now *idles* in the 25 W Super
regime, so `scripts/setup_jetson.sh` before every session is mandatory).

**The re-scoped Phases 4–9** (PROJECT_PLAN.md rewritten in place; numbering kept, content
redefined; the old text lives in git history): **Phase 4 kept** — CP 4.1 Net2Wider / 4.2
Net2Deeper / 4.3 BN re-estimation as written; CP 4.4 re-aimed at the graft seam
(`net2net/graft_init.py`: identity-embedding ChannelAdapter init) instead of the obsolete
OFA-space graph diff (no second search exists to warm-start). **Phase 5 = Graft-Interface
Ablation → winner-v1.5**: CP 5.0 optional FusedMBConv LUT screen (evidence only); CP 5.1
`detect/neck.py` `ZeroGatedTopDownNeck` (zero-init gates ⇒ day-0 identity) +
`GraftedPoseModel(neck=)` with the `model[-1]`-is-head loss contract preserved and `neck=None`
keeping old state_dicts loadable; CP 5.2 `eval/graft_ablate.py` — V0 control / V1 net2wider
adapter / V2 +top-down neck / V3 optional bottom-up, each at the exact CP 3.5 warm-head protocol
(5 ep, frozen gate donor, 3 seeds {1,2,3}, 640, batch 16), resumable cache
`graft_ablate_e5_r640`, Kaggle `MODE="graft_ablate"`; CP 5.3 Nano e2e benches + AGX 100-epoch
bare-AdamW full-FT of the top-2 (apples-to-apples with 0.841) → ceiling-first pick under the
*measured e2e* T_max → `state/winner_v1_5/`. **Phase 6 = Structured Pruning → winner-v2**:
DepGraph (`torch-pruning>=1.4,<2`, requirements-nas only), ignored_layers = head cv2/cv3/cv4
last convs + dfl, round_to=16, head UNFROZEN during recovery (pruning upstream of a frozen
consumer corrupts its weights), 15/30/45 % ladder × BN-re-estimate + recovery FT × per-point Nano
bench → `data/pruning_curve.json`; measured-only latency claims (off-LUT-grid); a no-win outcome
is recorded honestly and winner-v1.5 carries forward. **Phase 7 = Recipe-Parity Training**:
`eval/recipe_ft.py` (SGD 0.937 nesterov, no-decay BN/bias groups, 3-ep warmup, cosine, EMA,
close_mosaic; state_dict-only checkpoints — `GraftedPoseModel` is function-local, whole-model
pickling crashes, the Ultralytics Trainer is deliberately not wrapped); CP 7.2's parity long-train
is the baseline CP 8.3 must beat (the old CP 7.3 ↔ 8.3 pairing preserved); CP 7.3 = the honest
gap table (proxy → bare-AdamW → parity → KD vs 0.877). **Phases 8–9 kept**, cross-references
updated: teacher = in-repo yolo11s-pose 0.8819 vs a to-be-measured yolo11m-pose (user decision);
KD treatment differs from CP 7.2 only by the teacher term (same init/recipe/seed); CP 9.2
validates predicted-vs-measured (LUT sum + measured pose offset) on the *unpruned* winner-v1.5,
the deployed pruned/distilled engine gets a measured-only figure.

**Execution stages** (mirrored in the session task list): **H** = this pass. **R** = ars-3w scans
→ `docs/research/stageR_{graft_interface,prune_kd_edge,expansion_cost}.md` — gates only Stage-2
design detail; if findings contradict the pivot, stop and re-brief the user. **0** = Jetson truth
(above; prereq: rebuild `.venv-nas` via `scripts/setup_laptop_nas.sh`). **1** = CP 4.1–4.4.
**2** = Phase 5 → winner-v1.5. **3** = Phase 6 → winner-v2. **4** = Phases 7–8. **5** = Phase 9.

**Also fixed in this pass:** CLAUDE.md was stale — its "Current state" still ended at the CP 3.3
buildable slice (2026-06-28); the CP 3.3/3.4/3.5 closes had never landed there. Refreshed to
Phase-3-complete + this pivot.

## CP 4.1 CLOSED — Net2Wider (2026-07-05)

`current_checkpoint` 4.1→4.2, `last_completed` 3.5→4.1, `completed += "4.1"`.

**Built** `net2net/wider.py`: `widen_mapping` (identity prefix + seeded uniform replication),
`widen_conv2d(conv, next_conv, new_out, bn=, seed=)` and `widen_linear(…)` — the Net2Net §3.2
rule (producer rows copied through the mapping; consumer columns copied **divided by the
replication count**), returning fresh modules with the originals untouched; BN affine +
running stats duplicate through the same mapping. Scope guard: `groups=1` producers/consumers
only — the graft seam's 1×1 adapters and OFA's pointwise convs qualify; a depthwise producer
needs a different rule and the refinement track never widens one.

**DoD PASSES** (`tests/test_wider.py`, 6 tests, `.venv`/CI): widened conv→ReLU→conv and
linear→ReLU→linear match the original outputs within 1e-5; conv→BN→hardswish→conv preserved
with duplicated BN stats (eval mode); equal-width widen = exact weight copy; mapping is
deterministic under its seed with an identity prefix; grouped/mismatched/shrinking/wrong-BN
pairs raise. `check.sh` fast lane green (411 passed).

**Post-pivot role** (procedure.md "Plan pivot"): not BO warm-starts — `widen_mapping` is
CP 4.4's substrate (adapter identity-embedding) and the module serves any later width edit
around the winner. Next = CP 4.2 (Net2Deeper).

## CP 4.2 CLOSED — Net2Deeper (2026-07-05)

`current_checkpoint` 4.2→4.3, `last_completed` 4.1→4.2, `completed += "4.2"`.

**Built** `net2net/deeper.py`: `identity_conv2d` (Dirac kernel, zero bias, odd-kernel guard,
`padding=k//2`, stride 1), `identity_linear` (eye + zero bias), and `inserted(seq, index,
*modules)` (non-mutating Sequential splice). Net2Net §3.3 semantics documented in the module:
a bare identity insert is exact anywhere; a conv+activation *block* insert is exact when placed
after an existing **idempotent** activation (`relu(relu(x)) == relu(x)`); a BN-carrying insert
is NOT exact until CP 4.3's invert trick — deliberately split so each guarantee is testable.

**DoD PASSES** (`tests/test_deeper.py`, 5 tests, `.venv`/CI): identity conv exact at
k ∈ {1, 3, 5}; identity linear exact; conv-net and linear-net forwards unchanged (≤1e-6) under
both the bare-identity insert and the identity+ReLU-after-ReLU block insert; even kernels
rejected; the original Sequential is untouched (`inserted` returns a new one).

Next = CP 4.3 (BatchNorm handling — the freeze-vs-re-estimate decision).

## CP 4.3 CLOSED — BatchNorm handling: re-estimate, then invert (2026-07-05)

`current_checkpoint` 4.3→4.4, `last_completed` 4.2→4.3, `completed += "4.3"`.

**The decision the plan required** (freeze-BN-during-first-warm-start-epoch vs the
"re-estimate BN" trick): **re-estimate, then invert** — `net2net/bn.py`:

- `reestimate_bn(model, batches)` — resets every BN's running stats and rebuilds them by
  cumulative averaging (`momentum=None`) over forward-only passes with **only the BNs** in
  train mode (everything else eval, grads off); momenta and the model's train/eval mode are
  restored afterwards; the cumulative average makes the result batch-order-independent.
- `bn_to_identity_(bn)` — sets `(weight, bias) = (sqrt(running_var + eps), running_mean)` so an
  eval-mode BN computes exactly the identity given its current stats.

**Why over freeze-first-epoch:** (1) deterministic and optimizer-free — one forward pass, no
coupling to any training schedule; (2) reusable verbatim where the refinement track needs it —
Phase 6's post-prune recovery (re-estimate after channel surgery) and Phase 5's neck insertion;
(3) the freeze variant keeps stale statistics live inside the frozen window and only reaches the
same operating point if that first epoch is long enough — strictly more moving parts for the
same guarantee. Nuance recorded in the docstring: eval-mode identity holds for *any* stats (the
affine inverts whatever eval normalizes with); re-estimating first is what additionally makes
**train-mode** behaviour near-identity (batch stats ≈ the inverted running stats), so
post-insert training starts from a sane operating point instead of a distortion.

**DoD PASSES** (`tests/test_bn.py`, 4 tests, `.venv`/CI): deepen (identity conv + fresh BN) +
re-estimate + invert preserves the function within the 1e-3 bar — and in fact within 1e-5;
re-estimated stats recover the data's true moments (μ=3, σ²=4 within tolerance, 20 batches);
momentum and train/eval modes restored; empty-batch iterables and BN-less models raise instead
of silently leaving stats reset.

Next = CP 4.4 (graft-seam applicability — the re-scoped last Phase-4 checkpoint).

## CP 4.4 CLOSED — graft-seam applicability: identity-embedding adapter init — PHASE 4 COMPLETE (2026-07-05)

`current_checkpoint` 4.4→5.1, `last_completed` 4.3→4.4, `completed += "4.4"`. **Phase 4 is
complete** under the re-scoped plan (the original `net2net/diff.py` OFA-space graph diff is
obsolete — Phase 3 closed without Net2Net warm-starts and no second search exists; see "Plan
pivot").

**Built** `net2net/graft_init.py`: `identity_embed_conv1x1_(conv, seed=)` — re-initializes an
expanding 1×1 conv as the identity on its first `in_channels` outputs plus Net2Wider-replicated
copies for the extras (reuses `wider.widen_mapping` — CP 4.1 as substrate, exactly as re-scoped),
zero bias, returns the mapping; `apply_adapter_init(adapter, "net2wider", seed=)` maps it over a
ChannelAdapter's per-scale convs with distinct deterministic seeds (`seed+i`). Wired as
**`build_grafted_pose_model(adapter_init="net2wider")`** (`detect/pose_model.py`); `None` keeps
the original random 1×1s — the untouched **V0 control** for CP 5.2's ablation.

**Scope honesty (the re-scoped DoD's own words):** an initialization *prior*, NOT end-to-end
function preservation — Net2Wider's consumer-side division is impossible here because the
consumer is the frozen, donor-trained Pose head whose weights must not be touched. What the
prior buys: the head sees the backbone's real features (identity + exact duplicates) from step 0
instead of random channel mixtures — the LP-FT lesson CP 2.4 established at the head, applied
one level deeper at the adapter. Whether it helps is precisely the V0-vs-V1 question CP 5.2
measures; nothing is claimed here beyond the passthrough property.

**DoD PASSES** (`tests/test_graft_init.py`, 4 tests, `.venv`/CI — `detect.adapter` is
torch-only, so the real `ChannelAdapter` is exercised without ultralytics): first-`in_c`
passthrough exact on a 5→9 conv AND on the real graft shape ((40,112,160)→(64,128,256), all
three scales); extras are exact replicas of their mapped sources (not noise); deterministic
under seed (twin adapters get identical weights); non-1×1 / shrinking / grouped convs and
unknown init kinds raise. `check.sh` fast lane green (411 passed).

**Stage 1 complete → what unblocks.** Stage 2 (CP 5.1 `detect/neck.py` + the ablation) now has
its Net2Net substrate; it still wants Stage 0 (the honest e2e T_max the CP 5.3 selection gates
on) and Stage R (the ars-3w design-detail scans) first. Stage 0 remains the highest-value next
session (~hours, Jetson required).

### Environment note — `.venv-nas` rebuilt as the CPU variant (2026-07-05; not a checkpoint)

The Stage-0 prerequisite rebuild died with "No space left on device": the root filesystem was
**100 % full (1.7 GB free of 197 GB)** and the cu128 stack needs ~7–10 GB. Since the laptop's
CUDA has been broken all project (`torch.cuda.is_available()` False — every fine-tune already
runs on Kaggle/Colab/AGX) and the Stage-0 note already said CPU torch suffices for the ONNX
export, the venv was rebuilt CPU-variant through the script's own knob: `pip cache purge`
(freed 2.0 GB) → matched **torch 2.11.0+cpu / torchvision 0.26.0+cpu** pre-installed from
`https://download.pytorch.org/whl/cpu` (a matched pair, so the "libcudart.so.13"
PyPI-mismatch trap cannot occur) → `TORCH_CUDA_INDEX=cpu bash scripts/setup_laptop_nas.sh` for
the rest. The **OFA checkpoint was also missing** from `.cache/ofa/` and was re-fetched via
`python -m supernet.download_ofa` (sha256 matches the pin, `a7def36b…`).

Verified in the new env: the script's import smoke (`nas env ok: torch 2.11.0+cpu cuda False`);
`tests/test_grafted_pose_model.py` **6/6** (previously auto-skipped on this laptop); and a real
`build_grafted_pose_model(random_arch, adapter_init="net2wider")` forwarding
(1,3,640,640) → (1,29,8400) finite — the exact construction path Stage 0's exporter will use.

Restore the GPU variant only if the laptop's CUDA ever gets fixed: free ~10 GB, then plain
`bash scripts/setup_laptop_nas.sh`. **User action item (outside this project):** the disk is
still at 99 % (≈2 GB free) — critically low for the OS itself.

## Stage R COMPLETE — three ars-3w literature scans → docs/research/ (2026-07-06; not a checkpoint)

The pivot's research step (user picked the "3W quick scan" scope at the 2026-07-05
AskUserQuestion): three `deep-research` three-way scans, each a WHY/HOW/WHAT comparison of the
3 strongest papers for one open design decision, with web-verified anchor claims and explicit
fidelity notes (escalate to `lit-review` mode when writing the thesis chapters). **No finding
contradicts the pivot** — scan (iii) strengthens it. What each scan concretely changed:

1. **`stageR_graft_interface.md`** (YOLOF / EfficientDet-BiFPN / ViTDet + ReZero, LP-FT):
   **resolves the CP 5.1 gate-granularity question → one scalar zero-init gate per fusion
   edge** (ReZero's exact mechanism; BiFPN's fusion weights are per-edge scalars too — no
   evidence for per-channel gates, and they'd add TRT pointwise cost). Sets expectations:
   top-down-only fusion is the weakest-but-cheapest topology, and fusion per se is worth
   ≲1–2 AP-class once per-scale outputs exist (YOLOF's SiMo-within-<1-mAP; ViTDet's
   no-fusion pyramid ≈ FPN) — so the V0→V1 (init) vs V1→V2 (fusion) decomposition of the
   CP 5.2 ablation is the right experiment, and a V2≈V1 outcome is pre-registered as a
   literature-consistent negative result, not a failure. The frozen-consumer-head setting has
   NO direct literature — the ablation is novel evidence.
2. **`stageR_prune_kd_edge.md`** (DepGraph / arXiv:2509.12918 prune+CWD-on-YOLOv8-edge / CWD):
   confirms CP 6.1 as designed (group-level importance is DepGraph's own recommendation);
   grounds `round_to=16` in NVIDIA's TRT channel-alignment guidance (fp16 tensor cores want
   in-channels %8; implicit padding otherwise); adds an **optional CP 6.2 design input** —
   recover pruned nets *with distillation* (teacher = the unpruned winner-v1.5, free) instead
   of plain FT, decision deferred to CP 6.2 (user); and **re-ranks the CP 8.2 loss menu**:
   box branch → Localization Distillation (YOLO11's DFL is already a distribution — LD is a
   near-zero-friction fit), any feature term → CWD channel-normalized form at P3/P4/P5,
   keypoints → regression mimic (no literature standard; our design, say so in the thesis).
3. **`stageR_expansion_cost.md`** (OFA / CompOFA(+DϵpS) / Yu et al. ICLR 2020): the descope
   defense now has two independent literature grounds — cost (OFA ≈1,200 V100-h; even the
   efficiency line saves by *shrinking* spaces) and ranking fidelity (weight sharing degrades
   candidate ranking toward random-search parity; rank correlation worsens as the space
   grows — NAS-Bench-201 only ranks well when downscaled ~64×), with CP 3.5's σ-vs-top-gap
   measurements as the in-situ replication. D3's own rule of thumb, evaluated on Phase-3's
   outcome, already said "don't inject".

Stage-2 design detail is now unblocked: CP 5.1 proceeds with scalar per-edge gates (top-down
V2; PAN-style 3×3/s2 bottom-up for V3).

## CP 5.1 CLOSED — variant library: zero-gated nano-neck + graft wiring (2026-07-06)

`current_checkpoint` 5.1→5.2, `last_completed` 4.4→5.1, `completed += "5.1"`.

**Built** `detect/neck.py`: `ZeroGatedTopDownNeck` — P5→P4 and P4′→P3 fusion edges, each a 1×1
projection + ×2 nearest upsample + **one zero-initialized scalar gate per edge** (the Stage-R
resolved design: ReZero's mechanism, BiFPN's per-edge-scalar precedent; per-channel gates have
no evidence and add TRT cost); `bottom_up=True` adds the PAN-style return path (3×3 stride-2 +
gated adds over the *updated* maps) for V3. `gate_values()` reports the learned magnitudes so
CP 5.2 can answer "did the data ever turn the neck on?" (the Stage-R risk). `build_neck(kind)`
= the variant switch: `None` (V0/V1) | `"topdown"` (V2) | `"pan"` (V3). Torch-only.

**Graft wiring** (`detect/pose_model.py`): `GraftedPoseModel(…, neck=)` — `neck=None` keeps
the original 3-module `Sequential` (pre-CP-5.1 state_dicts, e.g. `full_finetune_weights.pt`,
load unchanged); with a neck the layout is `(backbone, adapter, neck, head)` and
`_predict_once` generalized to `*body, head = self.model` — **`model[-1]` stays the Pose head**
(the `v8PoseLoss`/`init_criterion`/`_apply` contract). `build_grafted_pose_model(neck="topdown"|"pan")`
composes it with `adapter_init=` (CP 4.4), covering all four CP 5.2 variants from one factory.

**DoD PASSES.** `tests/test_neck.py` (6, `.venv`/CI): neck-at-init output is **exactly** equal
to its input (torch.equal — the day-0 function-preservation DoD) for both topologies; open
gates change outputs and preserve shapes; the ReZero gradient dynamic is verified (at init the
gates receive gradient, the gated-off projections receive exactly zero; once a gate opens the
projection trains); dispatch + shape guards. `tests/test_grafted_pose_model.py` (+3, gated,
run in `.venv-nas`): with a neck the model is 4 modules and `model[-1]` is the real Ultralytics
`Pose`; eval outputs of the with-neck graft are **bit-identical** to the same modules without
the neck (both `"topdown"` and `"pan"`); pose loss is finite, gates receive finite grads, and
the gradient still reaches the backbone stem. Full fast lane green (429 passed); `.venv-nas`
run 19 passed.

**Next = CP 5.2** (`eval/graft_ablate.py` — V0 control / V1 `adapter_init="net2wider"` / V2
`neck="topdown"` / V3 `neck="pan"` (conditional), 3 seeds × 5-epoch warm-head proxy, resumable
cache, Kaggle `MODE="graft_ablate"`).

## CP 5.2 — buildable slice BUILT: eval/graft_ablate.py + Kaggle MODE="graft_ablate" (2026-07-06)

**No checkpoint advance** (current stays 5.2 — the DoD needs the GPU run; the CP 3.3
"buildable slice" precedent). Built, all `.venv`/CI-tested where pure:

- `eval/shortft.short_finetune` gains **`graft_kwargs`** — forwarded verbatim to
  `build_grafted_pose_model`, so the ablation runs the *exact* CP 3.5 oracle with only the
  interface changed; `None` (default) is byte-compatible with every prior proxy result.
- `eval/graft_ablate.py`: the variant table (V0 control / V1 `adapter_init="net2wider"` /
  V2 +`neck="topdown"` / V3 +`neck="pan"`), the CP 3.5 protocol (5 epochs, frozen gate donor,
  imgsz 640, batch 16, fresh seeds {1,2,3}), a resumable per-(variant, seed) jsonl cache
  namespaced **`graft_ablate_e5_r640`** (the denoise cache pattern), **V3 auto-gated by the
  >1σ rule** (`v3_warranted`, overridable `--include-v3`/`--skip-v3`), and per-seed **neck
  gate magnitudes** recorded via a scratch state_dict save → `gates_from_state_dict` — the
  Stage-R "did the data turn the neck on?" diagnostic. `assemble_report` emits per-variant
  mean±σ plus the two headline deltas (V1−V0 = the init effect, V2−V1 = the fusion effect —
  the decomposition Stage R identified as the right experiment).
- `kaggle/run.py` **`MODE="graft_ablate"`** (a clone of the denoise block; cache round-trips
  through the input Dataset the same way). RES/T_MAX untouched — the regime gate stays green.
- Tests: `tests/test_graft_ablate.py` (6): variant table, summarize/v3-gate math, gate
  extraction, stubbed-fine-tune orchestration (cache resume skips paid work; kwargs +
  freeze_head passthrough), report deltas. `check.sh` green (435 passed).

Session cost ≈ 9–12 warm-head fine-tunes ≈ ⅓ of the de-noise campaign → one Kaggle session.
**CLOSE needs:** the Kaggle run → `data/graft_ablate.json` (mean±σ per variant + gates) →
close entry + `plan_state` advance; then CP 5.3 (Nano e2e of V2/V3 graphs + AGX full-FT of the
top-2 + ceiling-first selection under the honest e2e T_max).

## CP 6.1 CLOSED (out of order) — DepGraph pruning harness; CPU-smoke DoD passes (2026-07-06)

`completed += "6.1"`; **`current_checkpoint` stays "5.2"** — the critical path is still the
CP 5.2 GPU run (the CP 1.4 backfill precedent for out-of-order closes). CP 6.1's DoD is a CPU
smoke, so it closes today; CP 6.2 (the ladder) waits for winner-v1.5.

**Built** `prune/prune_graft.py` (+ `torch-pruning>=1.4,<2` pinned in `requirements-nas.txt`,
the `kaggle/run.py` pip line, and `jetson/Dockerfile` — never `requirements.txt`):
`head_ignored_layers` collects the semantic output convs (last conv of every `cv2`/`cv3`/`cv4`
scale + `dfl.conv` — the output-format contract DepGraph must never touch);
`prune_graft(model, example_input, ratio=, round_to=16)` runs Torch-Pruning 1.6's `MetaPruner`
with **GroupMagnitudeImportance(p=2)** (DepGraph's own recommendation, per the Stage-R scan)
and returns a report (params before/after, per-conv channel changes, and an **alignment
audit**: every changed plain-conv out-channel must be a multiple of `round_to`).

**Two hard guards, one real catch.** (1) Frozen params outside the protected set are refused —
DepGraph slices consumer in-channels, so pruning around frozen weights silently corrupts them
(the plan's frozen-head trap). First implementation was too broad and tripped on Ultralytics'
**DFL conv, which is permanently frozen by design** — the guard now exempts parameters inside
the *ignored* modules (frozen AND protected = safe; the DFL's input channels can't change
because `cv2[i][-1]` is ignored too). (2) Ratio validated before any model inspection.

**DoD PASSES** (`tests/test_prune_graft.py`, 4 gated tests, `.venv-nas`, torch-pruning 1.6.0):
20 % group prune on the stub graft → forward OK, decoded output shape unchanged, params
reduced, every changed conv %16-aligned, head output convs untouched (64/1/24 channels);
whole-head-frozen refused; **the necked graft (CP 5.1 `topdown`) prunes cleanly too** —
DepGraph copes with the 0-dim scalar gates — so a winner-v1.5-with-neck needs no special
casing at CP 6.2. `prune/` added to mypy coverage.

**Remaining for Phase 6:** CP 6.2 (15/30/45 % ladder × `reestimate_bn` + recovery FT × Nano
e2e per point → `data/pruning_curve.json`; optional KD-recovery per the Stage-R flag — user
decision) and CP 6.3 (operating point, user decision) — both gated on winner-v1.5 + the board.

## CP 5.2 CLOSED — graft-interface ablation: the neck is real, PAN wins (+0.026 proxy mAP) (2026-07-07)

`current_checkpoint` 5.2→5.3, `last_completed` 6.1→5.2 (critical path), `completed += "5.2"`.
Kaggle kernel v19 (T4, 96 min; v18 died on the P100-reset gotcha — fixed by pinning
`machine_shape=NvidiaTeslaT4`); result `data/cp33_kaggle_out/graft_ablate.json` + the
per-(variant,seed) cache. Protocol = the exact CP 3.5 oracle (5-epoch warm-head, frozen gate
donor, fresh seeds {1,2,3}); winner-v1's backbone in all four arms.

| variant | proxy mAP (mean ± σ) | Δ |
|---|---|---|
| V0 control (random 1×1 adapters, no neck) | 0.6027 ± 0.0131 | — |
| V1 `adapter_init="net2wider"` | 0.5969 ± 0.0043 | −0.006 vs V0 |
| V2 V1 + zero-gated top-down neck | 0.6106 ± 0.0073 | +0.014 vs V1 → V3 auto-ran |
| **V3 V1 + PAN (top-down + bottom-up)** | **0.6287 ± 0.0078** | **+0.026 vs V0, +0.018 vs V2** |

**Findings.**
1. **The fusion decomposition answered the Stage-R question in the opposite direction of the
   pre-registered negative result:** cross-scale fusion is the lever (V2−V1 = +0.014, V3−V2 =
   +0.018 — each ≥ the ~0.014 top-cluster gaps that decided CP 3.5), while the adapter
   *init* prior is not (V1−V0 = −0.006, inside noise; the LP-FT-style hypothesis did not pay
   at the adapter level — worth keeping as an honest null in the thesis). V3's 0.6287 exceeds
   every de-noised frontier point (accuracy-first arch: 0.624 @ 12.65 ms) — **fixing the
   interface beat searching more architectures**, the pivot's core bet.
2. **The gates opened.** Top-down: g43 ≈ 0.39, g54 ≈ 0.20, consistent across all seeds (the
   data wants the coarse-to-fine path, especially into P3 — where 8-keypoint localization
   lives). Bottom-up gates stayed small (|g| ≈ 0.05–0.08, mixed signs) yet V3 > V2 — the
   return path's 3×3/s2 convs contribute beyond their gates; a thesis footnote, not a blocker.
3. **Determinism cross-check:** V0's seeds 2/3 reproduce the CP 3.5 de-noise mAPs
   **bit-exactly** (0.6169063798…, 0.6057934327…) — same code path, same seeds, different
   Kaggle session. Seed 1 deviated (0.5853 vs 0.6076) — one cross-session/environment
   nondeterministic draw; recorded as a caveat on V0's mean (its σ 0.0131 is inflated by that
   draw), does not affect the V3 verdict (same-session comparison, margin ≫ σ).

**Consequences → CP 5.3.** Top-2 = **V3 and V2**; both graphs' e2e Nano benches are in this
session's Stage-0 batch (V3 costs +410 K params, V2 +41 K — the latency side decides how much
of the +0.026 is affordable under the honest T_max). Next: AGX 100-epoch full-FTs
(`eval.full_finetune --adapter-init net2wider --neck pan --tag v3pan` / `--neck topdown --tag
v2td`) + ceiling-first selection → `state/winner_v1_5/` (user confirms the pick).

## STAGE 0 COMPLETE — the end-to-end truth: the winner does NOT beat the baseline; the claim is retired (2026-07-07)

The owed measurement (procedure.md "Plan pivot" consequence #1) ran on the Orin Nano — one
session, `power_mode 0`, `clocks_locked=true` (preflight-verified per bench), TRT fp32 first
(the LUT-comparable regime), then the fp16 deploy-precision trio. The baseline re-check
reproduced the anchor **exactly** (12.75 ms), so every number below is same-regime comparable.

**fp32 @640 (LUT-comparable), seven models:**

| model | e2e ms | vs baseline |
|---|---|---|
| yolo11n-pose (baseline re-check) | **12.75** | — |
| winner-v1 graft e2e (`d=[2,2,4,3,3]`) | **17.69** | **+38.7 %** |
| winner-v1 backbone only | 13.85 | sum was 11.208 → **×1.236** |
| fallback `[2,2,4,3,2]` (idx 11) e2e | 16.65 | +30.6 % |
| fallback `[2,4,3,4,4]` (idx 3) e2e | 16.11 | +26.4 % |
| winner + V2 top-down neck e2e | 18.13 | neck costs +0.44 ms |
| winner + V3 PAN neck e2e | 18.38 | neck costs +0.69 ms |

**fp16 @640 (the deploy precision):** baseline **7.584 ms** (1.68× from fp32) — winner e2e
**12.37 ms** (only 1.43×) — V3 e2e **12.75 ms** (1.44×). The depthwise-heavy OFA family gains
far less from tensor cores than the baseline's dense convs, so at deploy precision the gap
*widens* to **+63 %**.

**Decomposition (all three audit fears materialized, coherently):**
1. **The @224 additivity calibration inverts at 640**: measured whole-backbone / raw LUT sum =
   **1.236** (at 224 the fit was 0.934 — TRT fusion made the whole *faster* than the sum; at
   640 the cross-block activations are DRAM-bound on the 62.5 GB/s Nano and the sum
   *under*-predicts by 23.6 %). Recorded in `data/pose_stem_head_offset.json`
   (`backbone_measured_vs_lut_sum`) — the first @640 additivity data point.
2. **The pose stem/adapter/head offset is 3.84 ms** (17.69 − 13.85), vs the 0.39 ms *classifier*
   offset @224 that was never applicable.
3. The `×1.236 + 3.84` model predicts the two fallbacks within 0.3–0.45 ms of their measured
   values → the decomposition is sound; **no LUT-frontier candidate can beat 12.75 ms e2e**
   (the fastest sum, 10.15, lands at 16.11 measured). The ceiling-first re-pick path is moot —
   the gap is structural (family + offset), not arch choice.
4. `search/cost.py`'s own limitation note (peer-review R4.3: "rankings are not
   precision-invariant... a search result is faithful at the searched precision only")
   **predicted exactly this**: per-block fp32 LUT fidelity (ρ=0.991 @224) survived, absolute
   e2e transfer did not.

**What was stamped/fixed:** `state/winner_v1/winner.json` now carries the additive `e2e` block
(`speedup_pct_e2e = −38.7 %`, `winner_beats_baseline_e2e: false`, both fallbacks, regime
stamps) — pre-existing keys untouched; `data/pose_stem_head_offset.json` is the measured
CostOffset (`cost(..., res=640, stem_head=…)` now gives honest absolute e2e). "MAXN" audit
item resolved with a twist: the three flagged mentions (CP33_BACKENDS.md, jetson/README.md,
deploy.sh) are the **AGX** compute board, where mode 0 *is* MAXN — correct as written; only
`lut/README.md`'s Nano mode-0 label needed the clarification (612 MHz / 15 W pre-Super max ≠
"MAXN SUPER" 25 W/918 MHz, never measured here).

**The one genuinely good number:** V3 (the CP 5.2 accuracy winner, +0.026 proxy) at fp16 runs
**12.75 ms = 78 FPS — it MEETS the 60 FPS (16.7 ms) deployment bar with 24 % headroom**. The
"faster than yolo11n" headline is retired permanently; whether the thesis pivots to
"≥ baseline accuracy at 60-FPS-deployable latency" + the quantified transfer-gap findings is
the user's call (decision brief follows this entry). **Do not use any LUT-summed latency as an
absolute claim anywhere; ranking-only.**

## The @640 additivity study — why deployment runs slower, characterized over 9 archs (2026-07-07)

User-requested follow-up ("try different BO candidates to analyze why deployment runs
slower"). Probe = 8 new backbone-only exports benched in one Nano session (mode 0, fp32) +
the winner's Stage-0 row: six de-noise candidates (sums 10.15–12.70, diverse `d` patterns) +
the OFA **min corner** (`d=[2]⁵`, sum 6.85) + **max corner** (`d=[4]⁵`, sum 27.11) — a 4×
sum range. Tool: `search/additivity640.py` (pairing via export meta sidecars; refuses
unlocked-clock rows); report: `data/e2e/additivity640_report.json`.

| arch (d) | sum ms | measured ms | ratio |
|---|---|---|---|
| min corner `[2,2,2,2,2]` | 6.847 | 8.054 | 1.176 |
| `[2,4,3,4,4]` (idx3) | 10.153 | 12.269 | 1.208 |
| `[2,2,4,3,2]` (idx11) | 10.727 | 12.813 | 1.194 |
| `[2,3,4,2,4]` (idx4) | 11.193 | 13.128 | 1.173 |
| **winner `[2,2,4,3,3]`** | 11.208 | 13.852 | **1.236** |
| `[2,2,4,4,3]` (idx0) | 11.744 | 14.424 | 1.228 |
| `[2,2,4,4,2]` (idx2) | 12.645 | 15.291 | 1.209 |
| `[2,2,4,3,4]` (idx1) | 12.702 | 14.962 | 1.178 |
| max corner `[4,4,4,4,4]` | 27.114 | 30.981 | 1.143 |

**Findings.**
1. **The @640 law is affine, not multiplicative: measured ≈ 1.115·sum + 0.926 ms
   (R² = 0.9975).** The @224 fit was 0.934·sum − 0.02: at low resolution TRT cross-seam
   fusion makes the whole *faster* than its parts; at the deploy resolution the seams *cost*
   ~11.5 % (cross-block activation traffic on the 62.5 GB/s Nano) plus ~0.93 ms of
   per-engine fixed overhead. **Compositional latency prediction is resolution-dependent in
   sign** — the study's headline finding.
2. **The search's ranking SURVIVES: Spearman(measured, sum) = 0.983 over the 9-arch probe**
   (which spans the space's corners). Phase 3's frontier ordering, BO/TPE DoDs, and the
   winner's-curse analysis all stand; the damage was confined to absolute claims and ceiling
   feasibility. (Residuals are ±~0.4 ms — the winner is the largest positive residual, which
   is why its single-point ratio read 1.236.)
3. **The naive DRAM story is refuted in its simple form:** rel-err vs early-stage depth
   (d0+d1) correlates **negatively** (−0.60), i.e. shallower/smaller nets suffer relatively
   *more* — the affine intercept explains this mechanically (a fixed ~0.9 ms weighs more on
   small nets). The per-work penalty is the 11.5 % slope; attribution finer than
   "cross-block memory traffic + fixed engine overhead" would need per-layer profiling
   (future work, `trtexec --dumpProfile`).
4. **Honest feasibility, reconstructed:** for graft e2e ≤ baseline (12.75 fp32) a backbone
   needed sum ≤ (12.75 − 3.84 − 0.93)/1.115 ≈ **7.2 ms** — only the min-corner region
   qualifies; the searched frontier (10.15+) never contained a winner. For the 60-FPS bar
   (16.7 ms) the honest sum ceiling is ≈ 10.7 ms fp32 — the fast half of the frontier
   qualifies even at fp32, and everything measured qualifies at fp16.

**Consequence:** `cost(..., res=640)` absolute predictions must use this fit + the pose
offset (both opt-in, ranking-neutral); the @224 `data/latency_calibration.json` stays
untouched (different regime). The findings chapter now has the full quantified chain:
per-block LUT (ρ=0.991 @224) → additive ranking valid at 640 (ρ=0.983) → absolute transfer
breaks affinely (+11.5 %, +0.93 ms) → offset 3.84 ms → fp16 asymmetry (1.43× vs 1.68×).

## Phase 3b LAUNCHED — honest-ceiling re-search (user-directed, 2026-07-07)

**User decision:** "retry the search with the new T_max." Operationalized as the honest
beat-the-baseline ceiling: e2e ≤ 12.75 ms ⇔ **backbone sum ≤ 7.16 ms**
(= (12.75 − 0.926 − 3.837)/1.115) — a **0.31 ms band above the space's own floor** (min corner
6.847). The run asks: what is the best warm-head proxy accuracy this family can buy inside its
baseline-beating band? (A-priori expectation, recorded before results: ≈ 0.52–0.56, refining
the null trade the frontier re-scoring exposed; an upside surprise would re-open the framing.)

**Design notes.** (1) Uniform arch sampling starves in the band (a *biased* local sampler hit
2.8 %, 1,566 feasible of 56 k tried) → BO's warm-start seeds are pre-generated and pinned:
`state/honest_search/nsga2_seeds_tmax716.json` — 113 stratified feasible archs spanning
6.847–7.16 incl. the three reachable deeper-depth patterns (`[2,2,3,2,2]`, `[2,2,2,2,3]`);
`bo.py`'s incumbent-mutation candidate pools then sustain band-local proposals. (2) The RS
control **will starve** (bounded attempts by design) — the HV-vs-random comparison is
explicitly NOT a claim of this run; the product is the band's frontier + de-noise-able top-K.
(3) Own cache namespace **`hs_bo_cache_r640`** — the CP 3.3/3.4 DoD caches and their pinned
`(RES, T_MAX_MS)` regime are untouched (the regime gate keeps watching them; `HS_T_MAX` is a
separate constant). Protocol: BO, seeds {0,1}, budget 30/seed, n-init 15, the same warm-head
oracle (5 ep, frozen gate donor, imgsz 640). Kaggle `MODE="honest_search"`, resumable.

**Owed before ANY pick from this run:** 3-seed de-noise of the top-K (`search.denoise`
machinery, new candidates file — winner's-curse discipline) **and** an e2e Nano bench of the
chosen arch (the honest-cost prediction is a model; the claim needs the measurement).

**Addendum — the frontier re-scored with the honest cost (user question: "shouldn't the head
latency be summed into the search?").** Answer: yes, and it is exactly what `cost.py`'s
`stem_head` offset was built for — but a shared-head offset is **rank-neutral**, so it changes
*feasibility*, never candidate ordering; the casualty was always the ceiling. Applying
`e2e ≈ 1.115·sum + 0.93 + 3.84` to the full 130-point BO∪TPE frontier: **25 candidates do
honestly beat the baseline's 12.75 ms** — but they are the frontier's cheap end (best of them:
proxy **0.5202**, single-seed, `d=[2,2,3,2,2]`, margin **+0.0 %**; next: 0.515 at +0.4 %). The
accurate cluster (0.60–0.63) is entirely infeasible vs the baseline. So an honest-cost search
would have surfaced the verdict on day one: *in this family, beating yolo11n is only possible
at a ~0.10+ proxy-mAP sacrifice for ≤0.4 % margin* — a null trade under the known saturation
(anchor B: +70 % latency ↔ +0.5 full-train mAP). Under the surviving 60-FPS bar instead:
**130/130 feasible at ~fp16, 112/130 even at fp32** → accuracy-first selection is
unconstrained, and V3-on-winner (0.6287 measured proxy, 12.75 ms fp16) remains the best point
evaluated to date. Standing rule going forward: every selection uses measured e2e (CP 5.3
onward) or the honest fit+offset; the fast-cluster single-seed accs would need their own
de-noise before any hypothetical re-pick (winner's-curse discipline).

## Phase 3b CLOSED — the honest-ceiling band tops out at 0.4965: null trade CONFIRMED (2026-07-07)

**Result vs pre-registration: a miss, below the band.** The a-priori expectation (recorded in
the LAUNCHED entry before results) was best-band proxy ≈ 0.52–0.56. Measured
(`data/phase3b_honest_search.json`, Kaggle T4, ~3.4 h wall): **best proxy 0.4965 @ sum
6.989 ms**, frontier span 0.475–0.496 over 6.85–6.99 ms. Both seeds' BO trajectories
independently proposed the same top arch (d=[2,2,2,2,2], ks mostly 3 with e=6 hotspots at
stages 2–3; one shared-memo evaluation, two independent proposals — convergence, not
replication). BO HV 0.1534±0.0005 over 2 seeds, both `complete`; `rs_hv=0.0` exactly as
pre-declared (RS starves in the band — a non-claim, not a comparison).

**The feasible region degenerated to the min-depth corner.** Of 42 unique archs evaluated
(20+22; duplicates memo-deduped), **41 were d=[2,2,2,2,2]** and one was `[2,2,2,2,3]`
(7.137 ms — inside the ceiling by 0.023 ms, didn't reach the frontier). The 0.31 ms band above
the space's floor (6.847) admits essentially one depth pattern, so the "search" reduces to
ks/e tuning at minimum depth — and ks/e at min depth buys ≈0.02 proxy (0.475 → 0.496).
Notably the re-scored frontier's best honest-feasible point (0.5202 single-seed,
`d=[2,2,3,2,2]`, margin +0.0 %) was *not* re-found: its depth pattern sits exactly on the
ceiling and the 15-of-113 n-init draw missed it; the directed search could not recover it from
inside the band. Both numbers tell the same story at ±0.03.

**Conclusion — the negative result is now search-confirmed, not just re-scored.** Retrying the
search under the honest cost does not rescue the beat-yolo11n-fp32 claim: the band's ceiling
(single-seed max, i.e. *upward-biased* by the winner's-curse mechanism CP 3.5 quantified at
≈+0.04) is 0.4965, a **−0.13 proxy sacrifice** vs V3-on-winner's measured 0.6287±0.008
(3-seed) for a ≤1.5 % latency margin (honest e2e prediction of the band's best:
1.115·6.989+0.926+3.837 ≈ 12.56 ms vs baseline 12.75). Under the known task saturation
(anchor B) that trade is null. **No pick is made from this run** → the LAUNCHED entry's
de-noise + e2e-bench obligations are moot (they gate picks); noise σ≤0.025 cannot close a
0.13 gap, and the single-seed bias runs *against* the band, making the negative conservative.
The deployment framing stands where Stage 0 + CP 5.2 left it: accuracy-first under the 60-FPS
bar (V3 fp16 12.75 ms = 78 FPS), with the fp32-beat retired. Thesis value of this run: it
closes the "but what if the search had used the honest cost from day one?" objection with a
directed experiment — the answer is *the family cannot buy accuracy inside its
baseline-beating band*, and the interface work (Phase 5) is where the accuracy actually came
from. Artifacts: `data/phase3b_honest_search.json`, per-seed caches
`data/hs_bo_cache_r640.seed{0,1}.bo.jsonl`; user-owned decisions (thesis framing, CP 5.3
full-FT launch) remain open — this run removes the last "re-search might change the picture"
contingency from both.

## Plan amendment — the dense-family arm (A+B1+B2) + three-account parallel launch (2026-07-07)

**Context.** After Phase 3b closed, the user asked (1) why a MobileNet loses to YOLO on the
Nano at all, (2) what could overcome it, (3) whether a different supernet would, and finally
for a similar-cases search + full option briefing. The mechanism answer, from our own
artifacts: depthwise/MBConv primitives are DRAM-bound at 640 on the 62.5 GB/s Nano (backbone
0.30 TFLOP/s effective vs the dense head's 0.58 *inside the same engine*; baseline whole-net
0.60), SE blocks serialize, fp16 tensor-core gains skip depthwise (1.43× vs 1.68×) — and no
public supernet fixes both the *primitive* and the *scale* (classifier supernets are 224-px,
ImageNet-width; OFA-ResNet50's floor lands ~2× over budget @640 by roofline estimate). The
similar-cases scan pinned every leg to literature: G-GhostNet (arXiv:2201.03297 — CPU-light ≠
GPU-fast, family-level fix), YOLO-NAS (dense QA-RepVGG space, hardware-aware, ~3,800 GPU-h)
and DAMO-YOLO/MAE-NAS (ResNet/CSP-like under TRT) for what industry searched, FBNetV5
(arXiv:2111.10007) for classification→detection transfer being a named failure mode, and
arXiv:2509.12918 + arXiv:2501.16571 for prune→recover→TRT on YOLO/Jetson (26→68 FPS, −2.7
AP50 at −73.5 % params). Full option table (A / B1 / B2 / C screen-gated / D catalog / E
rejected) in the session log; C/D are scale-walled by the same evidence.

**User decisions (AskUserQuestion):** scope = **A + B1 + B2** — keep the current arc AND add
the pruned-baseline control AND the dense scaling search; CP 5.3 full-FTs launch on
**Kaggle now**; then (follow-up instruction) **use all three Kaggle accounts in parallel**
(`secrets/` holds three KGAT token+username pairs — the accounts already carried the gate-pose
dataset from the June quota rotation).

**Built + launched (all three kernels live, one campaign per account):**
1. **acct1 owaismalekarnous v22 — CP 5.3:** `kaggle/run.py` MODE=full_finetune now runs
   `FULL_FT_VARIANTS` = v3pan + v2topdown (100 ep, seed 0, warm head unfrozen), one per T4.
   The 0.841 bare-winner run is the standing control; ~2.5 h.
2. **acct2 asilarnous v14 — CP 6.2-B (B1):** new `prune/prune_baseline.py` — the *identical*
   Phase-6 ladder (DepGraph group-L2, round_to=16, 15/30/45 %) on the gate-trained yolo11n
   donor: prune → `reestimate_bn` → bare-AdamW recovery (50 ep) → deploy-contract ONNX +
   report row; donor re-anchored under the same validator. `head_ignored_layers` needed zero
   changes (stock PoseModel satisfies the same `model.model[-1]` contract). ~2–3 h.
3. **acct3 asilarnous47 v6 — Phase 3c wave 1 (B2):** new `search/dense_family.py` — 6
   yolo11-pose `scales:` candidates (ctrl_n = yolo11n's own triple from scratch, the recipe
   control; 5 sub-n points), stock Ultralytics recipe, seed 0, per-tag row files (resumable,
   coordination-free 2-T4 striping), deploy-ONNX per candidate. ~5–7 h.
   PROJECT_PLAN.md gains **Phase 3c** (CP 3c.1 wave → 3c.2 de-noise/wave-2-if-warranted →
   3c.3 cross-family verdict figure) and Phase 6 gains **CP 6.2-B**.

**Infra:** `kaggle/push.sh` multi-account (`KACCT=2|3` → the "(Copy)"/"(Copy 2)" credential
pairs; per-account `--pull` dirs `data/kaggle_out_<user>/`) + `KMODE=<mode>` (rewrites the
MODE line in the *staged* run.py only — one codebase, per-account campaigns). Kernel slug is
per-account (`<user>/tfm-nas-cp3-3-search`), so the three runs never collide.

**Standing obligations attached to this arm:** every latency is **measured-only** (next Nano
session benches: CP 5.3's winner-v1.5 e2e, both pruning ladders' ONNX, the 6 dense-wave ONNX
— plus the deferred riders: SE ablation, 512-res sweep, FusedMBConv + OFA-R50 LUT screens);
wave-1/ladder picks are single-seed → CP 3.5 de-noise discipline before ANY selection; the
cross-family comparison must plot *recipe-consistent* numbers (ctrl_n anchors the from-scratch
axis; CP 7.2 parity anchors the winner side). Thesis framing question stays open but is now
concretely: which measured frontier point ships, and the findings chapter explains why the
families rank as they do.

### Incident — prune_baseline OOM (rc=137) on Kaggle, and the yolo11 DepGraph prep (2026-07-08)

**Symptom.** acct2's first `prune_baseline` kernel (asilarnous v14) was SIGKILL-OOM'd
(rc=137, ~12 GB host) ~18 min in — *after* a clean donor val (0.887), so the model + data
paths were fine. The failure was inside torch-pruning's `get_all_groups` (dependency-group
construction), reproduced locally under a 4 GB cgroup cap (`systemd-run -p MemoryMax=4G`) with
an RSS watchdog that `interrupt_main()`s at 3 GB to capture the live stack instead of a bare
kill. **The OFA graft (CP 6.1) never hit this — all three causes are yolo11-structural:**

1. **C2PSA attention.** The `qkv → view(B,heads,dims,H·W) → matmul` reshapes register
   `_FlattenIndexMapping` chains whose index lists grow *multiplicatively* as groups assemble
   → unbounded RAM. No cheap fix; the attention blocks are kept **dense** via `ignored_layers`
   (`prune/yolo_tp_prep.attention_modules`, ~2 % of yolo11n params). tp expands an ignored
   module to `list(.modules())`, so only the *outermost* C2PSA is passed.
2. **C2f `chunk`.** `C2f.forward` (inherited by every `C3k2`) does `cv1(x).chunk(2,1)` — one
   conv feeding two index spaces; DepGraph mis-couples it (`IndexError: index 384 vs size
   256`). Fixed by rewriting each block into two explicit convs (`C2fSplit`), the canonical
   torch-pruning-YOLOv8 remedy — **function- and param-count-preserving** (conv/BN rows are
   independent, so slicing `cv1` into `cv0`+`cv1` is exact).
3. **Trace resolution.** DepGraph runs a real forward holding every activation via `grad_fn`,
   so host memory scaled with the 640 deploy res; the coupling result is res-independent.
   Trace at `TRACE_IMGSZ=128` (the CP 6.1 DoD size).

**Two silent copy bugs** in `C2fSplit`, caught only by *end-to-end* function-preservation
(weight-equality alone missed both): ultralytics stamps `eps=1e-3, momentum=0.03` onto BN
*after* construction (a fresh Conv has torch defaults), and a fresh `nn.Module` defaults to
**train** mode — each drifts a "byte-identical" copy by ~5e-2 per block. Also fixed in the same
pass: `recovery_finetune` now vals a `deepcopy` (Ultralytics' AutoBackend fuses Conv+BN *in
place*, which would strip BN from the model we then save/export) and donor params are counted
pre-val.

**round_to=16 floor (worth recording for the ladder read).** On a net this small, snapping
kept-channel counts to multiples of 16 dominates: measured pruning_ratio→param-sparsity is
0.05→0.286, 0.15→0.392, 0.30→0.584, 0.45→0.664. So the 15/30/45 % ladder is really a
**39 / 58 / 66 % param-reduction** ladder (three distinct operating points — kept as-is; going
below 0.15 is pointless, 0.05 and 0.10 both land ~29 %). The recovery+eval measures the
accuracy cost of each; if even the 39 % rung won't recover, that itself is the finding
(yolo11n is already compact).

**Verified** on the real donor under the 4 GB cap: prune 0.7 s @ 363 MB peak (was OOM),
split rel-err 4e-7, pruned forward @640 finite `(1,29,8400)`. New `tests/test_yolo_tp_prep.py`
(function-preservation in train+eval, param-count identity, train-mode inheritance, routing-attr
carry-over, C2PSA detection) + a `TRACE_IMGSZ` regression pin; 14 prune tests green. Kernel
re-launched. (The pre-fix launch-vs-push race that produced the *earlier* ModuleNotFoundError
is a separate, already-fixed issue — the `push.sh` HEAD≠upstream tripwire.)

## CP 3c.1 — dense-family scaling wave 1 landed (acct3); depth is a dead knob below yolo11n (2026-07-08)

**Ran:** all 6 WAVE1 candidates, 100 ep from scratch (no COCO pretrain), stock Ultralytics
recipe, seed 0, imgsz 640, on asilarnous47 (`data/kaggle_out_asilarnous47/dense_scaling/`; 6
row.json + 6 deploy ONNX + best.pt). mAP cross-checked against each `runs/<tag>/results.csv`.

**Finding — the depth multiplier does nothing sub-n.** The three width-0.25 candidates
(ctrl_n d=0.50, d25_w25 d=0.25, d33_w25 d=0.33) came out **byte-identical**: same 2,696,611
params AND same 0.8537 mAP to four decimals; likewise d33_w20 == d50_w20. The yamls carried the
correct distinct triples (verified), so this is architectural, not a bug: yolo11's depth mult
scales each C3k2 repeat by `n' = max(round(n·d), 1)`, but yolo11n's blocks already sit at base
`n ∈ {1,2}` — `round(2·0.5)=round(2·0.33)=round(2·0.25)=1`, and n=1 can't shrink — so every
`d ≤ 0.5` collapses to the same n=1 net. **You cannot make yolo11n shallower by scaling, only
narrower.** ⇒ the 6-point wave is really a **3-point width-only curve**, and wave-2 should drop
depth as a knob (or go supra-n, where the thesis budget doesn't reach).

**The width curve (single-seed, from scratch):**

| width | params | pose mAP50-95 |
|------|--------|------|
| 0.25 | 2,696,611 | 0.8537 |
| 0.20 | 1,904,069 | 0.8389 |
| 0.15 | 1,227,814 | 0.8147 |

**ctrl_n is the load-bearing control.** From-scratch yolo11n (its own 0.50/0.25 scale) =
**0.8537** vs the deployed COCO-pretrained + Ultralytics-recipe baseline **0.877** ⇒ pretrain +
recipe together are worth only **≈+0.023 mAP** on this gate task. That **re-frames winner-v1's
"deficit"**: the graft full-FT (v2topdown 0.846 / v3pan 0.842, both *also* from-scratch,
bare-AdamW) sits within **~0.8–1.2 pts of a from-scratch yolo11n at comparable params** — the
intrinsic *architecture* gap is small; most of the graft-vs-pretrained-baseline gap is the
missing COCO pretrain + the weaker recipe, now quantified. (Cross-family framing stays
user-owned; this just supplies the honest axis.)

**Owed before any pick (unchanged discipline):** the 6 deploy ONNX still need measured Nano
e2e latencies (the whole point of the dense arm — a device-native family whose *latency* may
beat the depthwise graft); and the 3 *distinct* width models must be de-noised at fresh seeds
{1,2,3} (the wave's 3 identical w0.25 runs are the SAME seed, so they confirm determinism, not
noise). Anchors are now baked into the report (`DENSE_ANCHORS`) for the CP 3c.3 figure.

## CP 6.2-B — pruned-baseline control ladder landed (acct2); non-monotonic ⇒ recovery-noise-bound (2026-07-08)

**Ran** (after the two-bug fix above): the DepGraph ladder on the gate-trained yolo11n donor —
prune → `reestimate_bn` → 50-ep bare-AdamW recovery → same-validator eval → deploy ONNX, at
15/30/45 % (`data/kaggle_out_asilarnous/prune_baseline/`; 3 ONNX + 3 .pt + report). 284 min,
clean run (per-epoch recovery prints monotone through 50/50, no divergence).

**Donor re-eval (same validator): 0.8869** mAP (note this is ~+0.01 over the 0.877 baseline
anchor used elsewhere — a validator-config difference; the ladder's Δ's are self-consistent
against THIS anchor, which is what the protocol requires).

| ratio | params | Δparams | pose mAP50-95 | Δ vs donor |
|------|--------|---------|------|------|
| 0.15 | 1,644,859 | −39 % | **0.8343** | −0.053 |
| 0.30 | 1,124,859 | −58 % | 0.7897 | −0.097 |
| 0.45 |   909,435 | −66 % | 0.8090 | −0.078 |

**The ladder is non-monotonic** — r45 (0.809) > r30 (0.790) despite pruning more. The recovery
ran to completion for every rung (log-verified), so this is **genuine single-seed recovery
variance**, not a bug: the 50-ep bare-AdamW recovery's noise is comparable to the inter-rung
accuracy gaps. Same lesson as CP 3.5 — **the single-seed ordering is untrustworthy; de-noise
the rungs at fresh seeds before reading any curve or picking an operating point** (CP 6.3).

**Standing:** these are ACCURACY only. The whole point of B1 is **latency** — a dense,
tensor-core-friendly pruned yolo11n may beat both the baseline and the depthwise graft e2e; the
3 ONNX are measured-only and await the Nano session. For the cross-family read: the best
single-seed rung (r15, 0.834, 39 % fewer params) lands just under the graft full-FTs
(0.842–0.846) and the from-scratch dense points (ctrl_n 0.854, w0.20 0.839) — all four families
now cluster in **0.79–0.85 from-scratch**, so latency, not accuracy, will separate them.

### All three parallel campaigns are now in (2026-07-08)

acct1 CP 5.3 (v2topdown 0.846 / v3pan 0.842) · acct2 CP 6.2-B (this) · acct3 CP 3c.1 (dense
width curve + ctrl_n control). **The single remaining blocker to the cross-family verdict is
one Nano bench session** covering: winner-v1.5 e2e (both necks), the 3 pruned-baseline ONNX,
the 6 dense-wave ONNX, plus the deferred riders (SE ablation, 512-res, FusedMBConv/OFA-R50 LUT
screens). Every pick (winner-v1.5, CP 6.3 operating point, CP 3c.2 wave-2, thesis framing)
stays user-owned and de-noise-gated.

## Cross-family locked-clock bench — the latency verdict (2026-07-08)

**One clean session, all families, mode 0 / 612 MHz, clocks locked, @640, batch 1** (models
saved under `models/`; per-model JSON in `data/e2e/`). fp32 is the reliable axis; **fp16 carries
±~20 % TRT-build variance** (autotuner kernel selection) — indicative only.

| family | model | mAP | fp32 ms | fp16 ms | vs baseline fp32 |
|---|---|---|---|---|---|
| baseline | yolo11n-pose | 0.877 | 12.74 | 7.75 | — |
| anchor | yolo11s-pose | 0.882 | 21.70 | 14.93 | +70 % |
| graft | winner-v1 no-neck | 0.841 | 17.67 | 12.38 | +39 % |
| graft | v2topdown | 0.846 | 18.15 | 12.58 | +42 % |
| graft | v3pan | 0.842 | 18.37 | 12.76 | +44 % |
| pruned | r15 (−39 %) | 0.834 | **9.54** | 5.93 | **−25 %** |
| pruned | r30 (−58 %) | 0.790 | **8.28** | 5.34 | **−35 %** |
| pruned | r45 (−66 %) | 0.809 | **7.94** | 7.18 | **−38 %** |
| dense | w0.25 (ctrl_n) | 0.854 | **11.33** | 8.11 | **−11 %** |
| dense | w0.20 | 0.839 | **11.26** | 6.93 | **−12 %** |
| dense | w0.15 | 0.815 | **9.53** | 6.30 | **−25 %** |

**Verdict.** Every dense/pruned model **beats** the baseline on latency; every graft **loses**
(the depthwise OFA backbone is memory-bound → 17–18 ms despite fewer params). Accuracy is nearly
flat across families (0.79–0.85 from-scratch), so **latency is the separator** — the dense-family
arm was the right call. Two standouts (faster + best accuracy in their region): **dense w0.25
(11.33 ms, 0.854)** and **pruned r15 (9.54 ms, 0.834)**. fp16: every model builds clean
(±~20 % build variance); the earlier "r15 fp16 FAIL" was another contention artifact — retried
on an idle board, r15 fp16 = 5.93 ms.

**Measurement incident (recorded so it doesn't recur).** The first bench pass was invalid: two
background bench batches (`bench_batch.sh`, `bench_all.sh`) plus a foreground run all hit the
single Jetson at once — GPU contention inflated numbers to 18–35 ms. A `pgrep` at a between-bench
gap mis-read the first batch as dead, which is why the second was launched. **Rule: exactly one
process may bench the Jetson at a time; never a background batch (they orphan and overlap) — run
foreground chunks.** The clean re-measurement confirms the Stage-0 numbers were fine all along
(baseline 12.74 now vs 12.75 then → the clocks *were* locked; the user's clock-lock worry is
answered).

## Round-2 frontier extension — pruning dominates scaling (2026-07-08)

The two round-2 Kaggle campaigns landed and were benched (same locked-clock session discipline,
one process at a time): **extended prune ladder** (r10/r20/r35/r55, on top of r15/r30/r45) and
**dense wave-2** (w13/w18/w22/w30, on top of w15/w20/w25). Full 7-rung × 7-width frontier is in
`models/README.md` + `data/e2e/`. New points (mAP · fp32 · fp16 ms):

| prune | mAP | fp32 | fp16 | | dense | mAP | fp32 | fp16 |
|---|---|---|---|---|---|---|---|---|
| r10 (1.9M) | 0.830 | 9.82 | 6.13 | | w13 (1.0M) | 0.813 | 9.56 | 6.45 |
| r20 (1.6M) | **0.838** | **9.52** | 5.91 | | w18 (1.6M) | 0.834 | 10.01 | 6.48 |
| r35 (1.1M) | 0.826 | 8.36 | 5.38 | | w22 (2.3M) | 0.845 | 11.55 | 6.95 |
| r55 (0.65M) | 0.798 | 7.66 | 5.07 | | w30 (3.9M) | 0.856 | 15.27 | — |

**Key finding — pruning Pareto-dominates dense scaling at matched accuracy.** At ~0.838: prune
r20 (9.52 ms) vs dense w20 (11.26 ms) → **15 % faster for the same mAP**; at 0.834 r15 (9.54) <
w18 (10.01). Pruned nets inherit the pretrained baseline weights; dense-scaled train from
scratch, so pruning owns the frontier below 0.85 and dense w25 owns just the top (0.854 @
11.33). **Pareto set:** dense w25 (0.854/11.33) › **prune r20 (0.838/9.52, −25 %)** › r35
(0.826/8.36) › r45 (0.809/7.94) › r55 (0.798/7.66). Standout = **prune r20**. w30 is dominated
(bigger *and* slower than baseline: supra-n width doesn't pay).

**Caveats.** The prune ladder is **noisy / non-monotonic** (r30 @ 0.790 is an outlier for its
1.1M params; r10 0.830 < r20 0.838 despite more params) → recovery variance, **de-noise at seeds
{1,2,3} owed before any CP 6.3 pick**. The dense accuracy curve is by contrast clean-monotonic in
width (0.813 → 0.856). fp16 remains ±20 % build-variance (and some dense fp16 builds are slow —
the autotuner, not a failure). Gap to the pretrained baseline's 0.877 stays ~0.02–0.04 (the
from-scratch/weak-recovery penalty) → the target for the Tier-2 distillation experiment (distil
prune r20 / a gentler rung against the 0.877 teacher).

## OFA-ResNet50 latency screen — D3 confirmed by measurement (2026-07-10)

User-directed feasibility gate (no checkpoint advance; current stays 5.3). The strategic
question "is it worth searching a *different* supernet?" reduces to exactly one candidate:
OFA-MBv3 is the only pretrained supernet we searched, and the **only other supernet that is
pretrained + samplable + dense** (tensor-core-friendly, unlike depthwise MBv3) is
**OFA-ResNet50**. Every other option is either depthwise (won't fix the Stage-0 memory-bound
loss) or needs training a supernet from scratch (~1,000+ GPU-h, free-tier-infeasible). D3
descoped R50 on a **roofline estimate** (~2× over budget); this session **measures** it.

**Method** (`expand/screen_r50.py`, `data/screen_r50/`, `models/screen_r50/`): export the
**min / median / max** corners of the standard `ofa_resnet50` design space as *backbones*
(classifier stripped, P3/P4/P5 taps at 80/40/20) to ONNX @640, then `bench_model` each on the
Nano — **mode 0 / 612 MHz, clocks locked, one process at a time**, TRT 10.3, fp32. TRT-fp32
latency is weight-value-independent, so random init is used (no pretrained checkpoint — a
legitimate shortcut for a latency-only screen). Legacy ONNX exporter forced (`dynamo=False`;
torch 2.11's default dynamo path needs onnxscript, absent in `.venv-nas`). `setup_jetson.sh`
before / `teardown_jetson.sh` after; config.local.yaml re-pointed to the LAN endpoint
192.168.1.150:22 for the session.

**Budget.** baseline yolo11n-pose = **12.75 ms** fp32 e2e; the graft's pose adapter+head offset
is **3.84 ms** (`data/pose_stem_head_offset.json`) — a *lower* bound for R50 (its adapter carries
~10× the channels) — so a backbone must land under **8.91 ms** for e2e ≤ baseline (Phase-3b
honest ceiling was 7.16 ms).

| corner | params | GFLOPs | backbone ms (fp32, std) | eff. TFLOP/s | e2e est. | ×baseline | ×bb-budget |
|---|---|---|---|---|---|---|---|
| **min** | 5.2M | 15.7 | **16.17** (±0.015) | 0.97 | ~20.0 | **1.57×** | **1.82×** |
| mid | 14.2M | 40.5 | 27.28 (±0.020) | 1.49 | ~31.1 | 2.44× | 3.06× |
| max | 46.1M | 122.3 | 58.11 (±0.050) | 2.10 | ~61.9 | 4.86× | 6.52× |

**Verdict — INFEASIBLE.** The *smallest* OFA-R50 backbone (16.17 ms) already **exceeds the whole
baseline e2e** (12.75 ms) — 1.82× the 8.91 ms backbone budget — *before* the pose head. The
entire elastic space (16→58 ms backbone) is over budget; **no subnet fits**, so a search has
nothing to find. The roofline estimate that descoped R50 is now measured, not assumed → **D3's
descoping is vindicated with a hard number.**

**Mechanism (the thesis-relevant nuance).** R50 is hardware-**efficient** — 0.97–2.10 effective
TFLOP/s, its dense 1×1/3×3 bottlenecks saturate the Orin tensor cores at **~2× the throughput of
yolo11n** (~0.5 TFLOP/s) and ~3–7× the MBv3 graft's ~0.30 — yet it loses anyway because it is too
**compute-heavy** (15.7 GFLOPs floor = 2.4× yolo11n's ~6.5). So the two cheaply-searchable
supernets fail for **opposite** reasons: **MBv3 is memory-bound** (Stage 0), **R50 is
compute-bound** (here). yolo11n sits in the sweet spot of that trade — which is precisely why
*compressing it* (prune/scale, the dense-family arm) beats grafting or re-searching a foreign
supernet. Memory is not the wall (peak 60–95 MiB on the 8 GB board); FLOPs are.

## Prune-the-graft latency screen — the "prune the OFA to reduce latency?" gate (2026-07-10)

Not a checkpoint (winner-refinement side-measurement; current stays CP 5.3). The user asked whether
structured pruning could drag the memory-bound OFA graft under the baseline. Screened the R50 way —
**latency first, no recovery training** (TRT latency is weight-value-independent; the channel COUNT
sets latency). `prune/screen_prune_graft.py` runs the CP 6.1 DepGraph harness
(`prune/prune_graft.py`) on the winner graft `d=[2,2,4,3,3]` at channel ratios 0.2/0.4/0.6, exports
each e2e ONNX @640 (added `prebuilt=` to `detect.export_grafted_onnx` so a pre-pruned graft exports
as-is), benched same-session on the Nano (mode 0, clocks locked, TRT 10.3, fp32+fp16).

| channel prune | param sparsity | fp32 ms | vs 12.75 | fp16 ms | vs 7.75 | fp16 FPS |
|---|---|---|---|---|---|---|
| unpruned | 0% | 17.69 | +38.7% | 12.38 | +59.7% | 81 |
| r=0.20 | 39% | 14.27 | +11.9% | 10.15 | +31.0% | 98 |
| r=0.40 | 64% | 11.81 | −7.4% | 8.41 | +8.5% | 119 |
| r=0.60 | 84% | 9.01 | −29.4% | 6.58 | −15.2% | 152 |

**Verdict.** Pruning reduces latency monotonically and the pruned e2e even beat the channel-linear
prior (the adapter + Pose-head internals prune too, and the compute-bound 1×1s drop ~width²). But
in **fp16 (the deployment precision) only r=0.60 — 84 % of params removed — beats the baseline**
(crossover ~r=0.48, ~73 % params); fp32 needs r≥0.40 (64 %). Every rung already clears 60 FPS (even
the unpruned graft at 81 FPS), so the deployment bar was never the issue — *beating the baseline* is,
and that demands catastrophic pruning of a net whose unpruned proxy is only 0.61 mAP. The dense
family already beats the baseline latency (`prune_base_*` fp16 < 7.75 ms) at 0.83+ mAP, so the
pruned graft is **accuracy-dominated by the dense arm**. Open: CP 6.2 recovery-train r=0.60 to
confirm the accuracy loss (prior: loses). Artifacts: `prune/screen_prune_graft.py`,
`models/screen_prune_graft/README.md` + `screen_prune_graft_result.json`, `data/e2e/graft_prune_r*`.
The "prune the OFA?" gate now joins the R50 gate: both cheaply-searchable levers on the graft are
closed by measurement; the win stays with compressing the dense yolo11.

## CP 6.2-G CLOSED — pruned-graft recovery: retention is real, dominance is not (2026-07-11)

**The accuracy half of the prune-the-graft screen** (its open question: does an 84 %-pruned
graft retain usable accuracy? prior: craters). `prune/recover_graft.py` (48dc6ad) built the
winner graft fresh per point (OFA-ImageNet backbone + warm gate head, all trainable), pruned
with the SAME CP 6.1 DepGraph harness the latency screen used (identical channel outcome —
r60 params match the screen exactly: 493,971), BN-re-estimated, then trained to capacity with
the repo's bare-AdamW loop (100 ep, seed 0) — **prune-then-TRAIN**, the `dense_scaling`-
comparable protocol. One ratio per T4 (Kaggle acct1, `tfm-nas-cp3-3-search` v23). Latencies are
the screen's measured numbers (TRT latency is weight-value-independent; params match ⇒ same graph).

| rung | params | sparsity | mAP50-95 | mAP50 | fp32 ms | fp16 ms |
|---|---|---|---|---|---|---|
| unpruned (anchor, full-FT) | 3.00M | — | 0.841 | — | 17.69 | 12.38 |
| r40 | 1.07M | 64.3 % | **0.8163** | 0.9218 | 11.81 | 8.41 |
| r60 | 0.49M | 83.6 % | **0.7589** | 0.8958 | 9.01 | 6.58 |

**Reading.**
1. **Retention beats the prior by a mile**: −2.5 pts at 64 % params gone, −8.2 pts at 84 %,
   mAP50 ≥ 0.896 everywhere. No accuracy cliff — the narrow-MBConv regime is *viable*.
2. **Accuracy-per-param is mid-pack**: r40 (0.816 @ 1.07M) edges dense w13 (0.813 @ 1.0M) even
   though the graft trained bare-AdamW while the dense wave got the stock Ultralytics recipe.
   The graft's deficit was never capacity; it is **ms-per-param** (memory-bound depthwise).
3. **Against the measured frontier both rungs are strictly dominated** (models/README.md, the
   2026-07-08 cross-family bench): r60 (0.759 @ 9.01/6.58) loses to prune r55 (0.798 @
   7.66/5.07), r35 (0.826 @ 8.36/5.38) and r20 (0.838 @ 9.52/5.91) on BOTH axes; r40 (0.816 @
   11.81/8.41) loses to r35 and w18. Gap to the front at matched latency ≈ **+7–8 pts** — beyond
   any plausible recovery-technique stack (+3–5, see the program's literature scan). **The
   screen's prior is confirmed: pruned to latency-parity, the graft is accuracy-dominated by
   the dense arm.**
4. **Floor-config caveat (lower bound, not a death certificate)**: these rungs used the weakest
   pruning configuration on every axis — uniform per-layer ratios (no global allocation),
   magnitude importance, one-shot at extreme sparsity, bare-AdamW recovery, no KD.

**Correction to the session briefing (recorded for honesty).** The briefing that framed this
result told the user the dense/prune latencies were "unbenched — pending THE Nano session".
Stale: the cross-family bench landed 2026-07-08 ("Round-2 frontier extension" above); the
frontier comparison here uses those measured numbers, and the expected-outcome statement in the
approved plan ("photo-finish at ~7.5 ms") was too rosy — the measured front sits at 5.1–5.9 ms
fp16 in the graft rungs' accuracy band. Remaining Nano work is only: deferred riders (SE /
512-res / FusedMBConv screen), optional recovered-r60 sanity bench, and future
technique-champion verifications.

**Program decision (user, AskUserQuestion 2026-07-11): PRUNING-AS-SEARCH** — chosen over
"consolidate & finish" and "full width re-search now" after a briefing + literature scan (HALP
latency-saliency knapsack, NeurIPS'22 · Pruning-as-Search, IJCAI'22 · AutoSlim · MetaPruning ·
Joint-DetNAS · prune+channel-KD as the standard edge-YOLO recipe · EfficientNetV2's Fused-MBConv
as the architecture-side fix for exactly our memory-bound diagnosis). Pruning is promoted to the
**second stage of the hardware-aware search** (latency-guided width search, driven by the same
LUT + measured-e2e machinery as stage 1), applied **symmetrically** to both families:
- **Track 0** (this entry + plumbing commits): technique knobs in `prune/prune_graft.py`
  (`global_pruning` / Taylor importance / `iterative_steps` / `pruning_ratio_dict`);
  `--technique --seed --candidates-json/--index` in `prune/recover_graft.py`; `--seed` +
  technique passthrough in `prune/prune_baseline.py`, `--seed` in `search/dense_family.py`;
  kaggle `PG_*` config extension.
- **Track 2**: `search/latency_model.py` — per-precision e2e latency surrogate fitted on the
  ~30 measured points in `data/e2e/` (replaces the 640-inverted additivity as the width-search
  latency oracle; **ranking-only**, on-device verification for any claim).
- **Track 3** (Kaggle): graft technique ladder — r50 knee rung, global_l2 vs global_taylor,
  iterative, then **HALP-lite** (`prune/allocate.py`: per-STAGE knapsack on latency-saved per
  saliency-lost, stage latencies from the @640 LUT rows + measured adapter/head share) at
  7.5 / 6.6 ms fp16 targets. Also produces the **G1 probe**: fallback topologies idx3 / idx11
  (the Stage-0-benched pair) pruned to matched latency — does topology RE-RANK under width?
- **Track 4** (Kaggle): fairness (best technique on the prune-baseline champion), **KD recovery**
  on both families' champions (teacher = gate donor 0.887; pulls Phase 8 forward),
  recipe-parity-lite (SGD+EMA+cosine in the bare loop), **de-noise at seeds {1,2,3}** (the prune
  ladder is non-monotonic ⇒ owed before ANY pick), and **G2**: 5-ep warm-head proxy fidelity on
  ~10 pruned nets vs their accumulated 100-ep oracles (pass = Spearman ρ≥0.70 AND top-1 regret
  ≤0.01 — the CP 2.4 bar).
- **Track 5 (GATED on G1 ∧ G2)**: the width-aware (ks,e,d,r) BO re-search — the
  literature-clean version of "prune the supernet before searching" (no supernet retraining:
  prune-derived candidates + the Track-2 latency oracle). Either gate failing = the measured
  answer that the joint search is unnecessary (decomposition) or unrankable (no proxy signal),
  recorded as a finding. User briefing before any Track-5 spend.

**Honest expectation, restated post-correction**: the graft likely stays off the front even with
the full technique stack; the program's value = the hardware-aware pruning method demonstrated
cross-family (its gains accrue to the DENSE champion too — prune r20 + KD is models/README's own
"clearest shot at Pareto-dominance"), the two measured gates closing the "prune the supernet?"
question, and the thesis-structural claim (NAS = stage-1 topology + stage-2 width, both
latency-driven — the search is part of the deployed network's production either way).

Artifacts: `data/cp33_kaggle_out/recover_graft_r{40,60}*` (gitignored), `models/graft_pruned/`
(README + result JSON tracked, binaries staged), frontier rows added to `models/README.md`.
No checkpoint-number advance (current stays 5.3; recorded like CP 6.2-B).

## CP 6.2-T Wave A — technique ladder round 1: global_taylor wins both rungs (2026-07-11)

First round of the pruning-as-search technique ladder (procedure "CP 6.2-G CLOSED", Track 3):
the winner graft pruned at r50/r60 under three allocation/importance configs, prune-then-TRAIN
100 ep bare-AdamW seed 0, one technique per Kaggle account (kernels v24/v19/v9), same protocol
as the CP 6.2-G rungs (which serve as the uniform-r40/r60 anchors).

| technique | r50 mAP (params) | r60 mAP (params) |
|---|---|---|
| uniform (floor) | 0.7883 (775K) | 0.7589 (494K) |
| global_l2 | 0.7758 (725K) | 0.7305 (424K) |
| **global_taylor** | **0.7947** (764K) | **0.7773** (443K) |

**Reading.**
1. **global_taylor wins both rungs and its margin GROWS with sparsity** (+0.6 pts at r50 →
   +1.8 at r60 vs uniform) — allocation+selection matter most where damage is extreme, exactly
   the literature's claim. At r60 it also keeps FEWER params than uniform (443K vs 494K) while
   scoring +1.8: strictly better selection, not more capacity.
2. **global_l2 is WORSE than uniform at both rungs** (−1.3 / −2.8): global *magnitude* ranking
   is scale-confounded across layers (raw L2 norms aren't comparable layer-to-layer without
   gradient weighting) — a useful negative: "global" alone is not the upgrade, saliency is.
3. **Latency caveat**: the global variants carry different per-layer channel profiles than the
   screened uniform rungs, so 7.4/6.58 ms do NOT transfer — technique champions are benched on
   the Nano before any frontier claim (TRT latency is weight-independent; one bench per graph).

**Launched same-session (all three accounts RUNNING):** A6 = iterative×3 @ global_taylor
r50+r60 (acct1 v25); Wave B = both HALP-lite specs (acct2 v20 — allocations
[0.7,0.5,0.1,0,0]+rest0.7 → pred 10.22 ms fp32, and [0.7,0.7,0.3,0.1,0.1]+rest0.7 → pred
8.86 ms fp32; the knapsack guts early/high-res stages + head and protects late stages — the
EfficientNetV2 principle rediscovered from our own @640 LUT rows); Wave C = fallback idx3
probe @ global_taylor r50+r60 (acct3 v10; idx11 takes the next free slot → gate G1).

## CP 6.4 Wave B + Wave C (idx3 half) — HALP-lite leads the ladder (2026-07-11)

**Wave B — the LUT-knapsack allocations, trained** (kernel acct2 v20; same prune-then-TRAIN
100 ep protocol; latencies are the spec's PREDICTIONS until the Nano bench — the linear
per-stage model under-credits savings, so measured should land at-or-below):

| point (allocation) | params | mAP50-95 | mAP50 | pred fp32 ms (fp16 est) |
|---|---|---|---|---|
| halp_fp32_10p4 ([.7,.5,.1,0,0]+rest .7) | 2.36M | **0.8042** | 0.903 | 10.22 (7.16) |
| halp_fp32_9p0 ([.7,.7,.3,.1,.1]+rest .7) | 1.82M | **0.7936** | 0.911 | 8.86 (6.20) |

**Latency-matched comparison — the program's central claim, provisionally confirmed:**
halp_9p0 (0.7936 @ pred 8.86 fp32) vs uniform r60 (0.7589 @ MEASURED 9.01) = **+3.5 pts at
matched latency**; vs global_taylor r60 (0.7773) = +1.6. The knapsack's "gut early stages +
head, keep late stages" allocation retains 3.7× the params of uniform-r60 at the same
latency class — capacity survives where the ms don't live. **Latency-aware allocation >
saliency-only allocation > uniform > global-magnitude.** Pending: Nano verification of the
two predicted latencies (weight-independent; one bench per graph).

**Wave C first half — idx3 probe (G1)** (acct3 v10): idx3@r50_gtay 0.7749 (548K),
idx3@r60_gtay 0.7478 (323K) — the winner topology beats idx3 at both rungs (+2.0/+3.0), and
idx3 at matched *ratio* is a smaller net (its unpruned e2e was already faster: 16.11 vs
17.67 fp32), so the matched-LATENCY read needs the bench/surrogate — but nowhere does idx3
overtake. **Preliminary G1: NO topology re-ranking under width** (idx11 = second half,
launched acct2 v21). Also launched: HALP de-noise both specs @ seed 1 (acct3 v11); A6
iterative still running (acct1 v25).

## Technique table CLOSED (iterative negative) + G1 VERDICT: no re-ranking → Track 5 closed (2026-07-11)

**A6 — iterative ×3 @ global_taylor** (acct1 v25): r50 0.7827 (828K), r60 0.7540 (654K) —
**worse than one-shot** at both rungs (−1.2 / −2.3). Reading: the literature's iterative
advantage assumes pruning a TRAINED net (prune→recover cycles); in this arm's prune-then-TRAIN
protocol the backbone is only ImageNet-pretrained at prune time, so interleaved 5-epoch
recoveries bias Taylor's gradients toward a half-trained state while the total budget stays
100 ep. Final technique ordering on the graft, at matched rungs:
**HALP-lite (latency-aware) > global_taylor (saliency) > uniform > iterative > global_l2.**

**Wave C second half — idx11 probe** (acct2 v21): idx11@r50_gtay 0.7816 (608K), idx11@r60_gtay
0.7543 (364K). Neither fallback overtakes the winner topology at matched ratio (winner +1.3 to
+3.0); the one latency-matched near-crossing (idx11-r50 0.7816 vs winner-r60 0.7773, ~+0.4) is
inside the seed tie-band, and the HALP allocation on the winner (0.8042 / 0.7936) dominates the
entire probe region regardless. **G1 = NO topology re-ranking under width → the axes decompose
→ Track 5 (width-aware joint re-search, the literature-clean "prune the supernet before
searching") is CLOSED by measurement, not argument.** G2 is thereby moot for the search
decision (gate was G1 ∧ G2); it stays optional thesis color only.

**HALP de-noise, seed 1** (acct3 v11): halp_10p4 0.8082 (seed0 0.8042 → mean 0.8062 ± 0.002),
halp_9p0 0.7926 (seed0 0.7936 → mean 0.7931 ± 0.0005). Champion band is stable so far; seed 2
launched (acct2 v22). Fairness leg launched (acct1 v26): global_taylor on the prune-baseline
donor at r20+r35 — the best graft technique applied to the frontier-owning family.
