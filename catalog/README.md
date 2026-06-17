# `catalog/`

The **block registry and LUT schema** for the whole project. Every neural
building block the NAS can search over is defined here exactly once, together
with the *configuration grid* that becomes the rows of the Jetson latency LUT.
Both the measurement pipeline (`lut/`) and every search phase (`search/`, and the
future `net2net/`, `expand/`, `distill/`) import this package, so they can never
disagree about what a block *is* or how a measured row is *keyed*.

> **One sentence to remember:** a block's config is a plain `dict`, and that dict
> — serialized — *is* the LUT's primary key. Change the dict's shape or value
> *types* and you silently re-key the append-only LUT.

---

## Role in the project

```
            ┌────────────────────────── catalog/ ──────────────────────────┐
            │  block modules (mbconv.py, seg_det.py)                        │
            │  + registry (blocks.py)  + cfg grids                          │
            │  + row_key/sweep (sweep.py)  + OFA topology (ofa_mbv3.py)     │
            └───────────────┬───────────────────────────┬──────────────────┘
                            │                           │
              builds ONNX, measures                walks an arch_dict into
              every grid cfg → LUT row             the same ("mbconv", cfg) rows
                            │                           │
                     lut/  ▼                     search/ ▼
              run_sweep / gen_dummy_lut        arch_to_blocks → cost.py (CP 2.2)
```

- **Single source of truth.** Adding or widening a block happens in exactly one
  place (this package), and every consumer picks it up automatically.
- **Append-only LUT discipline.** Each grid cfg maps to a stable `row_key`; new
  blocks or wider grids never invalidate existing measured rows. See
  [`lut/docs/schema.md`](../lut/docs/schema.md) and the project plan in
  [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).
- **Two-venv safe.** Only `contracts.py` is import-free of third-party packages
  (so it is shared by both `.venv` and `.venv-nas`); the rest of the package needs
  `torch`.

### Quick start

```python
from catalog.blocks import build_block, input_shape_for
from catalog.sweep import iter_sweep, row_key, sweep_size

print(sweep_size())                          # 2710  (full catalog)

name, cfg, shape, key = next(iter(iter_sweep()))
module = build_block(name, cfg).eval()       # an nn.Module
assert shape == input_shape_for(name, cfg)
assert key   == row_key(name, cfg, shape)    # 16-hex LUT key
```

---

## Module map

| File | What it holds | Public surface |
|---|---|---|
| `__init__.py` | Curated re-exports | `BLOCK_REGISTRY`, `build_block`, `input_shape_for`, `count_params`, `iter_sweep`, `row_key`, `sweep_size` |
| `blocks.py` | The registry: builders, input-shape fns, and the cfg **grids**; the `BLOCK_REGISTRY` dict | `build_block`, `input_shape_for`, `count_params`, `BLOCK_REGISTRY` |
| `mbconv.py` | Classification building blocks (MobileNet/EfficientNet family) | `Conv3x3`, `Conv1x1`, `DWConv`, `SEBlock`, `MBConv`, `Skip` |
| `seg_det.py` | Segmentation/detection building blocks | `Upsample`, `Deconv`, `PixelShuffleBlock`, `FPNLateral`, `FPNTopDown`, `DilatedConv3x3`, `ASPP`, `SegHead`, `DetHead` |
| `ofa_mbv3.py` | OFA-MobileNetV3-w1.0 macro-topology — the shared truth | `KS`, `E`, `D`, `MAX_DEPTH`, `STEM_RES`, `FIRST_BLOCK`, `STAGES`, `stage_in_c`, `reachable_mbconv_configs` |
| `sweep.py` | `row_key` derivation + sweep enumeration | `row_key`, `iter_sweep`, `sweep_size` |
| `flops.py` | Hook-based FLOPs counter (shared by real + dummy LUT) | `count_flops`, `count_flops_forward` |
| `shapes.py` | Canonical constants (`BATCH=1`, reference res/channel lists) | `RESOLUTIONS`, `CHANNELS`, `BATCH` |
| `contracts.py` | `TypedDict` contracts for cfgs / arch / LUT rows / cost | `MBConvCfg`, `ArchDict`, `LatencyStats`, `LutRow`, `Block`, `CostDict`, `CostOffset` |

> `count_flops`, the `contracts` TypedDicts, and the `ofa_mbv3` helpers are
> imported from their submodules directly — they are intentionally *not* in the
> `__init__` surface.

---

## Anatomy of a registry entry

`BLOCK_REGISTRY` (in `blocks.py`) maps each block name to a three-field spec:

```python
"mbconv": {
    "builder":     _b_mbconv,   # cfg -> nn.Module
    "input_shape": _in,         # cfg -> (B, C, H, W) for ONNX export
    "grid":        _MBCONV_GRID # list[cfg dict] to sweep
},
```

Grids are built with the tiny helper `_grid(**axes)`, which takes the **Cartesian
product** of the named axes:

```python
_DWCONV_GRID = _grid(in_c=[16, 32, 64, 96, 160], kernel=[3, 5, 7],
                     stride=[1, 2], res=[56, 28, 14])   # 5*3*2*3 = 90 cfgs
```

**The schema-uniformity invariant.** Every cfg in a block's grid carries *exactly*
the keys that block's `builder` and `input_shape` function read — no more, no less.
This is what keeps build → export → key-derivation in lockstep, and it is enforced
by `tests/test_catalog.py::test_grid_cfgs_share_one_schema`. Two input-shape
helpers cover all blocks:

- `_in(cfg)         -> (BATCH, cfg["in_c"], cfg["res"], cfg["res"])`
- `_in_single(cfg)  -> (BATCH, cfg["c"],    cfg["res"], cfg["res"])`  (for the
  single-channel blocks `fpn_topdown`, `dilconv`)

---

## Block catalog (15 blocks)

### Classification family — `mbconv.py`

| Block | Module | cfg keys | Grid | What it is |
|---|---|---|---|---|
| `conv3x3` | `Conv3x3` | `in_c, out_c, stride, res` | 200 | 3×3 Conv-BN-ReLU6, "same" padding; `stride` downsamples |
| `conv1x1` | `Conv1x1` | `in_c, out_c, res` | 144 | Pointwise 1×1 Conv-BN-ReLU6 |
| `dwconv` | `DWConv` | `in_c, kernel, stride, res` | 90 | Depthwise k×k conv (`groups=in_c`), channels preserved |
| `se` | `SEBlock` | `in_c, res` | 12 | Squeeze-Excite: GAP → 1×1 (÷4) → ReLU → 1×1 → Hardsigmoid → scale |
| `skip` | `Skip` | `in_c, res` | 15 | Identity — modeled as a block so the LUT has a cost for "choose skip" |
| `mbconv` | `MBConv` | `in_c, out_c, kernel, stride, expand, se, res` | **2107** | Inverted residual: (1×1 expand if `expand≠1`) → DW k×k → optional SE → 1×1 project; residual add when `stride==1 and in_c==out_c` |

### Segmentation / detection family — `seg_det.py`

| Block | Module | cfg keys | Grid | What it is |
|---|---|---|---|---|
| `upsample` | `Upsample` | `in_c, scale, mode, res` | 32 | `F.interpolate` (`nearest`/`bilinear`), parameter-free; out res = `res*scale` |
| `deconv` | `Deconv` | `in_c, out_c, res` | 18 | `ConvTranspose2d` k4 s2 p1 (exact 2× up) + BN + ReLU6 |
| `pixelshuffle` | `PixelShuffleBlock` | `in_c, out_c, upscale, res` | 12 | 3×3 conv → `PixelShuffle(upscale)` → BN → ReLU6; out res = `res*upscale` |
| `fpn_lateral` | `FPNLateral` | `in_c, out_c, res` | 32 | 1×1 conv that maps a backbone tap into the FPN channel count |
| `fpn_topdown` | `FPNTopDown` | `c, res` | 6 | Nearest 2× upsample + 3×3 conv (the skip-add is free, so omitted) |
| `dilconv` | `DilatedConv3x3` | `c, dilation, res` | 24 | 3×3 dilated Conv-BN-ReLU6, resolution preserved |
| `aspp` | `ASPP` | `in_c, out_c, res` | 4 | DeepLab ASPP: 1×1 + three atrous branches (rates 6/12/18) + image-pool, concatenated then projected |
| `seghead` | `SegHead` | `in_c, num_classes, res` | 8 | 3×3 Conv-BN-ReLU6 → Dropout2d(0.1) → 1×1 classifier |
| `dethead` | `DetHead` | `in_c, num_classes, num_anchors, res` | 6 | YOLO-style coupled head: two 3×3 Conv-BN-ReLU6 → 1×1 to `(5+num_classes)*num_anchors` |

> **Why `dethead` resolutions are `[40, 20, 10]`** and not the 224→7 classification
> pyramid: detection heads attach to feature-pyramid levels (a 640-input detector
> tapped at strides 16/32/64), so their input resolutions are intentionally
> different from the backbone sweep.

---

## The OFA-MobileNetV3 topology — `ofa_mbv3.py`

This module encodes the **fixed macro-structure** of the pretrained
`ofa_mbv3_d234_e346_k357_w1.0` supernet, verified against the installed `ofa`
package at `width_mult=1.0`:

```
base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
stride_stages    = [1, 2, 2, 2, 1, 2]
se_stages        = [False, False, True, False, True, True]
```

The first `16` is the **stem**; the second `16` is the **fixed first block**; the
five **searchable stages** output `[24, 40, 80, 112, 160]`; the trailing
`960, 1280` are the head (not searched).

**Elastic choice sets** (the `d234_e346_k357` in the checkpoint name):

| Symbol | Values | Meaning |
|---|---|---|
| `KS` | `[3, 5, 7]` | depthwise kernel sizes |
| `E` | `[3, 4, 6]` | expansion ratios |
| `D` | `[2, 3, 4]` | per-stage active depths |
| `MAX_DEPTH` | `4` | slots per stage (`ks`/`e` are length `5 * MAX_DEPTH = 20`) |
| `STEM_RES` | `112` | resolution after the 3→16 stride-2 stem (224→112) |

**The five searchable stages** (`STAGES`):

| Stage | out_c | stride | SE | res_in |
|---|---|---|---|---|
| 0 | 24 | 2 | ✗ | 112 |
| 1 | 40 | 2 | ✓ | 56 |
| 2 | 80 | 2 | ✗ | 28 |
| 3 | 112 | 1 | ✓ | 14 |
| 4 | 160 | 2 | ✓ | 14 |

`FIRST_BLOCK` is fixed: `16→16, k3, s1, expand=1, no SE, res=112`. (Because
`expand=1`, OFA skips the inverted-bottleneck's 1×1 expansion conv — and the
catalog's `MBConv` does the same via `if expand != 1`, so the structures match.)

### `reachable_mbconv_configs()` → 91

Every MBConv cfg the search space can actually produce: one fixed first block, plus
per stage an **entry** block (`prev_w → out_w` at the stage stride, run at `res_in`)
and a **repeat** block (`out_w → out_w`, stride 1, run at `res_in // stride`), each
over `KS × E`:

```
1 + 5 stages × 2 (entry, repeat) × |KS| × |E|  =  1 + 5·2·3·3  =  91
```

These 91 are **unioned into the MBConv grid** (`blocks.py`), so that the search
walking an `arch_dict` via `search/arch_to_blocks.py` — which uses *this same*
`STAGES`/`FIRST_BLOCK`/`stage_in_c` table — only ever emits blocks that have a LUT
row. The shared table is the guarantee that the search space and the measured table
can never drift apart.

---

## The `row_key` contract — `sweep.py`

Every LUT row is identified by a 16-hex SHA-1 of the block name, its cfg, and its
input shape:

```python
row_key = sha1(json.dumps({"b": block, "c": cfg, "s": list(shape)},
                          sort_keys=True).encode()).hexdigest()[:16]
```

Three properties this gives us:

1. **Order-independent.** `sort_keys=True` makes the hash invariant to dict
   insertion order (`tests/test_row_key.py::test_cfg_insertion_order_is_irrelevant`).
2. **Precision is *not* in the key.** Rows measured at different precisions coexist
   under the same `row_key`; always filter to one precision before keying rows in
   memory (`lut/loader.py::load_lut` does this and raises on collisions). This is
   why resuming under a new precision re-measures instead of skipping.
3. **Value *types* are load-bearing.** `se=False` serializes to `false`; `se=0`
   serializes to `0` — different JSON, different key. Code that builds cfgs destined
   for `row_key` must preserve exact Python types. The tripwire
   `tests/test_row_key.py::test_bool_vs_int_changes_the_hash` guards this, and it is
   the reason `contracts.py` uses `TypedDict` (plain dicts) rather than dataclasses.

> ⚠️ The golden hashes in `tests/test_row_key.py` **pin this contract**. A change
> that re-keys rows fails them by design — never update a golden without recording
> the decision in `procedure.md` (a re-key orphans every measured Jetson row).

---

## Sweep enumeration — `sweep.py`

```python
iter_sweep(only_blocks=None)  # yields (block_name, cfg, input_shape, row_key)
sweep_size(only_blocks=None)  # number of rows
```

The full catalog enumerates **2710 globally-unique rows**:

| Block | Rows | Block | Rows | Block | Rows |
|---|---|---|---|---|---|
| mbconv | 2107 | conv3x3 | 200 | conv1x1 | 144 |
| dwconv | 90 | fpn_lateral | 32 | upsample | 32 |
| dilconv | 24 | deconv | 18 | skip | 15 |
| se | 12 | pixelshuffle | 12 | seghead | 8 |
| dethead | 6 | fpn_topdown | 6 | aspp | 4 |
| | | | | **Total** | **2710** |

(`mbconv` = 2016 from the generic grid after the `out_c ≥ in_c` prune, plus the 91
OFA-reachable configs unioned in.) The same iterator drives both the real sweep
(`lut.orchestrate.run_sweep`) and the dummy generator
(`lut.orchestrate.gen_dummy_lut`), so they enumerate identical rows.

---

## FLOPs counter — `flops.py`

A small forward-hook counter for Conv2d / ConvTranspose2d / Linear multiply-adds:

| Layer | Counted FLOPs |
|---|---|
| `Conv2d` | `2 · out_c · (in_c / groups) · kh · kw · oh · ow` |
| `ConvTranspose2d` | `2 · out_c · in_c · kh · kw · oh · ow / groups` |
| `Linear` | `2 · in_features · out_features` |

```python
from catalog.flops import count_flops
flops = count_flops(module.eval(), (1, 16, 112, 112))
```

- **Pass an `eval()`-mode module** — the zeros-forward would otherwise update
  BatchNorm running stats.
- `count_flops_forward` additionally returns the output tensor, so callers (e.g.
  `gen_dummy_lut`'s IO-byte sizing) can avoid a second forward.
- It is a **static estimate** (BN / activations / pooling / elementwise are *not*
  counted) — useful as a predictive-model feature, never a deployment guarantee.
- Hand-computed goldens pin the arithmetic in `tests/test_flops_golden.py`, and the
  real and dummy paths share this one counter by construction.

---

## Type contracts — `contracts.py`

`TypedDict`s describing the dicts that flow through the pipeline:

| Type | Describes |
|---|---|
| `MBConvCfg` | one mbconv grid entry / one searchable OFA block position |
| `ArchDict` | a canonical OFA arch spec: `ks`/`e` length `5·MAX_DEPTH`, `d` length 5 |
| `LatencyStats` | `{mean, std, p50, p95, n}` |
| `LutRow` | one measured (or dummy) LUT row — mirrors `lut/docs/schema.md` |
| `Block` | `tuple[str, dict, tuple]` — a LUT-keyed block from `arch_to_blocks` |
| `CostDict` | a whole subnet's composed cost (CP 2.2) |
| `CostOffset` | the constant stem+head delta added to every subnet's cost |

**Why `TypedDict`, not `dataclass`:** the runtime representation must stay a plain
`dict`, because `row_key` is `sha1(json.dumps(...))` over those dicts — changing the
representation (or a value's type) silently re-keys the append-only LUT. Type
checkers see the contract; the wire format never changes. The module also keeps
itself free of third-party imports so it is safe to import from either venv.

> **Cost aggregation is heterogeneous** (see `search/cost.py`): `latency_ms`,
> `params`, and `flops` are **summed** across a subnet's blocks, but `peak_mem_mib`
> is the **max** — blocks run one at a time and free their scratch, so summing would
> massively overestimate memory (`lut/docs/schema.md`).

---

## Known simplifications (by design, not bugs)

1. **Blocks are latency *proxies*, not bit-exact OFA-MBv3 replicas.** They use
   `ReLU6` as the main activation (real MobileNetV3 uses h-swish in later stages)
   and SE gates with `Hardsigmoid` at a fixed ÷4 reduction (no MobileNetV3
   make-divisible-to-8 rounding). This is acceptable for a latency LUT — timing is
   dominated by tensor geometry, not activation identity — and the residual error
   between summed-LUT and a measured full subnet is exactly what
   `search/validate_additivity.py` is there to bound.
2. **`shapes.py`'s `RESOLUTIONS`/`CHANNELS` are reference constants, not the source
   of truth.** Only `BATCH` is imported by the registry; each grid hardcodes its own
   res/channel lists, so editing those two lists does **not** widen the sweep.
3. **`peak_mem_mib` is per-block** (TRT scratch + IO, excluding weights) and is
   **never summed** across blocks — compose deployable memory as
   `sum(weights) + max(working set)` via `search.cost.resident_mem_mib`.

---

## Extending the catalog

**Add a new block:**
1. Implement the `nn.Module` in `mbconv.py` or `seg_det.py`.
2. Add a `_b_<name>(cfg)` builder and (if its keys differ) an input-shape fn in
   `blocks.py`.
3. Add a `_<NAME>_GRID = _grid(...)` — keep it narrow; every cfg is a LUT row.
4. Register the three-field entry in `BLOCK_REGISTRY`.
5. Update the pinned counts in `tests/test_catalog.py` in the *same* commit, then
   regenerate/extend the LUT.

**Widen an existing grid:** append-only — adding axis values is safe (existing rows
keep their keys); update the pinned count and re-sweep. **Never edit an existing
cfg in place** — that re-keys measured rows (the no-orphan tripwire
`tests/test_lut_keydrift.py` will catch it).

---

## Tests & verification

| Test | Pins |
|---|---|
| `tests/test_catalog.py` | grid sizes (2710 / 2107 / 91), schema uniformity, global key uniqueness, corner-cfg build+forward |
| `tests/test_row_key.py` | the row_key golden hashes + bool-vs-int / order / format invariants |
| `tests/test_flops_golden.py` | hand-computed FLOPs for the shared counter |
| `tests/test_lut_keydrift.py` | no on-disk LUT row is orphaned by a catalog change |

```bash
bash scripts/check.sh                 # ruff + mypy + pytest (uses .venv)
bash scripts/check.sh -m "not slow"   # fast lane
```

### Audit status

**Verified correct as of 2026-06-16.** The 49 pinned catalog/row_key/flops tests
pass, *and* an exhaustive build+forward over **all 2710 cfgs** produces the
structurally-expected output shape for every one (2710/2710 — channel counts and
stride/upsample spatial arithmetic asserted by shape equality, not just no-crash).
15 blocks, 91 OFA-reachable cfgs; no defects, the 3 simplifications above are
intentional. Reproduce the exhaustive check with:

```bash
PYTHONPATH= .venv/bin/python -m pytest \
  tests/test_catalog.py tests/test_row_key.py tests/test_flops_golden.py
```
