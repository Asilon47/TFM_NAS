"""OFA-MobileNetV3-w1.0 macro-topology — single source of truth.

Encodes the fixed structure of the pretrained ``ofa_mbv3_d234_e346_k357_w1.0``
supernet so two consumers agree on it without duplicating the table:

- ``catalog.blocks`` unions :func:`reachable_mbconv_configs` into the MBConv LUT
  grid, so every block the search can request has a measurable LUT row.
- ``search.arch_to_blocks`` walks an ``arch_dict`` into an ordered block list
  using the same stage table.

Topology verified against the installed package
``ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py`` at
``width_mult=1.0``::

    base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    stride_stages    = [1, 2, 2, 2, 1, 2]   # first_block, then 5 stages
    se_stages        = [False, False, True, False, True, True]

The first ``16`` is the stem; the second ``16`` is the fixed first block; the
five searchable stages output ``[24, 40, 80, 112, 160]``. The trailing
``960, 1280`` are the head and are not part of the searchable backbone.
"""

# Elastic choice sets — the "d234_e346_k357" encoded in the checkpoint name.
KS = [3, 5, 7]          # kernel sizes
E = [3, 4, 6]           # expand ratios
D = [2, 3, 4]           # per-stage active depths
MAX_DEPTH = 4           # ks/e are length 5 * MAX_DEPTH = 20 (one slot per block)

STEM_RES = 112          # resolution after the 3->16 stride-2 stem (224 -> 112)

# The fixed, non-elastic first block (``blocks[0]``): 16->16, k3, s1, no
# expansion, no SE. expand=1 means OFA skips the inverted bottleneck's 1x1
# expansion conv — and the catalog's MBConv does the same (see
# catalog/mbconv.py, `if expand != 1`), so the two structures match exactly.
FIRST_BLOCK = {
    "in_c": 16, "out_c": 16, "kernel": 3, "stride": 1,
    "expand": 1, "se": False, "res": STEM_RES,
}

# The five searchable stages, in order. For each: output width, the stride
# applied at the stage's *first* block, SE on/off (constant within a stage),
# and the input resolution feeding the stage. A stage's repeat blocks run at
# ``res_in // stride`` (the entry block's output resolution).
STAGES = [
    {"out_c": 24,  "stride": 2, "se": False, "res_in": 112},
    {"out_c": 40,  "stride": 2, "se": True,  "res_in": 56},
    {"out_c": 80,  "stride": 2, "se": False, "res_in": 28},
    {"out_c": 112, "stride": 1, "se": True,  "res_in": 14},
    {"out_c": 160, "stride": 2, "se": True,  "res_in": 14},
]

_FIRST_STAGE_IN_C = 16  # channels feeding stage 0 = the first block's output


def stage_in_c(stage_idx: int) -> int:
    """Input channels to ``stage_idx`` = previous stage's output (16 for stage 0)."""
    return _FIRST_STAGE_IN_C if stage_idx == 0 else STAGES[stage_idx - 1]["out_c"]


# ---- Resolution scaling (D1 pose pivot) -------------------------------------
# The constants above describe the OFA ImageNet input (224). Pose runs the same
# backbone at 640, where every per-block input resolution re-keys. The spatial
# part of a stage (res_in) is fully *derived* — it is ``stem_res`` walked through
# the per-stage strides — so these helpers reproduce the @224 tables EXACTLY at
# res=224 (one source of truth; the legacy STAGES/FIRST_BLOCK are derived-equal,
# verified in tests/test_resolution.py) and yield the @640 grid for the owed
# deploy-resolution sweep. Channels/strides/SE are resolution-invariant.

BASE_RESOLUTION = 224   # the input the module-level constants describe
STEM_STRIDE = 2         # the 3->16 stem is stride-2 (res -> res // 2)

# Resolution-invariant part of each stage + the first block, derived from the
# legacy constants so out_c/stride/se can never drift from STAGES/FIRST_BLOCK.
_STAGE_SHAPES = [{"out_c": s["out_c"], "stride": s["stride"], "se": s["se"]}
                 for s in STAGES]
_FIRST_BLOCK_SHAPE = {k: v for k, v in FIRST_BLOCK.items() if k != "res"}


def stem_res_for(res: int) -> int:
    """Spatial resolution after the stride-2 stem (``res -> res // STEM_STRIDE``)."""
    return res // STEM_STRIDE


def stages_for_resolution(res: int) -> list[dict]:
    """The five searchable stages with ``res_in`` derived for input ``res``.

    ``out_c``/``stride``/``se`` are resolution-invariant; only ``res_in`` scales.
    Each stage's ``res_in`` is the previous stage's post-stride resolution,
    seeded from the post-stem resolution. ``stages_for_resolution(224)``
    reproduces the module-level :data:`STAGES` table exactly.
    """
    stages: list[dict] = []
    r = stem_res_for(res)
    for shape in _STAGE_SHAPES:
        stages.append({**shape, "res_in": r})
        r = r // shape["stride"]
    return stages


def first_block_for(res: int) -> dict:
    """The fixed first block at input ``res`` (its ``res`` = the post-stem resolution).

    ``first_block_for(224)`` reproduces :data:`FIRST_BLOCK`.
    """
    return {**_FIRST_BLOCK_SHAPE, "res": stem_res_for(res)}


def reachable_mbconv_configs(res: int = BASE_RESOLUTION) -> list[dict]:
    """Every MBConv cfg the OFA-MBv3-w1.0 search space can produce at input ``res``.

    One fixed first block, plus per stage an *entry* block (prev_w -> out_w at
    the stage stride) and a *repeat* block (out_w -> out_w, stride 1), each over
    ``KS x E``. Returns cfg dicts in the catalog's MBConv schema
    ``{in_c, out_c, kernel, stride, expand, se, res}``, de-duplicated and order
    preserved. Size: ``1 + 5 * 2 * |KS| * |E| = 91`` at any single resolution.
    ``res`` defaults to 224 (the ImageNet grid); ``res=640`` is the pose deploy
    grid — disjoint from 224, so unioning it into the catalog is append-only.
    """
    configs: list[dict] = []
    seen: set = set()

    def add(cfg: dict) -> None:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    add(dict(first_block_for(res)))

    for s, stage in enumerate(stages_for_resolution(res)):
        out_c, se = stage["out_c"], stage["se"]
        res_in = stage["res_in"]
        res_out = res_in // stage["stride"]
        for k in KS:
            for e in E:
                # Entry block: in_c -> out_c at the stage stride, at res_in.
                add({"in_c": stage_in_c(s), "out_c": out_c, "kernel": k,
                     "stride": stage["stride"], "expand": e, "se": se,
                     "res": res_in})
                # Repeat block: out_c -> out_c, stride 1, at res_out.
                add({"in_c": out_c, "out_c": out_c, "kernel": k,
                     "stride": 1, "expand": e, "se": se, "res": res_out})
    return configs
