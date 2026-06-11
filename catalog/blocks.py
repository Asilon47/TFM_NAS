"""Block registry.

Each entry maps a block name to:
  - `builder`: callable(cfg) -> nn.Module
  - `input_shape_fn`: callable(cfg) -> tuple  (input tensor shape for ONNX export)
  - `grid`: list of cfg dicts to sweep

The sweep module does the cartesian product; keep grids narrow — every cfg
becomes a row in the LUT.
"""
from itertools import product
import torch.nn as nn

from .mbconv import Conv3x3, Conv1x1, DWConv, MBConv, Skip, SEBlock
from .seg_det import (Upsample, Deconv, PixelShuffleBlock, FPNLateral,
                      FPNTopDown, DilatedConv3x3, ASPP, SegHead, DetHead)
from .shapes import BATCH
from .ofa_mbv3 import reachable_mbconv_configs


def _grid(**axes):
    keys = list(axes.keys())
    return [dict(zip(keys, vals)) for vals in product(*axes.values())]


# ---- builders ----------------------------------------------------------------

def _b_conv3x3(cfg):   return Conv3x3(cfg["in_c"], cfg["out_c"], cfg["stride"])
def _b_conv1x1(cfg):   return Conv1x1(cfg["in_c"], cfg["out_c"])
def _b_dwconv(cfg):    return DWConv(cfg["in_c"], cfg["kernel"], cfg["stride"])
def _b_se(cfg):        return SEBlock(cfg["in_c"])
def _b_skip(cfg):      return Skip()
def _b_mbconv(cfg):    return MBConv(cfg["in_c"], cfg["out_c"], cfg["kernel"],
                                     cfg["stride"], cfg["expand"], cfg["se"])

def _b_upsample(cfg):  return Upsample(cfg["scale"], cfg["mode"])
def _b_deconv(cfg):    return Deconv(cfg["in_c"], cfg["out_c"])
def _b_pshuffle(cfg):  return PixelShuffleBlock(cfg["in_c"], cfg["out_c"], cfg["upscale"])
def _b_fpn_lat(cfg):   return FPNLateral(cfg["in_c"], cfg["out_c"])
def _b_fpn_td(cfg):    return FPNTopDown(cfg["c"])
def _b_dilconv(cfg):   return DilatedConv3x3(cfg["c"], cfg["dilation"])
def _b_aspp(cfg):      return ASPP(cfg["in_c"], cfg["out_c"])
def _b_seghead(cfg):   return SegHead(cfg["in_c"], cfg["num_classes"])
def _b_dethead(cfg):   return DetHead(cfg["in_c"], cfg["num_classes"], cfg["num_anchors"])


# ---- input shape functions ---------------------------------------------------

def _in(cfg): return (BATCH, cfg["in_c"], cfg["res"], cfg["res"])
def _in_single(cfg): return (BATCH, cfg["c"], cfg["res"], cfg["res"])


# ---- grids -------------------------------------------------------------------
# Keep each grid small. Start narrow; widen later by editing this file — the LUT
# schema is append-only so existing rows remain valid when the grid grows.

_MBCONV_GRID = _grid(
    in_c=[16, 32, 64, 96],
    out_c=[16, 32, 64, 96, 160],
    kernel=[3, 5, 7],
    stride=[1, 2],
    expand=[3, 4, 6],
    se=[False, True],
    res=[56, 28, 14, 7],
)
# Prune: only keep cfgs where in_c<=out_c makes sense for the NAS search
_MBCONV_GRID = [c for c in _MBCONV_GRID if c["out_c"] >= c["in_c"]]

# Augment with the exact MBConv configs the OFA-MBv3-w1.0 search space can reach
# (CP 2.1). The generic grid above is kept; OFA's real widths (24/40/80/112),
# the 112 resolution, and the expand=1 first block live only here, so
# search.arch_to_blocks emits only LUT-covered rows. Unioned in, de-duplicated.
_MBCONV_GRID = _MBCONV_GRID + [c for c in reachable_mbconv_configs()
                               if c not in _MBCONV_GRID]

_CONV3X3_GRID = _grid(in_c=[16, 32, 64, 96, 160], out_c=[16, 32, 64, 96, 160],
                      stride=[1, 2], res=[112, 56, 28, 14])
_CONV1X1_GRID = _grid(in_c=[16, 32, 64, 96, 160, 320], out_c=[16, 32, 64, 96, 160, 320],
                      res=[56, 28, 14, 7])
_DWCONV_GRID  = _grid(in_c=[16, 32, 64, 96, 160], kernel=[3, 5, 7],
                      stride=[1, 2], res=[56, 28, 14])
_SE_GRID      = _grid(in_c=[32, 64, 96, 160], res=[28, 14, 7])
_SKIP_GRID    = _grid(in_c=[16, 32, 64, 96, 160], res=[28, 14, 7])

_UPSAMPLE_GRID = _grid(in_c=[32, 64, 96, 160], scale=[2, 4],
                       mode=["nearest", "bilinear"], res=[14, 28])
_DECONV_GRID   = _grid(in_c=[32, 64, 96], out_c=[16, 32, 64], res=[14, 28])
_PSHUFFLE_GRID = _grid(in_c=[64, 96, 160], out_c=[32, 64], upscale=[2],
                       res=[14, 28])
_FPN_LAT_GRID  = _grid(in_c=[64, 128, 256, 512], out_c=[128, 256], res=[56, 28, 14, 7])
_FPN_TD_GRID   = _grid(c=[128, 256], res=[28, 14, 7])
_DILCONV_GRID  = _grid(c=[64, 128, 256], dilation=[2, 4, 6, 12],
                       res=[28, 14])
_ASPP_GRID     = _grid(in_c=[256, 512], out_c=[256], res=[28, 14])
_SEGHEAD_GRID  = _grid(in_c=[128, 256], num_classes=[19, 21], res=[56, 28])
_DETHEAD_GRID  = _grid(in_c=[128, 256], num_classes=[80], num_anchors=[3],
                       res=[40, 20, 10])


BLOCK_REGISTRY = {
    "conv3x3":     {"builder": _b_conv3x3, "input_shape": _in,        "grid": _CONV3X3_GRID},
    "conv1x1":     {"builder": _b_conv1x1, "input_shape": _in,        "grid": _CONV1X1_GRID},
    "dwconv":      {"builder": _b_dwconv,  "input_shape": _in,        "grid": _DWCONV_GRID},
    "se":          {"builder": _b_se,      "input_shape": _in,        "grid": _SE_GRID},
    "skip":        {"builder": _b_skip,    "input_shape": _in,        "grid": _SKIP_GRID},
    "mbconv":      {"builder": _b_mbconv,  "input_shape": _in,        "grid": _MBCONV_GRID},
    "upsample":    {"builder": _b_upsample, "input_shape": _in,       "grid": _UPSAMPLE_GRID},
    "deconv":      {"builder": _b_deconv,  "input_shape": _in,        "grid": _DECONV_GRID},
    "pixelshuffle":{"builder": _b_pshuffle, "input_shape": _in,       "grid": _PSHUFFLE_GRID},
    "fpn_lateral": {"builder": _b_fpn_lat, "input_shape": _in,        "grid": _FPN_LAT_GRID},
    "fpn_topdown": {"builder": _b_fpn_td,  "input_shape": _in_single, "grid": _FPN_TD_GRID},
    "dilconv":     {"builder": _b_dilconv, "input_shape": _in_single, "grid": _DILCONV_GRID},
    "aspp":        {"builder": _b_aspp,    "input_shape": _in,        "grid": _ASPP_GRID},
    "seghead":     {"builder": _b_seghead, "input_shape": _in,        "grid": _SEGHEAD_GRID},
    "dethead":     {"builder": _b_dethead, "input_shape": _in,        "grid": _DETHEAD_GRID},
}


def build_block(block_name: str, cfg: dict) -> nn.Module:
    spec = BLOCK_REGISTRY[block_name]
    return spec["builder"](cfg)


def input_shape_for(block_name: str, cfg: dict):
    return BLOCK_REGISTRY[block_name]["input_shape"](cfg)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
