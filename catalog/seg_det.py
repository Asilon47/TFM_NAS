"""Blocks specific to image-processing models (detection / segmentation)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, scale: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode,
                             align_corners=False if self.mode == "bilinear" else None)


class Deconv(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 4, stride: int = 2):
        super().__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding=(kernel - stride) // 2, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, upscale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * upscale * upscale, 3, 1, 1, bias=False)
        self.ps = nn.PixelShuffle(upscale)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.ps(self.conv(x))))


class FPNLateral(nn.Module):
    """1x1 conv that brings a backbone tap into the FPN channel count."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.op = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.op(x)


class FPNTopDown(nn.Module):
    """Upsample(2x) + add-ready 3x3 conv. Models just the conv+upsample cost;
    the skip-add is free so it's omitted from the measured op."""
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DilatedConv3x3(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(c), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — DeepLab-style, 4 branches + image pool."""
    def __init__(self, in_c: int, out_c: int, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False),
                                nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, r, dilation=r, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True),
            ) for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c * (2 + len(rates)), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [self.b0(x)] + [b(x) for b in self.branches]
        outs.append(F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))


class SegHead(nn.Module):
    """Simple seg head: 3x3 conv → dropout → 1x1 classifier."""
    def __init__(self, in_c: int, num_classes: int = 19):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c), nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_c, num_classes, 1),
        )

    def forward(self, x):
        return self.op(x)


class DetHead(nn.Module):
    """YOLO-style coupled detection head: two 3x3 convs → 1x1 to (5+num_classes)*num_anchors."""
    def __init__(self, in_c: int, num_classes: int = 80, num_anchors: int = 3):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c), nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c), nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, (5 + num_classes) * num_anchors, 1),
        )

    def forward(self, x):
        return self.op(x)
