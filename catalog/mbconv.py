"""MBConv-family building blocks used by MobileNet/EfficientNet NAS search spaces."""
import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class Conv1x1(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class DWConv(nn.Module):
    def __init__(self, in_c: int, kernel: int, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.op = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel, stride, pad, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class SEBlock(nn.Module):
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        mid = max(1, c // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1), nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MBConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int,
                 expand: int, se: bool):
        super().__init__()
        mid = in_c * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_c, mid, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        pad = kernel // 2
        layers += [nn.Conv2d(mid, mid, kernel, stride, pad, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        if se:
            layers.append(SEBlock(mid))
        layers += [nn.Conv2d(mid, out_c, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(out_c)]
        self.op = nn.Sequential(*layers)
        self.residual = (stride == 1 and in_c == out_c)

    def forward(self, x):
        y = self.op(x)
        return x + y if self.residual else y


class Skip(nn.Module):
    """Identity, modeled as an op so the LUT has a cost for 'choose skip'."""
    def forward(self, x):
        return x
