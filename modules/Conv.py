import math

import torch
from torch import nn
import torch.nn.functional as F

from modules.autopad import autopad


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, (k, 1), s, (autopad(k, p), autopad(1, p)), groups=g, bias=False)
        self.conv2 = nn.Conv2d(c2, c2, (1, k), s, (autopad(1, p), autopad(k, p)), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(self.bn(x))

    def forward_fuse(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x)

class autoConv2d(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,
                 bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if isinstance(k, int):
            k = (k, k)
        self.kernels = nn.Parameter(torch.rand(c2, c1, *k))
        self.sig = nn.Hardswish()

        if bias:
            self.bias = nn.Parameter(torch.rand(c2))
        else:
            self.bias = None
        self.s = s
        self.ps = (autopad(k[0], p), autopad(k[1], p))
        self.c2 = c2
        self.g = g

    def forward(self, x):
        kernal = self.sig(self.kernels) * self.kernels
        print(kernal)
        return F.conv2d(x, kernal, self.bias, self.s, self.ps, groups=self.g)

class AutoConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = autoConv2d(c1, c2, k, s, autopad(k, p), g=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class TransConv(nn.Module):
    def __init__(self, channelIn, channelOut=None, c_ratio=16):
        super(TransConv, self).__init__()
        self.confuse = nn.Conv2d(channelIn, channelIn, 1, bias=False)
        self.tcov = nn.ConvTranspose2d(channelIn, channelIn, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(channelIn)

    def forward(self, x):
        x = self.confuse(x)
        x = self.tcov(x)
        x = self.bn(x)
        return x