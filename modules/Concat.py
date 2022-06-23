import torch
from torch import nn

from modules.Attention import ChannelAttention
from modules.autopad import autopad


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class SEConcat(nn.Module):
    # v1 positive
    def __init__(self, channels, dimension=1):
        super().__init__()
        self.d = dimension
        self.attes = nn.ModuleList([ChannelAttention(c) for c in channels])

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i] * self.attes[i](x[i])
        x = torch.cat(x, self.d)
        return x

class Smoother(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        self.smoother = nn.Conv3d(1, 1, (3, 1, 1), 1, (1, 0, 0))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.smoother(x)
        return x.squeeze(1)

class SmothingConcat(nn.Module):
    def __init__(self, channels, dimension=1):
        super().__init__()
        self.d = dimension
        self.smoother = Smoother(sum(channels))

    def forward(self, x):
        x = torch.cat(x, self.d)
        return self.smoother(x)

class SSEConcat(nn.Module):
    # v2 positive
    def __init__(self, channels, dimension=1):
        super().__init__()
        self.d = dimension
        self.attes = nn.ModuleList([ChannelAttention(c) for c in channels])
        self.smoother = Smoother(sum(channels))

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i] * self.attes[i](x[i])
        x = torch.cat(x, self.d)
        return self.smoother(x)


class FastNormFusion(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.sigma = torch.tensor(0.0001)
        self.dim = dim

    def forward(self, x: torch.tensor):
        all = torch.sum(x, dim=self.dim)
        x = x / (self.sigma + all)
        return x


class MultiScaleBalance(nn.Module):
    def __init__(self):
        super(MultiScaleBalance, self).__init__()
        # suppose max scale 160 -> 40
        self.extra_layer = nn.MaxPool2d(3, 2, autopad(3, None))
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.extra_layer(x)
        x = self.pooling(x)
        x = torch.max(x)
        return x


class MultiScaleBalanceV2(nn.Module):
    def __init__(self, channels):
        super(MultiScaleBalanceV2, self).__init__()
        # suppose max scale 160 -> 40
        self.extra_layer = nn.Conv2d(channels, 1, 3, 2, autopad(3, None))
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.extra_layer(x)
        x = self.pooling(x)
        x = torch.max(x)
        return x

class SSEBlender(nn.Module):
    # v3 completely
    def __init__(self, channels, dimension=1):
        super().__init__()
        self.d = dimension
        self.attes = nn.ModuleList([ChannelAttention(c) for c in channels])
        self._local_fusion = Smoother()
        self.msb = MultiScaleBalanceV2()
        self.FNF = FastNormFusion()

    def forward(self, x):
        gw = []
        for e in x:
            gw.append(self.FNF(self.msb(e)))
        for i in range(len(x)):
            x[i] = x[i] * self.attes[i](x[i]) * gw[i]
        x = torch.cat(x, self.d)
        x = self._local_fusion(x)
        return x
