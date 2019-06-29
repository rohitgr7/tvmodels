import torch.nn as nn

from ..layers import relu, conv_bn

__all__ = ['SEBlock']


class SEBlock(nn.Module):

    def __init__(self, ni, nf, act=relu):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            *conv_bn(ni, nf, 1), act(),
            *conv_bn(nf, ni, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)
