import math
import torch.nn as nn

from .se_blocks import SEBlock
from ..layers import convsame_bn_swish, Flatten, DropConnect, Swish

__all__ = ['EfficientNet', 'MBConvBlock']


class EfficientNet(nn.Module):

    def __init__(self, width_c, depth_c, divisor=8, min_depth=None, ps_head=.2, ps_connect=.2, nc=1000):
        super().__init__()
        self.width_c, self.depth_c, self.divisor = width_c, depth_c, divisor
        self.min_depth = min_depth or divisor

        self.net = nn.Sequential(
            # stem
            *convsame_bn_swish(3, self._round_ch(32), 3, 2),

            # MBConvBlocks
            self._mb_layers(self._round_ch(32), self._round_ch(
                16), 1, 3, 1, self._round_rep(1), ps_connect),
            self._mb_layers(self._round_ch(16), self._round_ch(
                24), 6, 3, 2, self._round_rep(2), ps_connect),
            self._mb_layers(self._round_ch(24), self._round_ch(
                40), 6, 5, 2, self._round_rep(2), ps_connect),
            self._mb_layers(self._round_ch(40), self._round_ch(
                80), 6, 3, 2, self._round_rep(3), ps_connect),
            self._mb_layers(self._round_ch(80), self._round_ch(
                112), 6, 5, 1, self._round_rep(3), ps_connect),
            self._mb_layers(self._round_ch(112), self._round_ch(
                192), 6, 5, 2, self._round_rep(4), ps_connect),
            self._mb_layers(self._round_ch(192), self._round_ch(
                320), 6, 3, 1, self._round_rep(1), ps_connect),

            # Head
            *convsame_bn_swish(self._round_ch(320), self._round_ch(1280), ksz=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(ps_head, True),
            Flatten(),
            nn.Linear(self._round_ch(1280), nc)
        )

    def _round_ch(self, ch):
        if not self.width_c:
            return ch
        ch *= self.width_c
        new_ch = max(self.min_depth, int(ch + self.divisor / 2) //
                     self.divisor * self.divisor)
        if new_ch < 0.9 * ch:
            new_ch += self.divisor
        return int(new_ch)

    def _round_rep(self, repeats):
        if not self.depth_c:
            return repeats
        return int(math.ceil(self.depth_c * repeats))

    def _mb_layers(self, ni, nf, expand, ksz, stride, nblocks, ps):
        layers = [MBConvBlock(ni, nf, expand, ksz, stride, ps)]
        layers += [MBConvBlock(nf, nf, expand, ksz, 1, ps)
                   for _ in range(1, nblocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MBConvBlock(nn.Module):

    def __init__(self, ni, nf, expand, ksz, stride, ps=0.2, skip=True, se_ratio=0.25):
        super().__init__()
        mid = ni * expand
        layers = []

        if expand > 1:
            layers.extend(convsame_bn_swish(ni, mid, 1))

        layers.extend(convsame_bn_swish(mid, mid, ksz, stride, groups=mid))

        if se_ratio > 0:
            layers.append(SEBlock(mid, int(ni * se_ratio), act=Swish))
        layers.extend(convsame_bn_swish(mid, nf, 1, 1, swish=False))
        self.net = nn.Sequential(*layers)

        self.dropconnect = DropConnect(ps) if ps > 0 else None
        self.skip = skip and (stride == 1) and (ni == nf)

    def forward(self, x):
        out = self.net(x)
        if self.skip:
            if self.dropconnect is not None:
                out = self.dropconnect(out)
            out += x
        return out
