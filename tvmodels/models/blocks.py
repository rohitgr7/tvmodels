from torch import nn
from .layers import Flatten, conv2d, relu

__all__ = ['Bottleneck', 'ResNetBasicBlock', 'ResNetBlock',
           'ResNet', 'SEBlock', 'SENetBlock', 'SEResNetBlock']


class Bottleneck(nn.Module):

    def forward(self, x):
        out = self.net(x)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return self.relu(out)


class ResNetBasicBlock(Bottleneck):
    expansion = 1

    def __init__(self, ni, nf, stride, downsample=None, groups=1, base_width=64, **kwargs):
        super().__init__()
        W = int(nf * (base_width / 64.)) * groups
        layers = [conv2d(ni, W, 3, stride, 1, bn=True), relu(),
                  conv2d(W, W, 3, 1, 1, bn=True, groups=groups)]
        self.net = nn.Sequential(*layers)
        self.relu = relu()
        self.downsample = downsample


class ResNetBlock(Bottleneck):
    expansion = 4

    def __init__(self, ni, nf, stride, downsample=None, se=False, ratio=16, groups=1, base_width=64):
        super().__init__()
        W = int(nf * (base_width / 64.)) * groups
        layers = [conv2d(ni, W, 1, bn=True), relu(),
                  conv2d(W, W, 3, stride, 1, groups, bn=True), relu(),
                  conv2d(W, nf * self.expansion, 1, bn=True)]
        if se:
            layers.append(SEBlock(nf * 4, ratio))
        self.net = nn.Sequential(*layers)
        self.relu = relu()
        self.downsample = downsample


class ResNet(nn.Module):

    def __init__(self, block, base_blocks, ni=64, ps=0., init_ksz3=False, down_ksz=1, down_pad=0, nc=1000, se=False, **kwargs):
        super().__init__()
        self.ni, self.se = ni, se
        layers = []

        if init_ksz3:
            layers += [conv2d(3, 64, 3, 2, 1, bn=True), relu(),
                       conv2d(64, 64, 3, 1, 1, bn=True), relu(),
                       conv2d(64, ni, 3, 1, 1, bn=True), relu()]
        else:
            layers += [conv2d(3, ni, 7, 2, 3, bn=True), relu()]

        layers += [nn.MaxPool2d(3, 2, padding=int(not se), ceil_mode=se),
                   self._res_blocks(block, 64, base_blocks[0], **kwargs)]

        layers += [self._res_blocks(block, 128 * (2**i), base_blocks[i + 1], down_ksz, down_pad, 2, **kwargs)
                   for i in range(3)]

        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(ps),
                   Flatten(), nn.Linear(512 * block.expansion, nc)]
        self.net = nn.Sequential(*layers)

    def _res_blocks(self, block, nf, nblocks, down_ksz=1, down_pad=0, stride=1, **kwargs):
        downsample = None
        if (stride != 1) or (self.ni != nf * block.expansion):
            downsample = conv2d(self.ni, nf * block.expansion,
                                down_ksz, stride, down_pad, bn=True)

        layers = [block(self.ni, nf, stride, downsample, se=self.se, **kwargs)]
        self.ni = nf * block.expansion
        layers += [block(self.ni, nf, 1, se=self.se, **kwargs)
                   for _ in range(1, nblocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):

    def __init__(self, nc, ratio=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d(nc, nc // ratio, 1), relu(),
            conv2d(nc // ratio, nc, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)


class SENetBlock(Bottleneck):
    expansion = 4

    def __init__(self, ni, nf, stride, downsample=None, groups=64, ratio=16, **kwargs):
        super().__init__()
        layers = [conv2d(ni, nf * 2, 1, bn=True), relu(),
                  conv2d(nf * 2, nf * 4, 3, stride,
                         1, groups, bn=True), relu(),
                  conv2d(nf * 4, nf * 4, 1, bn=True),
                  SEBlock(nf * 4, ratio)]
        self.net = nn.Sequential(*layers)
        self.relu = relu()
        self.downsample = downsample


class SEResNetBlock(Bottleneck):
    expansion = 4

    def __init__(self, ni, nf, stride, downsample=None, groups=1, ratio=16, **kwargs):
        super().__init__()
        layers = [conv2d(ni, nf, 1, stride, bn=True), relu(),
                  conv2d(nf, nf, 3, 1, 1, groups, bn=True), relu(),
                  conv2d(nf, nf * self.expansion, 1, bn=True),
                  SEBlock(nf * 4, ratio)]
        self.net = nn.Sequential(*layers)
        self.relu = relu()
        self.downsample = downsample
