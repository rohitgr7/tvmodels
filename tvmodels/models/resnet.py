from torch import nn
from .layers import Bottleneck, SEBlock, Flatten, conv2d, relu
from ..utils import load_pretrained

__all__ = ['ResNetBasicBlock', 'ResNetBlock', 'ResNet',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
# __all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

META = {
    'resnet18': ['resnet18.pth', 'https://drive.google.com/open?id=1AujRUDxC93cUlTCNjMPJy2C6X5KH7RX7'],
    'resnet34': ['resnet34.pth', 'https://drive.google.com/open?id=1Vkft9-pD5RQfv7hHwIurL333dYzIYbzs'],
    'resnet50': ['resnet50.pth', 'https://drive.google.com/open?id=1-u5nQd7Ev7siOnvhNavHT2sP2f85z6q9'],
    'resnet101': ['resnet101.pth', 'https://drive.google.com/open?id=1hQeOxbVwr1-hkqTWBd9EOUDjSf8W7hnu'],
    'resnet152': ['resnet152.pth', 'https://drive.google.com/open?id=1qReWCS9KsHk5UToAZIC9jrbJm0khwNnO']
}


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
                   self._res_blocks(block, 64, base_blocks[0], **kwargs),
                   self._res_blocks(block, 128, base_blocks[
                                    1], down_ksz, down_pad, 2, **kwargs),
                   self._res_blocks(block, 256, base_blocks[
                                    2], down_ksz, down_pad, 2, **kwargs),
                   self._res_blocks(block, 512, base_blocks[
                                    3], down_ksz, down_pad, 2, **kwargs),
                   nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(ps),
                   Flatten(), nn.Linear(512 * block.expansion, nc)]
        self.net = nn.Sequential(*layers)

    def _res_blocks(self, block, nf, nblocks, down_ksz=1, down_pad=0, stride=1, **kwargs):
        downsample = None
        if (stride != 1) or (self.ni != nf * block.expansion):
            downsample = conv2d(self.ni, nf * block.expansion,
                                down_ksz, stride, down_pad, bn=True)

        layers = [block(self.ni, nf, stride, downsample, se=self.se, **kwargs)]
        self.ni = nf * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.ni, nf, 1, se=self.se, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def resnet18(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBasicBlock, [2, 2, 2, 2], nc=nc)
    return load_pretrained(m, META['resnet18'], dest, pretrained)


def resnet34(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBasicBlock, [3, 4, 6, 3], nc=nc)
    return load_pretrained(m, META['resnet34'], dest, pretrained)


def resnet50(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc)
    return load_pretrained(m, META['resnet50'], dest, pretrained)


def resnet101(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc)
    return load_pretrained(m, META['resnet101'], dest, pretrained)


def resnet152(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 8, 36, 3], nc=nc)
    return load_pretrained(m, META['resnet152'], dest, pretrained)
