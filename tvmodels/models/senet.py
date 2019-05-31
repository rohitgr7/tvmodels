from torch import nn
from .layers import Bottleneck, SEBlock, conv2d, relu
from .resnet import ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['SENetBlock', 'SEResNetBlock', 'se_resnet50', 'se_resnet101',
           'se_resnet152', 'se_resnext50_32x4', 'se_resnext101_32x4', 'senet154']

META = {
    'se_resnet50': ['se_resnet50.pth', 'https://drive.google.com/open?id=1juma-phILOUZfjf8jGwqIKAjuEdev-kQ'],
    'se_resnet101': ['se_resnet101.pth', 'https://drive.google.com/open?id=1vFWGq0x2XUxYzKNE__ylQq_3Jeat2I3T'],
    'se_resnet152': ['se_resnet152.pth', 'https://drive.google.com/open?id=1qVx8AhnJK_5FIROBz2QsD2GjPA-HLL81'],
    'se_resnext50_32x4': ['se_resnext50_32x4.pth', 'https://drive.google.com/open?id=1El0GAdnTYYMPBLGWnyk8MJnRJkGvQx7p'],
    'se_resnext101_32x4': ['se_resnext101_32x4.pth', 'https://drive.google.com/open?id=1lcUJD3VqSFgVJCPFUnQOl9MBYWuABWkR'],
    'senet154': ['senet154.pth', 'https://drive.google.com/open?id=1c3kD_plELHJAvBXO5yO8ZJin-s7dYIoD']
}


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


def se_resnet50(nc=1000, pretrained=False, dest=None):
    m = ResNet(SEResNetBlock, [3, 4, 6, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet50'], dest, pretrained)


def se_resnet101(nc=1000, pretrained=False, dest=None):
    m = ResNet(SEResNetBlock, [3, 4, 23, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet101'], dest, pretrained)


def se_resnet152(nc=1000, pretrained=False, dest=None):
    m = ResNet(SEResNetBlock, [3, 8, 36, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet152'], dest, pretrained)


def se_resnext50_32x4(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc,
               se=True, groups=32, base_width=4)
    return load_pretrained(m, META['se_resnext50_32x4'], dest, pretrained)


def se_resnext101_32x4(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc,
               se=True, groups=32, base_width=4)
    return load_pretrained(m, META['se_resnext101_32x4'], dest, pretrained)


def senet154(nc=1000, pretrained=False, dest=None):
    m = ResNet(SENetBlock, [3, 8, 36, 3], ni=128, ps=0.2,
               init_ksz3=True, down_ksz=3, down_pad=1, nc=nc, se=True)
    return load_pretrained(m, META['senet154'], dest, pretrained)
