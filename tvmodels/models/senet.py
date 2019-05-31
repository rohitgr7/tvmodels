from .blocks import ResNetBlock, ResNet, SEResNetBlock, SENetBlock
from ..utils import load_pretrained

__all__ = ['se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4', 'se_resnext101_32x4', 'senet154']

META = {
    'se_resnet50': ['se_resnet50.pth', 'https://drive.google.com/open?id=1juma-phILOUZfjf8jGwqIKAjuEdev-kQ'],
    'se_resnet101': ['se_resnet101.pth', 'https://drive.google.com/open?id=1vFWGq0x2XUxYzKNE__ylQq_3Jeat2I3T'],
    'se_resnet152': ['se_resnet152.pth', 'https://drive.google.com/open?id=1qVx8AhnJK_5FIROBz2QsD2GjPA-HLL81'],
    'se_resnext50_32x4': ['se_resnext50_32x4.pth', 'https://drive.google.com/open?id=1El0GAdnTYYMPBLGWnyk8MJnRJkGvQx7p'],
    'se_resnext101_32x4': ['se_resnext101_32x4.pth', 'https://drive.google.com/open?id=1lcUJD3VqSFgVJCPFUnQOl9MBYWuABWkR'],
    'senet154': ['senet154.pth', 'https://drive.google.com/open?id=1c3kD_plELHJAvBXO5yO8ZJin-s7dYIoD']
}


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
