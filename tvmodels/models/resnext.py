from .blocks import ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['resnext50_32x4', 'resnext101_32x4',
           'resnext101_32x8', 'resnext101_64x4', 'resnext152']

META = {
    'resnext50_32x4': ['resnext50_32x4.pth', 'https://drive.google.com/open?id=1NOni3ClRTaO2cUB-wRSm69qZye66CvPd'],
    'resnext101_32x4': ['resnext101_32x4.pth', 'https://drive.google.com/open?id=1flTAweS7XpnmduRe714266PPP_W-WyUM'],
    'resnext101_32x8': ['resnext101_32x8.pth', 'https://drive.google.com/open?id=1RhPki5kNIGUsBL8YY7uBo5KGCQwgkD02'],
    'resnext101_64x4': ['resnext101_64x4.pth', 'https://drive.google.com/open?id=1B3OsbSaQlD5I3IhexHNbTfZ8n6EibSkZ'],
    'resnext152': []
}


def resnext50_32x4(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext50_32x4'], dest, pretrained)


def resnext101_32x4(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext101_32x4'], dest, pretrained)


def resnext101_32x8(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=8)
    return load_pretrained(m, META['resnext101_32x8'], dest, pretrained)


def resnext101_64x4(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=64, base_width=4)
    return load_pretrained(m, META['resnext101_64x4'], dest, pretrained)


def resnext152(nc=1000, pretrained=False, dest=None):
    m = ResNet(ResNetBlock, [3, 8, 36, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext152'], dest, pretrained)
