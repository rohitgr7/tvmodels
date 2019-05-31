from .blocks import ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['resnext50_32x4', 'resnext101_32x4',
           'resnext101_32x8', 'resnext101_64x4', 'resnext152']

META = {
    'resnext18': [],
    'resnext34': [],
    'resnext50_32x4': ['resnext50_32x4.pth', 'https://drive.google.com/open?id=1CPd2ZoGEDmFZFXnJaeCce1im8sVYAXSL'],
    'resnext101_32x4': ['resnext101_32x4.pth', 'https://drive.google.com/open?id=1iGySCd5mUt-ia3zUORyeGcCysfBB66wF'],
    'resnext101_32x8': ['resnext101_32x8.pth', 'https://drive.google.com/open?id=1Nd1dJhciTHRMVCrlKTcrzkG6oea5_qF3'],
    'resnext101_64x4': ['resnext101_64x4.pth', 'https://drive.google.com/open?id=1TjPAog6GGWZseAeHa4RrUYVJbh4LEEru'],
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
