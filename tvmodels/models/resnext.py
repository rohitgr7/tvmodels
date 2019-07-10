from .blocks import ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['resnext50_32x4', 'resnext101_32x4', 'resnext101_32x8', 'resnext101_32x16',
           'resnext101_32x32', 'resnext101_32x48', 'resnext101_64x4', 'resnext152']

META = {
    'resnext50_32x4': ['resnext50_32x4.pth', 'https://drive.google.com/open?id=1NOni3ClRTaO2cUB-wRSm69qZye66CvPd'],
    'resnext101_32x4': ['resnext101_32x4.pth', 'https://drive.google.com/open?id=1flTAweS7XpnmduRe714266PPP_W-WyUM'],
    'resnext101_32x8': ['resnext101_32x8.pth', 'https://drive.google.com/open?id=1-G1X-tQvjil7UJhvqoQhhQgyOfutaGkg'],
    'resnext101_32x16': ['resnext101_32x16.pth', 'https://drive.google.com/open?id=1-B29ewr2bJRn7XMfsH2SuA96xgviLMTE'],
    'resnext101_32x32': ['resnext101_32x32.pth', 'https://drive.google.com/open?id=1-1YB417ZIwbifHUTB4mIn9Re4ofoVOmq'],
    'resnext101_32x48': ['resnext101_32x48.pth', 'https://drive.google.com/open?id=1-18aLEcE0WigP449LiN4xWl9EWuWW8uj'],
    'resnext101_64x4': ['resnext101_64x4.pth', 'https://drive.google.com/open?id=1B3OsbSaQlD5I3IhexHNbTfZ8n6EibSkZ'],
    'resnext152': []
}


def resnext50_32x4(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext50_32x4'], dest, pretrained)


def resnext101_32x4(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext101_32x4'], dest, pretrained)


def resnext101_32x8(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=8)
    return load_pretrained(m, META['resnext101_32x8'], dest, pretrained)


def resnext101_32x16(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=16)
    return load_pretrained(m, META['resnext101_32x16'], dest, pretrained)


def resnext101_32x32(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=32)
    return load_pretrained(m, META['resnext101_32x32'], dest, pretrained)


def resnext101_32x48(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=32, base_width=48)
    return load_pretrained(m, META['resnext101_32x48'], dest, pretrained)


def resnext101_64x4(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc, groups=64, base_width=4)
    return load_pretrained(m, META['resnext101_64x4'], dest, pretrained)


def resnext152(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 8, 36, 3], nc=nc, groups=32, base_width=4)
    return load_pretrained(m, META['resnext152'], dest, pretrained)
