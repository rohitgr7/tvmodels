from .blocks import ResNetBlock, ResNet, SEResNetBlock, SENetBlock
from ..utils import load_pretrained

__all__ = ['se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4', 'se_resnext101_32x4', 'senet154']

META = {
    'se_resnet50': ['se_resnet50.pth', 'https://drive.google.com/open?id=1b5E74XhwfJch640k_kLuARXhW_j67XlR'],
    'se_resnet101': ['se_resnet101.pth', 'https://drive.google.com/open?id=1ILaXyDCMPQy0-GcGq1tpJlAAeoAx1Oax'],
    'se_resnet152': ['se_resnet152.pth', 'https://drive.google.com/open?id=1_uSyr9tSyUOQQ7MhSxj9I67_W6eeKaQu'],
    'se_resnext50_32x4': ['se_resnext50_32x4.pth', 'https://drive.google.com/open?id=1Yp39MYzO4Q8aENNoC69aDqI9nGbRYgD2'],
    'se_resnext101_32x4': ['se_resnext101_32x4.pth', 'https://drive.google.com/open?id=18La-mP-XIudMgMS0HW-Tyu6-81LT1h5F'],
    'senet154': ['senet154.pth', 'https://drive.google.com/open?id=1Pyk8iyBz17ub7JMDmXbGR5FRcCD4rR0p']
}


def se_resnet50(pretrained=False, nc=1000, dest=None):
    m = ResNet(SEResNetBlock, [3, 4, 6, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet50'], dest, pretrained)


def se_resnet101(pretrained=False, nc=1000, dest=None):
    m = ResNet(SEResNetBlock, [3, 4, 23, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet101'], dest, pretrained)


def se_resnet152(pretrained=False, nc=1000, dest=None):
    m = ResNet(SEResNetBlock, [3, 8, 36, 3], nc=nc, se=True)
    return load_pretrained(m, META['se_resnet152'], dest, pretrained)


def se_resnext50_32x4(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc,
               se=True, groups=32, base_width=4)
    return load_pretrained(m, META['se_resnext50_32x4'], dest, pretrained)


def se_resnext101_32x4(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc,
               se=True, groups=32, base_width=4)
    return load_pretrained(m, META['se_resnext101_32x4'], dest, pretrained)


def senet154(pretrained=False, nc=1000, dest=None):
    m = ResNet(SENetBlock, [3, 8, 36, 3], ni=128, ps=0.2,
               init_ksz3=True, down_ksz=3, down_pad=1, nc=nc, se=True)
    return load_pretrained(m, META['senet154'], dest, pretrained)
