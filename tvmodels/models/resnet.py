from .blocks import ResNetBasicBlock, ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

META = {
    'resnet18': ['resnet18.pth', 'https://drive.google.com/open?id=1d-AgSMO7HcKihDEXHqzDX2dAxYtH7N2m'],
    'resnet34': ['resnet34.pth', 'https://drive.google.com/open?id=1HiUX27F3Luu27K3rfLc9dUXzwmwcBVpZ'],
    'resnet50': ['resnet50.pth', 'https://drive.google.com/open?id=10aA1Qex-5CvzZMCmHaWMMfeFToN3DHkr'],
    'resnet101': ['resnet101.pth', 'https://drive.google.com/open?id=1rbgoMnCHNGHHXbJhYiOsgAMGYkeQ7tvP'],
    'resnet152': ['resnet152.pth', 'https://drive.google.com/open?id=15fAZ1PJ6ESiIkRG55TwKr_vMZ5xhHtGu']
}


def resnet18(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBasicBlock, [2, 2, 2, 2], nc=nc)
    return load_pretrained(m, META['resnet18'], dest, pretrained)


def resnet34(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBasicBlock, [3, 4, 6, 3], nc=nc)
    return load_pretrained(m, META['resnet34'], dest, pretrained)


def resnet50(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 6, 3], nc=nc)
    return load_pretrained(m, META['resnet50'], dest, pretrained)


def resnet101(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 4, 23, 3], nc=nc)
    return load_pretrained(m, META['resnet101'], dest, pretrained)


def resnet152(pretrained=False, nc=1000, dest=None):
    m = ResNet(ResNetBlock, [3, 8, 36, 3], nc=nc)
    return load_pretrained(m, META['resnet152'], dest, pretrained)
