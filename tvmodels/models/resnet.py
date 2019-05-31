from .blocks import ResNetBasicBlock, ResNetBlock, ResNet
from ..utils import load_pretrained

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

META = {
    'resnet18': ['resnet18.pth', 'https://drive.google.com/open?id=1AujRUDxC93cUlTCNjMPJy2C6X5KH7RX7'],
    'resnet34': ['resnet34.pth', 'https://drive.google.com/open?id=1Vkft9-pD5RQfv7hHwIurL333dYzIYbzs'],
    'resnet50': ['resnet50.pth', 'https://drive.google.com/open?id=1-u5nQd7Ev7siOnvhNavHT2sP2f85z6q9'],
    'resnet101': ['resnet101.pth', 'https://drive.google.com/open?id=1hQeOxbVwr1-hkqTWBd9EOUDjSf8W7hnu'],
    'resnet152': ['resnet152.pth', 'https://drive.google.com/open?id=1qReWCS9KsHk5UToAZIC9jrbJm0khwNnO']
}


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
