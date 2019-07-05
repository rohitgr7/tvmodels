from .blocks import EfficientNet
from ..utils import load_pretrained

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', ]

META = {
    'efficientnet_b0': ['efficientnet_b0.pth', 'https://drive.google.com/open?id=1XpmtF_Gi7bF0VcBd_tgVAHn_fnEs0lUN'],
    'efficientnet_b1': ['efficientnet_b1.pth', 'https://drive.google.com/open?id=1RtZH6CTnvayoobT-juafLQfXLToZWILB'],
    'efficientnet_b2': ['efficientnet_b2.pth', 'https://drive.google.com/open?id=1Dvt25WgXWZFuWjm3FusNRKbLHMV3k7dm'],
    'efficientnet_b3': ['efficientnet_b3.pth', 'https://drive.google.com/open?id=1GpI0epWykKf6HKO-iLH8pJCGio15JC19'],
    'efficientnet_b4': ['efficientnet_b4.pth', 'https://drive.google.com/open?id=1PQbXesyou-SVEUK2PHVBSepY8oBsmw4R'],
    'efficientnet_b5': ['efficientnet_b5.pth', 'https://drive.google.com/open?id=1Bxh1s01jglut0S8citmPIbpJpwYDqXho'],
    'efficientnet_b6': [],
    'efficientnet_b7': []
}


def efficientnet_b0(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1., depth_c=1., ps_head=0.2, nc=nc)
    return load_pretrained(m, META['efficientnet_b0'], dest, pretrained)


def efficientnet_b1(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1., depth_c=1.1, ps_head=0.2, nc=nc)
    return load_pretrained(m, META['efficientnet_b1'], dest, pretrained)


def efficientnet_b2(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1.1, depth_c=1.2, ps_head=0.3, nc=nc)
    return load_pretrained(m, META['efficientnet_b2'], dest, pretrained)


def efficientnet_b3(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1.2, depth_c=1.4, ps_head=0.3, nc=nc)
    return load_pretrained(m, META['efficientnet_b3'], dest, pretrained)


def efficientnet_b4(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1.4, depth_c=1.8, ps_head=0.4, nc=nc)
    return load_pretrained(m, META['efficientnet_b4'], dest, pretrained)


def efficientnet_b5(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1.6, depth_c=2.2, ps_head=0.4, nc=nc)
    return load_pretrained(m, META['efficientnet_b5'], dest, pretrained)


def efficientnet_b6(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=1.8, depth_c=2.6, ps_head=0.5, nc=nc)
    return load_pretrained(m, META['efficientnet_b6'], dest, pretrained)


def efficientnet_b7(pretrained=False, nc=1000, dest=None):
    m = EfficientNet(width_c=2., depth_c=3.1, ps_head=0.5, nc=nc)
    return load_pretrained(m, META['efficientnet_b7'], dest, pretrained)
