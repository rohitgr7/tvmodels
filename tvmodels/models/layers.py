from torch import nn
from torch.nn.utils import spectral_norm

__all__ = ['Bottleneck', 'SEBlock', 'Flatten', 'conv2d', 'relu']


class Bottleneck(nn.Module):

    def forward(self, x):
        print(x.shape)
        out = self.net(x)
        print(self.net)
        print(out.shape)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return self.relu(out)


class SEBlock(nn.Module):

    def __init__(self, nc, ratio=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d(nc, nc // ratio, 1), relu(),
            conv2d(nc // ratio, nc, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def _init_default(m, func=nn.init.kaiming_normal_):
    if hasattr(m, 'weight'):
        func(m.weight)
    if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.)
    return m


def conv2d(ni, nf, ksz, stride=1, pad=0, groups=1, bn=False, spectr=False):
    conv = nn.Conv2d(ni, nf, ksz, stride, pad, bias=not bn, groups=groups)
    conv = _init_default(conv)
    if spectr:
        conv = spectral_norm(conv)
    layers = [conv]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


def relu(leaky=0.):
    return nn.LeakyReLU(leaky, True) if leaky > 0. else nn.ReLU(True)
