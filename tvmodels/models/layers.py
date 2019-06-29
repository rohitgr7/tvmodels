import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ['Conv2dSamePad', 'DropConnect', 'Flatten',
           'Swish', 'relu', 'conv2d', 'conv_bn', 'convsame_bn_swish']


class Conv2dSamePad(nn.Conv2d):

    def __init__(self, ni, nf, ksz, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(ni, nf, ksz, stride, 0, dilation, groups, bias)
        self.stride, self.ksz = stride, ksz

    def _get_pad(self, sz):
        out_sz = (sz + self.stride - 1) // self.stride
        return max(0, (out_sz - 1) * self.stride + (self.ksz - 1) * self.dilation[0] + 1 - sz)

    def forward(self, x):
        pad_h, pad_w = self._get_pad(x.shape[2]), self._get_pad(x.shape[3])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w //
                          2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DropConnect(nn.Module):

    def __init__(self, ps):
        super().__init__()
        self.ps = 1 - ps

    def forward(self, x):
        if not self.training:
            return x
        rand_tensor = self.ps + \
            torch.rand([x.size(0), 1, 1, 1], dtype=x.dtype, device=x.device)
        return x.div(self.ps) * rand_tensor.floor()


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Swish(nn.Module):

    def forward(self, x):
        return x * x.sigmoid()


def relu(leaky=0.):
    return nn.LeakyReLU(leaky, True) if leaky > 0. else nn.ReLU(True)


def _init_default(m, func=nn.init.kaiming_normal_):
    if hasattr(m, 'weight'):
        func(m.weight)
    if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.)
    return m


def conv2d(ni, nf, ksz, stride=1, pad=0, dilation=1, groups=1, bias=False, spectr=False):
    conv = nn.Conv2d(ni, nf, ksz, stride, pad, dilation, groups, bias)
    conv = _init_default(conv)
    if spectr:
        conv = spectral_norm(conv)
    return conv


def conv_bn(ni, nf, ksz, stride=1, pad=0, dilation=1, groups=1, bn=False, spectr=False):
    layers = [conv2d(ni, nf, ksz, stride, pad, dilation,
                     groups, bias=not bn, spectr=spectr)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return layers


def convsame_bn_swish(ni, nf, ksz, stride=1, groups=1, swish=True):
    layers = [Conv2dSamePad(ni, nf, ksz, stride, groups=groups, bias=False),
              nn.BatchNorm2d(nf, 1e-3, 0.01)]
    if swish:
        layers.append(Swish())
    return layers
