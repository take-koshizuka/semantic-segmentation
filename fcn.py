from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from base import Auxiliarylayers

class FCN(nn.Module):
    def __init__(self, n_classes, aux_weight, with_global=False, pretrained=True):
        super(FCN, self).__init__()
        self.n_classes = n_classes
        self.aux_weight = aux_weight

        self.head = FCNHead(2048, nclass, norm_layer, self._up_kwargs, with_global)
        self.aux = Auxiliarylayers(in_channels=1024, n_classes=n_classes)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.backbone(x)

        x = self.head(c4)
        x = interpolate(x, imsize, **self._up_kwargs)
        outputs = [x]
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)

        
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
