from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import get_resnet50, get_resnet101
from base import  Backbone, Auxiliarylayers
from utils import AuxLoss

class FCN(nn.Module):
    def __init__(self, n_classes, aux_weight, backbone='resnet50', pretrained=True):
        super(FCN, self).__init__()
        self.n_classes = n_classes
        self.aux_weight = aux_weight

        if backbone == 'resnet50':
            resnet = get_resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            resnet = get_resnet101(pretrained=pretrained)
        
        self.backbone = Backbone(resnet)
        self.head = FCNHead(2048, n_classes)
        self.aux = Auxiliarylayers(in_channels=1024, n_classes=n_classes)
        self.criterion = AuxLoss(aux_weight)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.backbone(x)
        out = self.head(c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        
        auxout = self.aux(c3)
        auxout = F.interpolate(auxout, size=(h, w), mode="bilinear", align_corners=True)
    
        return (out, auxout)

    def parameters(self):
        params = [ 
            { 'params' : self.backbone.parameters(), 'lr': 1e-4 },
            { 'params' : self.head.parameters(), 'lr' : 1e-3 },
            { 'params' : self.aux.parameters(), 'lr' : 1e-3 }
        ]
        return params

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv(x)
