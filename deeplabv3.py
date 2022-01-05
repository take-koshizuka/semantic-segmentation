# パッケージのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import AuxLoss

class Deeplabv3(nn.Module):
    def __init__(self, n_classes, aux_weight, backbone='resnet50'):
        super(Deeplabv3, self).__init__()
        self.n_classes = n_classes
        if backbone == 'resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=n_classes, aux_loss=(aux_weight > 0))
        elif backbone == 'resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=n_classes, aux_loss=(aux_weight > 0))
        
        if aux_weight > 0:
            self.criterion = AuxLoss(aux_weight=aux_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        return (out["out"], out["aux"])

    def parameters(self):
        params = [ 
            { 'params' : self.model.backbone.parameters(), 'lr': 1e-4 },
            { 'params' : self.model.classifier.parameters(), 'lr' : 1e-3 },
            { 'params' : self.model.aux_classifier.parameters(), 'lr' : 1e-3 }
        ]
        return params