# パッケージのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import AuxLoss

class Deeplabv3_resnet50(nn.Module):
    def __init__(self, n_classes, aux_weight):
        super(Deeplabv3_resnet50, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=n_classes, aux_loss=(aux_weight > 0))
        if aux_weight > 0:
            self.criterion = AuxLoss(aux_weight=aux_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        return (out["out"], out["aux"])

class Deeplabv3_resnet101(nn.Module):
    def __init__(self, n_classes, aux_weight):
        super(Deeplabv3_resnet101, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=n_classes, aux_loss=(aux_weight > 0))
        if aux_weight > 0:
            self.criterion = AuxLoss(aux_weight=aux_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        return (out["out"], out["aux"])
