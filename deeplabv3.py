# パッケージのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Deeplabv3_resnet50(nn.Module):
    def __init__(self, n_classes, aux_weight):
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
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=n_classes, aux_loss=(aux_weight > 0))
        if aux_weight > 0:
            self.criterion = AuxLoss(aux_weight=aux_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        return (out["out"], out["aux"])

class AuxLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(AuxLoss, self).__init__()
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs[0], targets, reduction='mean')
        loss_aux = F.binary_cross_entropy_with_logits(outputs[1], targets, reduction='mean')
        return loss + self.aux_weight * loss_aux