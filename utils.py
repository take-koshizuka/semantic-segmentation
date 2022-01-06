import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import IoU
from typing import Any, Optional

class EarlyStopping(object):
    def __init__(self, monitor='loss', direction='min'):
        self.monitor = monitor
        self.direction = direction
        self.best_state = None
        if direction == 'min':
            self.monitor_values = { self.monitor : float('inf') }
        elif direction == 'max':
            self.monitor_values = { self.monitor : -float('inf') }
        else:
            raise ValueError("args: [direction] must be min or max")

    def judge(self, values):
        return (self.direction == 'min' and self.monitor_values[self.monitor] > values[self.monitor]) \
                    or (self.direction == 'max' and self.monitor_values[self.monitor] < values[self.monitor])

    def update(self, values):
        self.monitor_values[self.monitor] = values[self.monitor]


class AuxLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(AuxLoss, self).__init__()
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs[0], targets, reduction='mean')
        loss_aux = F.binary_cross_entropy_with_logits(outputs[1], targets, reduction='mean')
        return loss + self.aux_weight * loss_aux

class AuxSeLoss(nn.Module):
    def __init__(self, n_classes, aux_weight=0.4, se_weight=0.2):
        super(AuxSeLoss, self).__init__()
        self.n_classes = n_classes
        self.aux_weight = aux_weight
        self.se_weight = se_weight
    
    def forward(self, outputs, targets):
        loss = F.binary_cross_entropy_with_logits(outputs[0], targets, reduction='mean')
        loss_aux = F.binary_cross_entropy_with_logits(outputs[1], targets, reduction='mean')
        se_targets = self._get_batch_label_vector(targets, n_classes=self.n_classes).type_as(outputs[0])
        loss_se = F.binary_cross_entropy_with_logits(outputs[2], se_targets, reduction='mean')
        return loss + self.aux_weight * loss_aux + self.se_weight * loss_se
    
    @staticmethod
    def _get_batch_label_vector(target, n_classes):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, n_classes)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=n_classes, min=0,
                               max=n_classes-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class pIoU(IoU):
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            absent_score=absent_score,
            threshold=threshold,
            reduction='none',
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
        )

    def compute(self) -> torch.Tensor:
        """Computes intersection over union (IoU)"""
        v = super().compute()
        return v[1]
