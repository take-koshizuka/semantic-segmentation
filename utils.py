import torch.nn as nn
import torch.nn.functional as F

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