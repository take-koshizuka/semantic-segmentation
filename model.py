# ======================
# Import Libs
# ======================

import os
from utils import BCEWithLogitsLoss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pathlib
from copy import deepcopy
import cv2
import albumentations as albu
import matplotlib.pyplot as plt
import time
import copy
import torchmetrics
from utils import BCEWithLogitsLoss
from torchmetrics import MetricCollection, Accuracy, AUC

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False


class Net(nn.Module):
    def __init__(self, net, device):
        super(Net, self).__init__()
        self.net = net
        self.device = device
        self.train_criterion = BCEWithLogitsLoss()
        self.valid_criterion = BCEWithLogitsLoss()
        metrics = MetricCollection([Accuracy()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch['image'].to(self.device).float()
        y = batch['mask'].to(self.device).float()
        #batch['rgb_path']
        out = self.forward(x)
        loss = self.train_criterion(out, y)
        pred = (out > 0.0).long()
        score = self.train_metrics(pred.flatten(), y.long().flatten())
        score = { k: v.item() for k, v in score.items() }
        score['loss'] = loss
        return score
        
    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            x = batch['image'].to(self.device).float()
            y = batch['mask'].to(self.device).float()
            #batch['rgb_path']
            out = self.forward(x)
            loss = self.valid_criterion(out, y)
            pred = (out > 0.0).long()
            score = self.valid_metrics(pred.flatten(), y.long().flatten())
            score = { k: v.item() for k, v in score.items() }
            score['val_loss'] = loss.item()
        return score

    def generating_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            x = batch['image'].to(self.device).float()
            out = self.forward(x)
            rgb_fname = pathlib.Path(batch['rgb_path']).stem
        return out, rgb_fname

    def training_epoch_end(self, outputs):
        avg_loss = self.train_criterion.compute().item()
        logs = self.train_metrics.compute()
        logs = { k: v.item() for k, v in logs.items() }
        logs['avg_loss'] = avg_loss
        return { 'avg_loss' : avg_loss, 'log' : logs }

    def validation_epoch_end(self, outputs):
        avg_loss = self.valid_criterion.compute().item()
        logs = self.valid_metrics.compute()
        logs = { k: v.item() for k, v in logs.items() }
        logs['avg_loss'] = avg_loss
        return { 'avg_loss' : avg_loss, 'log' : logs }

    

    def state_dict(self, optimizer, scheduler=None):
        dic =  {
            "net": deepcopy(self.net.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict())
        }
        if not scheduler is None:
            dic["scheduler"] = deepcopy(scheduler.state_dict())
        
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint, amp=False):
        self.net.load_state_dict(checkpoint["net"])
        if amp:
            amp.load_state_dict(checkpoint["amp"])
    

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_downsample=4):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            n_downsample (int): depth of the network
        """
        super(UNet, self).__init__()

        # input and output channels for the downsampling path
        out_channels = [64 * (2 ** i) for i in range(n_downsample)]
        in_channels = [in_channels] + out_channels[:-1]

        self.down_path = nn.ModuleList(
            [self._make_unet_conv_block(ich, och) for ich, och in zip(in_channels, out_channels)]
        )

        # input channels of the upsampling path
        in_channels = [64 * (2 ** i) for i in range(n_downsample, 0, -1)]
        self.body = self._make_unet_conv_block(out_channels[-1], in_channels[0])

        self.upsamplers = nn.ModuleList(
            [self._make_upsampler(nch, nch // 2) for nch in in_channels]
        )

        self.up_path = nn.ModuleList(
            [self._make_unet_conv_block(nch, nch // 2) for nch in in_channels]
        )

        self.last = nn.Conv2d(64, n_classes, kernel_size=1)

        self._return_intermed = False

    @staticmethod
    def _make_unet_conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    @property
    def return_intermed(self):
        return self._return_intermed

    @return_intermed.setter
    def return_intermed(self, value):
        self._return_intermed = value

    def _make_upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        blocks = []
        for down in self.down_path:
            # UNet conv block increases the number of channels
            x = down(x)
            blocks.append(x)
            # Downsampling, by mass pooling
            x = F.max_pool2d(x, 2)

        x = self.body(x)

        for upsampler, up, block in zip(self.upsamplers, self.up_path, reversed(blocks)):
            # upsample and reduce number of channels
            x = upsampler(x)
            x = torch.cat([x, block], dim=1)
            # UNet conv block reduces the number of channels again
            x = up(x)

        x = self.last(x)

        if self.return_intermed:
            return x, blocks

        return x

