import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        c1 = self.resnet.layer1(x)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        c4 = self.resnet.layer4(c3)
        return c1, c2, c3, c4

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs

class Auxiliarylayers(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Auxiliarylayers, self).__init__()

        self.cbr = conv2DBatchNormRelu(
            in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        return x