# パッケージのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AuxLoss
from resnet import get_resnet50, get_resnet101
from base import  Backbone, conv2DBatchNormRelu, Auxiliarylayers

class PSPNet_resnet50(nn.Module):
    def __init__(self, n_classes, aux_weight, pretrained=True):
        super(PSPNet_resnet50, self).__init__()
        self.n_classes = n_classes
        self.aux_weight = aux_weight

        resnet = get_resnet50(pretrained=pretrained)
        self.backbone = Backbone(resnet)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1])
        
        self.decode_feature = DecodePSPFeature(n_classes=n_classes)

        self.aux = Auxiliarylayers(in_channels=1024, n_classes=n_classes)

        self.criterion = AuxLoss(aux_weight)
    
    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.backbone(x)
        x = self.pyramid_pooling(c4)
        output = self.decode_feature(x)
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=True)

        auxout = self.aux(c3)
        auxout = F.interpolate(auxout, size=(h, w), mode="bilinear", align_corners=True) 
        return (output, auxout)

class PSPNet_resnet101(nn.Module):
    def __init__(self, n_classes, aux_weight, pretrained=True):
        super(PSPNet_resnet101, self).__init__()
        self.n_classes = n_classes
        self.aux_weight = aux_weight

        # パラメータ設定
        img_size = 475
        resnet = get_resnet101(pretrained=pretrained)
        self.backbone = Backbone(resnet)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1])
        
        self.decode_feature = DecodePSPFeature(n_classes=n_classes)

        self.aux = Auxiliarylayers(in_channels=1024, n_classes=n_classes)

        self.criterion = AuxLoss(aux_weight)
    
    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.backbone(x)
        x = self.pyramid_pooling(c4)
        output = self.decode_feature(x)
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=True)

        auxout = self.aux(c3)
        auxout = F.interpolate(auxout, size=(h, w), mode="bilinear", align_corners=True) 
        return (output, auxout)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()

        # forwardで使用する画像サイズ
        # 各畳み込み層の出力チャネル数
        out_channels = int(in_channels / len(pool_sizes))
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear", align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(h, w), mode="bilinear", align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(h, w), mode="bilinear", align_corners=True)

        # 最終的に結合させる、dim=1でチャネル数の次元で結合
        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.cbr = conv2DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        return x

