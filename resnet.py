import torch
import torch.nn as nn

def get_resnet50(pretrained=True):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
    return model

def get_resnet101(pretrained=True):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained)
    return  model

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