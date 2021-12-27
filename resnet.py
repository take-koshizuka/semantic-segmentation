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