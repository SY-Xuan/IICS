import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from .backbones.resnet import AIBNResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(
            m.weight.data, a=0,
            mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class ft_net_intra(nn.Module):
    def __init__(self, num_classes, stride=1):
        super(ft_net_intra, self).__init__()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3])

        self.model = model_ft
        self.classifier = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, num, bias=False))
             for num in num_classes])
        for classifier_one in self.classifier:
            init.normal_(classifier_one[1].weight.data, std=0.001)
            init.constant_(classifier_one[0].weight.data, 1.0)
            init.constant_(classifier_one[0].bias.data, 0.0)
            classifier_one[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x, k=0):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier[k](x)
        return x


class ft_net_inter(nn.Module):
    def __init__(self, num_classes, stride=1):
        super(ft_net_inter, self).__init__()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3])

        self.model = model_ft
        self.classifier = nn.Sequential(nn.BatchNorm1d(
            2048), nn.Linear(2048, num_classes, bias=False))
        init.normal_(self.classifier[1].weight.data, std=0.001)
        init.constant_(self.classifier[0].weight.data, 1.0)
        init.constant_(self.classifier[0].bias.data, 0.0)
        self.classifier[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1))
        prob = self.classifier(x)
        return prob, x
