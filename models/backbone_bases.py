import os.path

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from util.misc import is_main_process


class Resnet34Fc(nn.Module):
    def __init__(self):
        super(Resnet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.__in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class Resnet50Fc(nn.Module):
    def __init__(self):
        super(Resnet50Fc, self).__init__()

        model_resnet50 = models.resnet50(pretrained=is_main_process())
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features  # 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # torch.Size([72, 2048, 7, 7])
        x1 = self.avgpool(x)  # torch.Size([72, 2048, 1, 1])
        x1 = x1.view(x1.size(0), -1)

        return x, x1  # torch.Size([72, 2048])

    def output_num(self):
        return self.__in_features


class Resnet101Fc(nn.Module):
    def __init__(self):
        super(Resnet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        return x, x1

    def output_num(self):
        return self.__in_features


class Resnet152Fc(nn.Module):
    def __init__(self):
        super(Resnet152Fc, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.__in_features = model_resnet152.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        return x, x1

    def output_num(self):
        return self.__in_features


def build_backbone(args):
    bkb_type = args.backbone

    if bkb_type == 'rn50':
        model_bck = Resnet50Fc()

    elif bkb_type == 'rn101':
        model_bck = Resnet101Fc()

    elif bkb_type == 'rn152':
        model_bck = Resnet152Fc()

    else:
        raise ValueError("Wrong backbone type")
    return model_bck
