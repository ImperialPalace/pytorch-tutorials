#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/5
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 112 * 112, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # flat
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    cnn = CNNClassifier(3, 10)
    print(cnn)

    input = torch.randn((1, 3, 224, 224))
    ouput = cnn(input)
    print(ouput.detach().numpy())


'''
python main_module.py
'''