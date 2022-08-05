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

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 112 * 112, 1024),
            nn.Dropout(),
            nn.Linear(1024, n_classes),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1)  # flat
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    cnn = CNNClassifier(3, 10)
    print(cnn)

    input = torch.randn((1, 3, 224, 224))
    ouput = cnn(input)
    print(ouput.detach().numpy())


'''
python main_sequential.py
'''