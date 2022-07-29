#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = make_layer(3)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layer(input_channel):
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    layers = []
    layers += [nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    return nn.Sequential(*layers)


def vgg16(num_class):
    return VGG(num_class)
