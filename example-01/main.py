#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import torch
from torch import nn


def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )


def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )


class Encoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1, *args, **kwargs)
                                           for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])

    def forward(self, x):
        return self.conv_blocks(x)


class Decoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f)
                                          for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        x = self.dec_blocks(x)
        x = self.last(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes, n_classes, activation='relu'):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [64 * 32 * 32, *dec_sizes]

        self.encoder = Encoder(self.enc_sizes, activation=activation)

        self.decoder = Decoder(self.dec_sizes, n_classes)

    def forward(self, x):
        x = self.encoder(x)

        x = x.flatten(1)  # flat

        x = self.decoder(x)

        return x


if __name__ == '__main__':
    model = CNNClassifier(3, [32, 64], [512, 256], 10, activation='lrelu')
    print(model)

    input = torch.randn((1, 3, 32, 32))
    ouput = model(input)
    print(ouput.detach().numpy())
