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


def conv_block(in_c, out_c, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_c, out_c, *args, **kwargs),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU())


class CNNClassifier(nn.Module):

    def __init__(self, in_c, enc_sizes, n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]

        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1)
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*conv_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(128 * 32 * 32, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)  # flat

        x = self.decoder(x)

        return x


if __name__ == '__main__':
    cnn = CNNClassifier(3, [32, 64, 128], 10)
    print(cnn)

    input = torch.randn((1, 3, 32, 32))
    ouput = cnn(input)
    print(ouput.detach().numpy())

'''
python main_sequential3.py
'''
