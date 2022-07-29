#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import torch
from torch import nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

net = nn.Sequential(
    nn.Conv2d(3,3, kernel_size=3,stride=1),
    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
).to(device)

print(net)
