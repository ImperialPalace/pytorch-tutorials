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


class MyModule(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.trace = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            self.trace.append(x)
        return x

model = MyModule([1, 16, 32])
import torch

model(torch.rand((4,1)))

[print(trace.shape) for trace in model.trace]