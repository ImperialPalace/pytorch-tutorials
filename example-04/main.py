#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import math

import torch
from torch import nn

if __name__ == '__main__':

    a = torch.tensor([[0.2, 0.8]]).type(
        torch.float32)
    b = torch.tensor([[0, 1]]).type(
        torch.float32)
    # criterion = torch.nn.L1Loss(reduction='mean')
    # criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = torch.nn.BCELoss(reduction='mean')
    m = nn.Sigmoid()
    loss = criterion(a, b)
    print(loss)

    print(math.log(0.8))