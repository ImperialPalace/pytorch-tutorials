#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import torch

from vgg import vgg16

if __name__ == '__main__':
    num_classes = 1000

    model = vgg16(num_classes)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
