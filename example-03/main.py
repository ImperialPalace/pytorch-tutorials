#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import torch
import argparse
from dataset import DataGenerator
from vgg import vgg16

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/train')
args = parser.parse_args()

if __name__ == '__main__':
    num_classes = 1000

    model = vgg16(num_classes)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    width = 224
    height = 224
    dataloader = DataGenerator(args.data_path, width, height)
    for inputs, targets in dataloader:
        print(inputs.shape, targets)
