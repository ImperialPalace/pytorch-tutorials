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

from torchvision import transforms

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/train')
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    model = vgg16(num_classes)
    model.to(device)

    width = 224
    height = 224

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

    dataloader = DataGenerator(args.data_path, width, height, transform)

    for inputs, targets in dataloader:
        output = model(inputs)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output, targets)
        print(loss)
