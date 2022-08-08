#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/8
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime

from torchvision import transforms

from dataset import DataGenerator

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/')

args = parser.parse_args()

if __name__ == '__main__':
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    num_classes = 2

    width = 224
    height = 224

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

    test_transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.ToTensor(), normalize])

    train_loader = DataGenerator(os.path.join(args.data_path, "train"), width, height, transform)
    test_loader = DataGenerator(os.path.join(args.data_path, "valid"), width, height, test_transform)

    try:
        for batch_idx, (inputs, target) in enumerate(train_loader):
            print("input shape: {}, target shape: {}".format(inputs.shape, target.shape))

    except StopIteration:
        print("done one loop")