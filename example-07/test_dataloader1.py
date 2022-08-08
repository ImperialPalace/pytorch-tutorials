#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/8
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime

import torchvision.datasets as datasets
from torchvision import transforms

from utils import *
import os

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()

if __name__ == '__main__':
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")


    num_classes = 2

    width = 224
    height = 224

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'valid')

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    try:
        for batch_idx, (inputs, target) in enumerate(train_loader):
            print("input shape: {}, target shape: {}, target value: {}".format(inputs.shape, target.shape, target))

    except StopIteration:
        print("done one loop")
