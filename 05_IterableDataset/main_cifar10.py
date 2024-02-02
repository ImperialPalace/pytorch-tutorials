#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/8
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms

if __name__ == '__main__':
    # Datasets
    data = "./input_data"

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(data, train=True, transform=train_transforms, download=True)
    test_set = datasets.CIFAR10(data, train=False, transform=test_transforms, download=False)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)

    for item in train_loader:
        print(item)


