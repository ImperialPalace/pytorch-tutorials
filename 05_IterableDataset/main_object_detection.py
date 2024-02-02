#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/9
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse

import torch
from PIL import Image
from torchvision import datasets, transforms
import os
import cv2


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, label_dir, img_ext, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            label_dir: Mask file directory.
            img_ext (str): Image file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── labels
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = Image.open(os.path.join(self.img_dir, img_id + self.img_ext))

        label = 100
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_data(data_path):
    img_ids_path = os.path.join(data_path, "ImageSets/Main/trainval.txt")
    with open(img_ids_path, "r") as fd:
        img_ids = fd.read().splitlines()

    img_dir = os.path.join(data_path, "JPEGImages")
    label_dir = os.path.join(data_path, "labels")

    img_ext = ".jpg"
    return img_ids, img_dir, label_dir, img_ext


parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/document/VOCdevkit/VOCdevkit_with_labels/VOC2012/')
parser.add_argument('--output_path', help='output path', default='./output')

args = parser.parse_args()

if __name__ == '__main__':
    # Datasets
    img_ids, img_dir, label_dir, img_ext = get_data(args.data_path)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]),
    ])

    train_set = VOCDataset(img_ids, img_dir, label_dir, img_ext, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    for image,taget in train_loader:
        print(image.shape,taget.shape)
