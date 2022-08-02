#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import random
import cv2
from PIL import Image
import torch
import torch.nn.functional as F


class DataGenerator(object):

    def __init__(self, data_path, width, height, transform):
        self.data_path = data_path
        self.width = width
        self.height = height

        self.inputs = []
        self.labels = []
        self.name_classes = []

        self.index = 0

        self.transform = transform

        self.init()

    def init(self):
        self.name_classes = os.listdir(self.data_path)
        for name in self.name_classes:
            sub_data_path = os.path.join(self.data_path, name)
            for item in os.listdir(sub_data_path):
                file_path = os.path.join(sub_data_path, item)
                self.inputs.append(file_path)
                self.labels.append(self.name_to_label(name))

        random.seed(100)
        random.shuffle(self.inputs)

        random.seed(100)
        random.shuffle(self.labels)

    def name_to_label(self, name):
        label = self.name_classes.index(name)
        return label

    def num_classes(self):
        return len(self.name_classes)

    def size(self):
        return len(self.inputs)

    def comput_inputs(self, index):
        path = self.inputs[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def comput_target(self, index):
        return F.one_hot(torch.tensor([self.labels[index]], dtype=int), num_classes=self.num_classes()).type(
            torch.float32)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.size():
            inputs = self.comput_inputs(self.index)
            targets = self.comput_target(self.index)
            self.index = self.index + 1
            return torch.unsqueeze(inputs, dim=0).cuda(), targets.cuda()
        else:
            self.index = 0
            raise StopIteration
