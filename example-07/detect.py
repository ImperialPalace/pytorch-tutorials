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
import torch.nn.functional as F
from torchvision import transforms
import os
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

from torch import nn
import math

import cv2
import time

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/')
parser.add_argument('--weights_path', help='data path',
                    default='/work/pytorch-tutorials/example-05/checkpoint/2022-08-02_16:18:15/3.pth')
parser.add_argument('--output_path', help='output path', default='./output')

args = parser.parse_args()

if __name__ == '__main__':
    st = time.time()

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    output_path = os.path.join(args.output_path, date_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Create dir : {}".format(output_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    model = vgg16(num_classes)
    model.to(device)

    model.load_state_dict(torch.load(args.weights_path), strict=False)

    width = 224
    height = 224

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    test_transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.ToTensor(), normalize])

    test_loader = DataGenerator(os.path.join(args.data_path, "valid"), width, height, test_transform)

    model.eval()
    correct = 0
    test_loss = 0
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                output = model(inputs)
                output = F.log_softmax(output, dim=-1)
                pred = output.argmax(dim=1, keepdim=True)
                name_class = test_loader.label_to_name(pred.item())
                path = test_loader.get_current_image()

                image = cv2.imread(path)

                new_image = cv2.putText(
                    img=image,
                    text=name_class,
                    org=(10, 100),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=(125, 246, 55),
                    thickness=1
                )
                out = os.path.join(output_path, os.path.basename(path))
                cv2.imwrite(out, new_image)

    except StopIteration:
        print("done one loop")

    print("Save data: {}".format(output_path))

    print("Done, use time: {0:.2f} ms".format((time.time() - st) * 1000))
