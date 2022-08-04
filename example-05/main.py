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

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/')
parser.add_argument('--weights_path', help='data path', default='/work/pytorch-tutorials/vgg16-397923af.pth')
parser.add_argument('--log_interval', help='log_interval', type=int, default=100)
parser.add_argument('--checkpoint_path', help='checkpoint_path', type=str, default="./checkpoint")
parser.add_argument('--epochs', help='epochs', type=int, default=100)

args = parser.parse_args()


def train(args, model, data_loader, optimizer, epoch):
    model.train()
    try:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {}, {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, data_loader.size(), loss.item()))
    except StopIteration:
        print("done one loop")


def test(model, data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                output = model(inputs)
                test_loss += F.cross_entropy(output, targets, reduction="sum").item()

                output = F.log_softmax(output, dim=-1)
                pred = output.argmax(dim=1, keepdim=True)
                gt = targets.argmax(dim=1, keepdim=True)
                print("gt : pred == {}:{}".format(pred, gt))
                correct += pred.eq(gt).sum().item()

        test_loss /= test_loader.size()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_loader.size(),
            100. * correct / test_loader.size()))

    except StopIteration:
        print("done one loop")


if __name__ == '__main__':
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    model = vgg16(num_classes)
    model.to(device)

    a = torch.load(args.weights_path)
    a.popitem()
    a.popitem()
    a.popitem()
    a.popitem()
    a.popitem()
    a.popitem()

    model.load_state_dict(a, strict=False)

    width = 224
    height = 224

    normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

    test_transform = transforms.Compose(
        [transforms.Resize((width, height)), transforms.ToTensor(), normalize])

    train_loader = DataGenerator(os.path.join(args.data_path, "train"), width, height, transform)
    test_loader = DataGenerator(os.path.join(args.data_path, "valid"), width, height, test_transform)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

        checkpoint_path = os.path.join(args.checkpoint_path, date_time)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        path = os.path.join(checkpoint_path, "{}.pth".format(epoch))
        torch.save(model.state_dict(), path)
