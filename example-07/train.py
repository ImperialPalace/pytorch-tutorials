#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/7/29
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os
import time
from datetime import datetime

import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from utils import *
from vgg import vgg16


def train(args, model, data_loader, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses,  top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    try:
        for batch_idx, (inputs, target) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_interval == 0:
                progress.display(batch_idx + 1)

    except StopIteration:
        print("done one loop")


def test(model, data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    try:
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(data_loader):
                inputs = inputs.cuda()
                target = target.cuda()

                output = model(inputs)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()

                output = F.log_softmax(output, dim=-1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    except StopIteration:
        print("done one loop")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--data_path', help='data path', default='/work/home/syh/vgg/data/')
parser.add_argument('--weights_path', help='data path', default='/work/pytorch-tutorials/vgg16-397923af.pth')
parser.add_argument('--log_interval', help='log_interval', type=int, default=100)
parser.add_argument('--checkpoint', help='checkpoint', type=str, default="./checkpoint")
parser.add_argument('--epochs', help='epochs', type=int, default=100)
parser.add_argument('-b', '--batch-size', default=8, type=int,
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

    criterion = torch.nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, val_loader)
        scheduler.step()

        checkpoint = os.path.join(args.checkpoint, date_time)
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

        path = os.path.join(checkpoint, "{}.pth".format(epoch))
        torch.save(model.state_dict(), path)
