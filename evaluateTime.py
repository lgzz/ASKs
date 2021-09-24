from __future__ import print_function
import argparse
import numpy as np
import os, time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models


parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=str,
                    help='depth of the neural network')
parser.add_argument('--anycfg', default='ASK_5a', type=str,
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def main():
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    orimodel = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    model = models.__dict__[args.arch+'_cuda_any'](dataset=args.dataset, depth=args.depth, anycfg=args.anycfg)

    if args.cuda:
        orimodel.cuda()
        model.cuda()

    orimodel.eval()
    model.eval()
    for (data, _), i in zip(test_loader,range(1100)):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        if i == 100:
            time1 = time.time()
        orimodel(data)
    time2 = time.time()
    print('基准网络模型单幅图片处理耗时: %0.3f ms' % ( (time2 - time1) ))
    
    for (data, _), i in zip(test_loader,range(1100)):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        if i == 100:
            time1 = time.time()
        model(data)
    time2 = time.time()
    print('加速后网络模型单幅图片处理耗时: %0.3f ms' % ( (time2 - time1) ))

if __name__ == '__main__':
    main()
