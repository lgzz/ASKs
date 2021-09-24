from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from compute_flops import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default='16', type=str,
                    help='depth of the neural network')
parser.add_argument('--anycfg', default='ASK_5a', type=str,
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def main():

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    log = open(os.path.join(args.save, 'log_seed_{}.txt'.format(args.seed)), 'w')
    #state = {k: v for k, v in args._get_kwargs()}
    #print_log(state, log)

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.arch.endswith('_any'):
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, anycfg=args.anycfg)
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    #print_log(model, log)	
	
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_prec1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], best_prec1), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    model.eval()
    params = print_model_param_nums(model)
    flops = print_model_param_flops(model=model, input_res=32, multiply_adds=False)
    print_log('  + Number of params: %.2fM' % (params / 1e6), log)
    print_log('  + Number of FLOPs: %.5fM' % (flops / 1e6), log)

    current_learning_rate = args.lr
    for epoch in range(args.start_epoch, args.epochs):        
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            current_learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate
        print_log('\n==>>[Epoch={:03d}/{:03d}] [learning_rate={:6.4f}]'.format(epoch, args.epochs, current_learning_rate) + ' [Best : Accuracy={:.4f}]'.format(best_prec1), log)
        train(train_loader, model, optimizer, epoch, log)
        prec1 = test(test_loader, model, log, best_prec1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer': optimizer.state_dict(),
		}, is_best, filepath=args.save)
	
def train(train_loader, model, optimizer, epoch, log):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_log('Train Epoch: {:03d} [{:05d}/{:05d} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), log)

def test(test_loader, model, log, best_prec1):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print_log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Best({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), 100*max(float(correct) / len(test_loader.dataset), best_prec1)), log)
    return float(correct) / len(test_loader.dataset)

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

if __name__ == '__main__':
    main()
