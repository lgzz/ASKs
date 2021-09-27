import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from models.vgg_any import *
from models.resnet_any import *
from models.AnyModule import AnyModule_d
from models.ASK_cuda import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='vgg_any', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default='16', type=str,
                    help='depth of the neural network')
parser.add_argument('--anycfg', default='', type=str,
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def main():
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    orimodel = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, anycfg=args.anycfg)
    model = models.__dict__[args.arch[0:-3]+'cuda_any'](dataset=args.dataset, depth=args.depth, anycfg=args.anycfg)
    if args.cuda:
        orimodel.cuda()
        model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            orimodel.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    cfg_list = eval('any_%s%s_cfglist'%(args.arch[0:-4],args.depth))[args.anycfg]
    
    # copy weights
    conv_layers = 0
    bn_weight_data = []
    bn_bias_data = []
    bn_mean_data = []
    bn_var_data = []
    conv_weight_data = []
    linear_weight_data = []
    linear_bias_data = []
    linear_bn_weight_data = []
    linear_bn_bias_data = []
    linear_bn_mean_data = []
    linear_bn_var_data = []
    for m in orimodel.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_weight_data.append(m.weight.data.clone())
            bn_bias_data.append(m.bias.data.clone())
            bn_mean_data.append(m.running_mean.clone())
            bn_var_data.append(m.running_var.clone())
        elif isinstance(m, AnyModule_d):
            divide = len(cfg_list[conv_layers])
            weights = m.conv0.weight.data.clone().squeeze()
            for i in range(1, divide):
                weights = torch.cat([weights, eval('m.conv%d'%i).weight.data.clone().squeeze()], dim=1)
            conv_weight_data.append(weights)
        elif isinstance(m, nn.Linear):
            linear_weight_data.append(m.weight.data.clone())
            linear_bias_data.append(m.bias.data.clone())
        elif isinstance(m, nn.BatchNorm1d):
            linear_bn_weight_data.append(m.weight.data.clone())
            linear_bn_bias_data.append(m.bias.data.clone())
            linear_bn_mean_data.append(m.running_mean.clone())
            linear_bn_var_data.append(m.running_var.clone())
            
    conv_layer = 0
    bn_layer = 0
    linear_layer = 0
    linear_bn_layer = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data = bn_weight_data[bn_layer]
            m.bias.data = bn_bias_data[bn_layer]
            m.running_mean = bn_mean_data[bn_layer]
            m.running_var = bn_var_data[bn_layer]
            bn_layer += 1
        elif isinstance(m, ASK_cuda):
            m.weights.data = conv_weight_data[conv_layer]
            conv_layer += 1
        elif isinstance(m, nn.Linear):
            m.weight.data = linear_weight_data[linear_layer]
            m.bias.data = linear_bias_data[linear_layer]
            linear_layer += 1
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data = linear_bn_weight_data[linear_bn_layer]
            m.bias.data = linear_bn_bias_data[linear_bn_layer]
            m.running_mean = linear_bn_mean_data[linear_bn_layer]
            m.running_var = linear_bn_var_data[linear_bn_layer]
            linear_bn_layer += 1
    
    print('\nTest the unaccelerated model:')
    test(test_loader, orimodel)
    print('\nTest the accelerated model:')
    test(test_loader, model)

def test(test_loader, model):
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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), 100*float(correct) / len(test_loader.dataset)))
    return float(correct) / len(test_loader.dataset)

if __name__ == '__main__':
    main()
