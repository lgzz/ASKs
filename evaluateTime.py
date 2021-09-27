import argparse
import time
import models
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=str,
                    help='depth of the neural network')
parser.add_argument('--anycfg', default='', type=str,
                    help='')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
kwargs = {'num_workers': 1, 'pin_memory': True}

def main():
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
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

    if args.arch.endswith('_any'):
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, anycfg=args.anycfg)
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    model.cuda()
    model.eval()

# Calculate the inference latency by averaging 1000 forward calculations after warming up the GPU for 100 times.
# Repeat 3 times, and take the smallest value.
    times = [0,0,0]
    for j in range(3):
        for (data, _), i in zip(test_loader,range(1100)):
            data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
            if i == 100:
                time1 = time.time()
            model(data)
        time2 = time.time()
        times[j] = time2 - time1
    print('function took %0.3f ms' % ( min(times) * 1000.0))

if __name__ == '__main__':
    main()
