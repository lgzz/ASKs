import math
import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable
from .ASK_cuda import *

__all__ = ['resnet_cuda_any']

any_resnet20_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(19)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(19)],
    'ASK_5a'  :[[[2,4,5,6,8]] for i in range(19)],
    'ASK_5b'  :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(19)],
    'ASK_4a'  :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(19)],
    'ASK_4b'  :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(19)],
    'ASK_3a'  :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(19)],
    'ASK_3b'  :[[[2,5,8],[4,5,6]] for i in range(19)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(19)],
}

any_resnet32_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(31)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(31)],
    'ASK_5a'  :[[[2,4,5,6,8]] for i in range(31)],
    'ASK_5b'  :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(31)],
    'ASK_4a'  :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(31)],
    'ASK_4b'  :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(31)],
    'ASK_3a'  :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(31)],
    'ASK_3b'  :[[[2,5,8],[4,5,6]] for i in range(31)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(31)],
}
any_resnet56_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(55)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(55)],
    'ASK_5a'  :[[[2,4,5,6,8]] for i in range(55)],
    'ASK_5b'  :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(55)],
    'ASK_4a'  :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(55)],
    'ASK_4b'  :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(55)],
    'ASK_3a'  :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(55)],
    'ASK_3b'  :[[[2,5,8],[4,5,6]] for i in range(55)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(55)],
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, cfg1, cfg2, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ASK_cuda(inplanes, cfg, cfg1, stride=stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ASK_cuda(cfg, planes, cfg2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class resnet_cuda_any(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None, anycfg='ASK_5a'):
        super(resnet_cuda_any, self).__init__()
        depth = int(depth)
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg_list = eval('any_resnet%s_cfglist'%depth)[anycfg]     
        
        self.layer = 0

        self.inplanes = 16
        self.conv1 = ASK_cuda(3, 16, self.cfg_list[self.layer])
        self.layer += 1
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2*n:3*n], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], self.cfg_list[self.layer], self.cfg_list[self.layer+1], stride, downsample))
        self.layer += 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i], self.cfg_list[self.layer], self.cfg_list[self.layer+1]))
            self.layer += 2

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = resnet_cuda_any(depth=56)
    x=Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)