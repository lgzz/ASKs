import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from .ASK_cuda import *

__all__ = ['vgg_cuda_any']

defaultcfg = {
    '11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    '19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}
any_vgg11_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(8)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(8)],
    'ASK_5a' :[[[2,4,5,6,8]] for i in range(8)],
    'ASK_5b' :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(8)],
    'ASK_4a' :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(8)],
    'ASK_4b' :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(8)],
    'ASK_3a' :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(8)],
    'ASK_3b' :[[[2,5,8],[4,5,6]] for i in range(8)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(8)],
}
any_vgg13_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(10)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(10)],
    'ASK_5a'  :[[[2,4,5,6,8]] for i in range(10)],
    'ASK_5b'  :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(10)],
    'ASK_4a'  :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(10)],
    'ASK_4b'  :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(10)],
    'ASK_3a'  :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(10)],
    'ASK_3b'  :[[[2,5,8],[4,5,6]] for i in range(10)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(10)],
}
any_vgg16_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(13)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(13)],
    'ASK_5a' :[[[2,4,5,6,8]] for i in range(13)],
    'ASK_5b' :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(13)],
    'ASK_4a' :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(13)],
    'ASK_4b' :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(13)],
    'ASK_3a' :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(13)],
    'ASK_3b' :[[[2,5,8],[4,5,6]] for i in range(13)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(13)],
}
any_vgg19_cfglist = {
    'ASK_7'  :[[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]] for i in range(16)],
    'ASK_6'  :[[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]] for i in range(16)],
    'ASK_5a' :[[[2,4,5,6,8]] for i in range(16)],
    'ASK_5b' :[[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]] for i in range(16)],
    'ASK_4a' :[[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]] for i in range(16)],
    'ASK_4b' :[[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]] for i in range(16)],
    'ASK_3a' :[[[2,4,5],[2,5,6],[4,5,8],[5,6,8]] for i in range(16)],
    'ASK_3b' :[[[2,5,8],[4,5,6]] for i in range(16)],
    'ASK_2'  :[[[2,5],[4,5],[5,6],[5,8]] for i in range(16)],
}
class vgg_cuda_any(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, anycfg='ASK_5a'):
        super(vgg_cuda_any, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.cfg_list = eval('any_vgg%s_cfglist'%depth)[anycfg]        

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        layer = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = ASK_cuda(in_channels, v, self.cfg_list[layer])
                layer += 1
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = vgg_cuda_any()
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)