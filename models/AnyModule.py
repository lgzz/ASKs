import torch
import torch.nn as nn

pair = {
    1 : [0,0],
    2 : [0,1],
    3 : [0,2],
    4 : [1,0],
    5 : [1,1],
    6 : [1,2],
    7 : [2,0],
    8 : [2,1],
    9 : [2,2],
}

class AnyModule_d(nn.Module):
    def __init__(self, inp, oup, cfg, stride=1):
        super(AnyModule_d, self).__init__()

        self.cfg = cfg
        self.divide = len(cfg)
        for i in range(self.divide):
            self.add_module('conv%d'%i,nn.Conv2d(inp*len(cfg[i]), oup//self.divide, kernel_size=1, padding=0, bias=False, stride=stride))

    def forward(self, x):
        xx = nn.functional.pad(x,(1,1,1,1),"constant",0)
        w = x.size(2)
        out = xx[:,:,self.cfg[0][0][0]:self.cfg[0][0][0]+w,self.cfg[0][0][1]:self.cfg[0][0][1]+w]
        for i in range(1,len(self.cfg[0])):
            out = torch.cat([out,xx[:,:,self.cfg[0][i][0]:self.cfg[0][i][0]+w,self.cfg[0][i][1]:self.cfg[0][i][1]+w]], dim=1)
        out = self.conv0(out)
        for k in range(1, self.divide):
            out_k = xx[:,:,self.cfg[k][0][0]:self.cfg[k][0][0]+w,self.cfg[k][0][1]:self.cfg[k][0][1]+w]
            for i in range(1,len(self.cfg[k])):
                out_k = torch.cat([out_k,xx[:,:,self.cfg[k][i][0]:self.cfg[k][i][0]+w,self.cfg[k][i][1]:self.cfg[k][i][1]+w]], dim=1)
            out_k = eval('self.conv%d'%k)(out_k)
            out = torch.cat([out, out_k], dim=1)
        return out
