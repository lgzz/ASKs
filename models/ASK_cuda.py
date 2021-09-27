import torch
import torch.nn as nn
import ask_cuda

any_func_list = {    
	'[[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]]' : ask_cuda.ask_d2_1245689_2345678_cuda_forward,
	'[[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]]' : ask_cuda.ask_d4_124568_234568_245678_245689_cuda_forward,
	'[[2,4,5,6,8]]' : ask_cuda.ask_d1_24568_cuda_forward,
	'[[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]]' : ask_cuda.ask_d4_24568_24568_24568_13579_cuda_forward,
	'[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]]' : ask_cuda.ask_d4_1245_2356_4578_5689_cuda_forward,
	'[[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]]' : ask_cuda.ask_d4_2456_2568_4568_2458_cuda_forward,
	'[[2,4,5],[2,5,6],[4,5,8],[5,6,8]]' : ask_cuda.ask_d4_245_256_458_568_cuda_forward,
	'[[2,5,8],[4,5,6]]' : ask_cuda.ask_d2_258_456_cuda_forward,
	'[[2,5],[4,5],[5,6],[5,8]]' : ask_cuda.ask_d4_25_45_56_58_cuda_forward,
}

class ASK_cuda(nn.Module):
    def __init__(self, inp, oup, cfg, stride=1):
        super(ASK_cuda, self).__init__()
        self.stride = stride
        self.forward_func = any_func_list[str(cfg).replace(" ","")]
        divide = len(cfg)
        lens = 0
        for i in range(divide):
            lens += len(cfg[i])
        self.weights = nn.Parameter(torch.Tensor(oup//divide, inp*lens))
    def forward(self, x):
        return self.forward_func(x, self.weights, self.stride)
