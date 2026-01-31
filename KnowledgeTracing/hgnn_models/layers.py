import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module): # Inherited from module
    #   in_features: size of each input sample
    #   out_features: size of each output sample
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        # Convert a non trainable type Tensor to trainable type Parameter
        # Parameter definition
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        # Parameter initialization function
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # forward function
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)  #对原始题目特征进行线性变换（矩阵乘法，线性变换，可学习矩阵）
        if self.bias is not None:
            x = x + self.bias  #加上偏置项（特征偏置）
        x = G.matmul(x) #超图卷积操作，用G聚合邻居特征（G中共享知识点的题目信息聚合）关注不同题目的题目特征）   
        return x

