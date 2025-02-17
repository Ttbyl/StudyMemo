import math
import torch
import torch.nn as nn

def SoftMax(x, dim = -1):
    x_max = x.max(dim = dim, keepdim = True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim = dim, keepdim = True)
    return x_exp / x_sum