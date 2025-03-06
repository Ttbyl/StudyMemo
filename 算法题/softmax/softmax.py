import torch
import torch.nn as nn
import math
# 定义一个softmax函数，用于计算softmax值
def softmax(x , dim = -1):
    # 计算x在dim维度上的最大值
    x_max = x.max(dim = dim, keepdim= True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim = dim, keepdim = True)
    return x_exp / x_sum

def mask_softmax(x, dim = -1, mask = True):
    if mask:
        x[x < 0] = float("-inf") # 如果mask为True，则把小于0的值设置为负无穷，e的负无穷为0
    # 计算x在dim维度上的最大值
    x_max = x.max(dim = dim, keepdim= True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim = dim, keepdim = True)
    return x_exp / x_sum

if __name__ == '__main__':
    input = torch.randn(2,2)
    print(input)
    output = mask_softmax(input)
    print(output)