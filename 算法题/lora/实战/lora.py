'''
Author: Wangzhibo && ttbylzb11@gmail.com
Date: 2025-02-14 09:34:46
LastEditors: Wanzhiboo && ttbylzb11@gmail.com
LastEditTime: 2025-02-14 16:00:44
FilePath: /实战/lora.py
Description: 

Copyright (c) 2025 by ttbylzb11@gmail.com, All Rights Reserved. 
'''
# 写lora定义
import torch
import torch.nn as nn
import math

# lora


class lora(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=8):
        super(lora, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        # 假设输入是infeatures * outfeatures
        self.lora_a = nn.Parameter(torch.randn(self.in_features, self.r))
        self.lora_b = nn.Parameter(torch.randn(self.r, self.out_features))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.scaling * x @ self.lora_a @ self.lora_b
        return x
