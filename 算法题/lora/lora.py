'''
Author: Wangzhibo && ttbylzb11@gmail.com
Date: 2025-02-14 09:34:46
LastEditors: Wanzhiboo && ttbylzb11@gmail.com
LastEditTime: 2025-02-14 15:00:12
FilePath: /StudyMemo/算法题/lora/lora.py
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
        x = self.scaling * x @ self.lora_a @ self.lora_b
        return x


if __name__ == '__main__':
    model = nn.Linear(10, 20) # 这里Linear是全连接，冻结
    lora_model = lora(10, 20)
    input = torch.randn(20, 10)
    output = model(input)
    lora_output = lora_model(input)
    print(output.shape)
    print(lora_output.shape)
