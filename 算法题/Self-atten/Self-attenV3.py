'''
Author: Wangzhibo && ttbylzb11@gmail.com
Date: 2025-02-17 17:34:11
LastEditors: Wanzhiboo && ttbylzb11@gmail.com
LastEditTime: 2025-03-04 10:23:47
FilePath: /StudyMemo/算法题/Self-atten/Self-attenV3.py
Description: 

Copyright (c) 2025 by ttbylzb11@gmail.com, All Rights Reserved. 
'''
import math
import torch
import torch.nn as nn

def SoftMax(x, dim = -1):
    x_max = x.max(dim = dim, keepdim = True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim = dim, keepdim = True)
    return x_exp / x_sum

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int , dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.K_proj = nn.Linear(hidden_dim, hidden_dim)
        self.V_proj = nn.Linear(hidden_dim, hidden_dim)

        self.Dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity

    def forward(self, x, mask = None):
        # x shape: [batch_size, seq_len, hidden_dim]
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        # Q  K shape: [batch_size, seq_len, hidden_dim]
        score = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        # score shape: [batch_size, seq_len, seq_len]

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf")) # 替换0为-inf

        score = SoftMax(score, dim = -1)
        print(score)
        score = self.Dropout(score)

        return score @ V

if __name__ == "__main__":
    input = torch.randn(4,3,2)
    
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ]
    )
    # mask shape : [3,3] -> [4,3,3]
    mask = mask.unsqueeze(0).repeat(4, 1, 1)
    print(mask.shape)
    Net = SelfAttention(2)
    output = Net(input)
    output_mask = Net(input, mask)
    