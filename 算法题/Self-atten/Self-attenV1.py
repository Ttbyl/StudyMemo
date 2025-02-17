import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    x_max = x.max(dim = dim, keepdim= True)[0] # 求最大值，这里max会返回最大值和最大值索引
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim = dim, keepdim= True)
    return x_exp / x_sum


class SelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # Q K V proj
        self.Q_proj = nn.Linear(dim, dim)
        self.K_proj = nn.Linear(dim, dim)
        self.V_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x shape : (batch_size, seq_len, dim)
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        # score : softmax(Q @ K.T / math.sqrt(dim))
        # Q shape: (batch_size, seq_len, dim)
        # K.transpose(-1 ,-2) shape: (batch_size, dim , seq_len)
        # score shape : (batch_size, dim , dim)
        # score1 = torch.softmax(Q @ K.transpose(-1, -2) /
        #                       math.sqrt(self.dim), dim=-1)
        
        score = softmax(Q @ K.transpose(-1,-2) / math.sqrt(self.dim), dim = -1)
        print(score)

        return score @ V


if __name__ == "__main__":
    input = torch.randn(4, 2, 4)
    net = SelfAttention(4)
    output = net(input)
    print(output)
