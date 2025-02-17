import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # 对于小模型而言，这种更节省效率，但是对于大模型更占用显存
        self.QKV_proj = nn.Linear(hidden_dim, hidden_dim * 3)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x shape : (batch_size, seq_len , dim)
        QKV = self.QKV_proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        score = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        print(softmax(score, dim=-1))

        score = self.dropout(softmax(score, dim=-1))

        return score @ V


if __name__ == "__main__":
    input = torch.rand(3, 4, 2)
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0]
        ]
    )
    mask = mask.unsqueeze(dim=1).repeat(1, 4, 1)
    print(mask.shape)
    net = SelfAttention(2)
    output = net(input, mask)
    print(output)
