import torch
import torch.nn as nn
import math

class Self_Attention(nn.Module):
    def __init__(self, hidden_dim : int , dropout_rate : float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.K_proj = nn.Linear(hidden_dim, hidden_dim)
        self.V_proj = nn.Linear(hidden_dim, hidden_dim)

        self.Dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, input):
        # input shape : [batch_size, seq_len, hidden_dim]
        Q = self.Q_proj(input)
        K = self.K_proj(input)
        V = self.V_proj(input)
        # 计算Q*K/sqrt(dk)
        # Q K V : [batch_size, seq_len , hidden_dim]
        score = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        score = self.Dropout(torch.softmax(score, dim= -1))

        return score @ V

        
