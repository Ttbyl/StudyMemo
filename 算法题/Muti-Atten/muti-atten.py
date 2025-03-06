import torch
import torch.nn as nn
import math

class Muti_Attention(nn.Module):
    def __init__(self, hidden_dim : int, head_num : int, dropout_rate : float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        assert head_num * self.head_dim == hidden_dim

        self.Q_proj = nn.Linear(hidden_dim, hidden_dim) # shape: [hidden_dim, head_dim * head_num]
        self.K_proj = nn.Linear(hidden_dim, hidden_dim) # shape: [hidden_dim, head_dim * head_num]
        self.V_proj = nn.Linear(hidden_dim, hidden_dim) # shape: [hidden_dim, head_dim * head_num]
        self.Output_proj = nn.Linear(hidden_dim, hidden_dim) # shape: [hidden_dim, head_dim * head_num]

        self.Dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, input, attention_mask = None):
        # input shape : [batch_size, seq_len, hidden_dim]

        batch_size, seq_len, _ = input.size()

        Q = self.Q_proj(input)
        K = self.K_proj(input)
        V = self.V_proj(input)

        # Q K V : [batch_size, seq_len , hidden_dim] --> [batch_size, head_num, seq_len, head_dim]
        Q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K_state = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V_state = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # atten_weight: [batch_size, head_num, seq_len, seq_len]
        atten_weight = Q_state @ K_state.transpose(-1, -2) / math.sqrt(self.head_dim)

        # mask 
        if attention_mask is not None:
            atten_weight = atten_weight.masked_fill(attention_mask == 0,  float("-inf"))
        
        print(atten_weight.shape)

        score = torch.softmax(atten_weight, -1)
        score = self.Dropout(score)

        # output_mid : [batch_size, head_num, seq_len, head_dim] --> [batch_size, seq_len, hidden_dim]
        output_mid = score @ V_state
        output_mid = output_mid.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.Output_proj(output_mid)


if __name__ == "__main__":
    attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
    )
     # 维度为[batch_size, head_num, seq_len, seq_len]

    x = torch.rand(3, 2, 128)
    net = Muti_Attention(128, 8)
    print(net(x, attention_mask).shape)



