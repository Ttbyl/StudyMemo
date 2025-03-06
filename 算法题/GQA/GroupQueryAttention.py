'''
Author: Wangzhibo && ttbylzb11@gmail.com
Date: 2025-03-06 15:02:57
LastEditors: Wanzhiboo && ttbylzb11@gmail.com
LastEditTime: 2025-03-06 15:38:44
FilePath: /StudyMemo/算法题/GQA/GroupQueryAttention.py
Description: 

Copyright (c) 2025 by ttbylzb11@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, nums_head: int , nums_keys_values_head: int , dropoutrate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_keys_values_head = nums_keys_values_head
        assert hidden_dim % nums_head == 0 # 必须被整除
        assert nums_head % nums_keys_values_head == 0 # nums_head 必须被整除
        self.head_dim = hidden_dim // nums_head

        self.q_proj = nn.Linear(hidden_dim , hidden_dim)
        self.k_proj = nn.Linear(hidden_dim , nums_keys_values_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim , nums_keys_values_head * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim , hidden_dim)

        self.Dropout = nn.Dropout(dropoutrate) if dropoutrate > 0 else nn.Identity

    def forward(self, x , attention_mask = None):
        # x shape : [batch_size, seq_len, hidden_dim]
        batch_size , seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q shape: [batch_size, seq_len, hidden_dim] --> [batch_size, nums_head, seq_len, head_dim]
        # k v shape: [batch_size, seq_len , nums_keys_values_head] --> 
        # [batch_size, nums_keys_values_head, seq_len, head_dim]
        
        q = q.view(batch_size, seq_len, self.nums_head , self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.nums_keys_values_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.nums_keys_values_head, self.head_dim).transpose(1,2)

        # broadcast
        k = k.repeat_interleave(self.nums_head // self.nums_keys_values_head ,dim = 1)
        v = v.repeat_interleave(self.nums_head // self.nums_keys_values_head ,dim = 1)

        atten_weight = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            atten_weight = atten_weight.masked_fill(attention_mask == 0 , float("-inf"))
        
        score = torch.softmax(atten_weight, dim=-1)

        score = self.Dropout(score)

        output = score @ v
        # ouput shape: [batch_size, nums_head, seq_len, head_dim] 
        # --> [batch_size, seq_len, hidden_dim] 
        return self.o_proj(output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))

if __name__ == "__main__":
    net = GroupQueryAttention(256, 8, 4)
    x = torch.randn(8, 16, 256)
    output = net(x)
    print(output.shape)