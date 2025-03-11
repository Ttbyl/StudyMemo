import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.Linear = nn.Linear(in_feature,out_feature)
    
    def forward(self,input):
        return self.Linear(input)
    
class MoeLayer(nn.Module):
    def __init__(self, nums_experts, in_feature, out_feature):
        super().__init__()
        self.nums_experts = nums_experts
        self.gate = nn.Linear(in_feature, nums_experts)
        self.experts = nn.ModuleList([Linear(in_feature, out_feature) for _ in range(nums_experts)])

    def forward(self, input):
        # input shape : [batch_size , seq_len, in_feature]
        gate = self.gate(input)
        score = torch.softmax(gate, dim=-1)
        # score shape: [batch_size, seq_len, nums_experts]
        # nums_experts * [batch_size, seq_len, out_feature] --> [batch_size, seq_len, nums_experts ,out_feature]
        expert_output = torch.stack([expert(input) for expert in self.experts], dim=-2)
        output = (score.unsqueeze(-2) @ expert_output).squeeze(-2)
        # score : [batch_size, seq_len, nums_experts] --> [batch_size, seq_len, 1,  nums_experts]
        # expert_output : [batch_size, seq_len, nums_experts ,out_feature]
        # @ : [batch_size, seq_len, 1, out_feature]
        return output
    
input_size = 5
output_size = 3
num_experts = 4
seq_len = 4
batch_size = 10

model = MoeLayer(num_experts, input_size, output_size)

demo = torch.randn(batch_size, seq_len, input_size)

output = model(demo)

print(output.shape)  #
