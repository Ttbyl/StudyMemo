import torch
import torch.nn
import math
import warnings
class LinearLora(torch.nn.Module):
    def __init__(self,input_features, output_features, r=4, lora_alpha=1, dropout= 0.1 ,merge=False):
        super(LinearLora, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        self.merge = merge 

        self.linear = torch.nn.Linear(input_features, output_features)

        if self.r > 0:
            self.lora_a = torch.nn.Parameter(torch.randn(input_features, r))
            self.lora_b = torch.nn.Parameter(torch.randn(r, output_features))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameter()
        
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        if merge:
            self.merge_weight()
        else:
            warnings.warn("not merge weight")

    def merge_weight(self):
        if self.r > 0:
            self.linear.weight.data += (self.scaling * self.lora_a @ self.lora_b).T # 这里需要注意转置问题
        else:
            assert self.r == 0, "rank == 0, but set merge = True"
    
    def unmerge_weight(self):
        if self.r > 0:
            self.linear.weight.data -= (self.scaling * self.lora_a @ self.lora_b).T
        else:
            assert self.r == 0, "rank == 0, but set merge = True"
        

    def reset_parameter(self):
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b)

    def forward(self, x):
        if self.merge and self.r > 0:
            x = self.linear(x) + self.scaling * x @ self.lora_a @ self.lora_b
        else:
            x = self.linear(x)
        x = self.dropout(x)
        return x

# test
if __name__ == '__main__':
    batch_size = 32
    seq_len = 128
    input_features = 768
    output_features = 512
    r = 4
    lora_alpha = 1
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, input_features)

    lora_layer = LinearLora(input_features, output_features, r, lora_alpha, dropout, merge=False)
    out = lora_layer(x)

    lora_layer.merge_weight()
    out_merge = lora_layer(x)

    print("difference:", (out - out_merge))
