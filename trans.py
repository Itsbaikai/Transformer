import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        # 初始化位置矩阵
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        # 批处理维度
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # 为输入的句子增加位置编码
        return x + self.encoding[:, :x.size(1)].detach()

# 示例用法
d_model = 512
max_len = 100
positional_encoding = PositionalEncoding(d_model, max_len)


input_sequence = torch.randn(1, max_len, d_model)
output_sequence = positional_encoding(input_sequence)
