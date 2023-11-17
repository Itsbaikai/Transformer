import torch
import torch.nn as nn

# 构建一个简单的线性层的神经网络 并使用 PyTorch 进行前向传播。
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 创建一个模型实例
model = SimpleNet()

# 定义输入数据
input_data = torch.randn(1, 10)

# 运行模型
output = model(input_data)
print(input_data)
print(output)
