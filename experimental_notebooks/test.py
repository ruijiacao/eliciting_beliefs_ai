import torch.nn as nn
import torch

# pool with window of size=3, stride=2
v = torch.tensor([[1., 2, 3, 4, 5, 6, 7]])
m = nn.AvgPool1d(kernel_size=7)
print(m(v))