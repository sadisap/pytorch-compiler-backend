import torch
from torch import nn


class TinyModel(nn.Module):
    def forward(self, x):
        x = x * 2
        x = x + 3
        x = torch.relu(x)
        return x