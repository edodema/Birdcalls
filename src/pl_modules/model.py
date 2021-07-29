import torch
from torch.nn import Module


class Detection(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xb):
        return xb


class Classification(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xb):
        return xb
