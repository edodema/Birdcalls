from typing import Tuple

import torch
from torch import nn
from torch.nn import Module
from torchinfo import summary
import hydra
from omegaconf import DictConfig
from src.common.utils import PROJECT_ROOT


class Detection(Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(in_features=128 * 313, out_features=1)

    def forward(self, xb):
        b, c, w, h = xb.shape
        out = xb.view(b, c * w * h)
        out = self.fc1(out)
        return out


class Classification(Module):
    out_features = 397

    def __init__(self, out_features: int, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=30,
            stride=10,
            padding=0,
        )

        self.fc_out = nn.Linear(
            in_features=13100, out_features=Classification.out_features
        )

    def forward(self, xb):
        out = self.conv1(xb)

        # Reshape
        b, c, h, w = out.shape
        out = out.reshape((b, c * h * w))
        logits = self.fc_out(out)
        return logits

    @staticmethod
    def output_size_conv(
        input: Tuple[int, int],
        kernel: Tuple[int, int],
        padding: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Return the size of the output of a convolutional layer.
        :param input: Size of the input image.
        :param kernel: Kernel size, it is assumed to be a square.
        :param padding: Padding size.
        :param stride: Stride.
        :return: The output size.
        """
        out_w = (input[0] - kernel[0] + 2 * padding[0]) / stride[0] + 1
        out_h = (input[1] - kernel[1] + 2 * padding[1]) / stride[1] + 1
        return int(out_w), int(out_h)

    @staticmethod
    def output_size_pool(
        input: Tuple[int, int],
        pooling: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Return the size of the output of a convolutional layer.
        :param input: Size of the input image.
        :param pooling: Pooling size.
        :param stride: Stride.
        :return: The output size.
        """
        out_w = (input[0] - pooling[0]) / stride[0] + 1
        out_h = (input[1] - pooling[1]) / stride[1] + 1
        return int(out_w), int(out_h)
