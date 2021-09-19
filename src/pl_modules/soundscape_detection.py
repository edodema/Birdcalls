import torch
from torchvision.models import resnet50
from torch import nn


class SoundscapeModel(nn.Module):
    # Shape of the input image (c,h,w)
    shape = torch.Size((1, 128, 313))

    def __init__(self, **kwargs):
        super().__init__()

        self.cnn = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(in_features=1000, out_features=1)

    def forward(self, xb):
        out = self.cnn(xb)
        out = self.resnet(out)
        logits = self.fc(out)

        return logits
