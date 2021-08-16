import torch
from torch import nn
import hydra
from omegaconf import DictConfig
from src.common.utils import PROJECT_ROOT

# TODO: Install torchsummary -> torchinfo
from torch.nn import Module


class Detection(Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, xb):
        return xb


class Classification(Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(in_features=10, out_features=1)

    def forward(self, xb):
        out = self.fc1(xb)
        return out


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    classification = hydra.utils.instantiate(
        cfg.model.classification, _recursive_=False
    )
    print(classification)


if __name__ == "__main__":
    main()
