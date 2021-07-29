import os
import torch
from torch.utils.data import DataLoader
import torchaudio
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.common.utils import (
    PROJECT_ROOT,
    plot_specgram,
    plot_waveform,
    plot_spectrogram,
    save_vocab,
    load_vocab,
)
import torchaudio.transforms as T
import plotly.express as px
from src.pl_data.dataset import BirdcallDataset


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    birdcalls_ds: BirdcallDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train.birdcalls, _recursive_=False
    )

    birdcalls_dl = DataLoader(
        dataset=birdcalls_ds,
        batch_size=4,
        collate_fn=BirdcallDataset.collate_fn,
        # shuffle=True,
    )

    for xb in birdcalls_dl:
        bird = xb["bird"]
        spec = xb["spec"]

        print(bird)
        print(spec)

        break


if __name__ == "__main__":
    main()
