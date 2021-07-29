import json

import torchaudio
from torch.utils.data import DataLoader
import torchaudio.transforms as T
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
    get_spectrogram,
)
from src.pl_data.dataset import BirdcallDataset


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    path = Path(cfg.data.datamodule.datasets.train.birdcalls.csv.birdcalls_path)
    dirs = sorted([dir for dir in path.iterdir()])
    files = [file for file in dirs[0].iterdir()]

    for i, file in enumerate(files):
        waveform, sample_rate = torchaudio.load(file, format="ogg")
        fives = waveform[:, : sample_rate * 5]

        spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=None,
            hop_length=512,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="htk",
        )

        spec = spectrogram(fives)

        plot_spectrogram(spec[0])
        if i > 5:
            break


if __name__ == "__main__":
    main()
