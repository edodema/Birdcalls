import os
import torch
import torchaudio
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.common.utils import (
    PROJECT_ROOT,
    plot_specgram,
    plot_waveform,
    plot_spectrogram,
)
import torchaudio.transforms as T

import plotly.express as px


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    train_birdcalls = datamodule.datasets.train.birdcalls
    path = Path(train_birdcalls.path)
    dirs = [d for d in path.iterdir() if d.is_dir()]
    files = [f for f in dirs[0].iterdir()]
    file = files[0]

    # train_soundscapes = datamodule.datasets.train.soundscapes
    # path = Path(train_soundscapes.path)
    # files = [f for f in path.iterdir()]
    # file = files[0]

    waveform, sample_rate = torchaudio.load(file, format="ogg")
    # plot_waveform(waveform=waveform, sample_rate=sample_rate)
    # plot_specgram(waveform=waveform, sample_rate=sample_rate)

    n_fft = 1024
    win_length = None
    hop_length = 512

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    spec = spectrogram(waveform)
    plot_spectrogram(spec[0], title="Spectrogram", ylabel="Hz")

    # Mel spectrogram
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)
    plot_spectrogram(melspec[0], title="MelSpectrogram", ylabel="Mel Hz")


if __name__ == "__main__":
    main()
