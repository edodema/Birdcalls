"""
Utilities that come handy.

- Environment setup:
    - get_env
    - load_envs

- Logging:
    - log_hyperparameters

- Data analysis:
    - plot_audio_len

- Spectrogram computing:
    - plot_waveform
    - plot_spectrogram
    - compute_spectrogram
    - get_spectrogram

- Preprocessing:
    - save_vocab
    - load_vocab
    - birdcall_vocabs
"""

import os
from pathlib import Path
from typing import Optional, Dict, Union, Callable, Tuple
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torchaudio
import torchaudio.transforms as T
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import librosa
import json
import numpy as np
from collections import Counter

# https://github.com/lucmos/nn-template/blob/969c36f069723d2a99ad31eb4b883160a572f651/src/common/utils.py#L13
def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


# https://github.com/lucmos/nn-template/blob/969c36f069723d2a99ad31eb4b883160a572f651/src/common/utils.py#L39
def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environmental variables.
load_envs()

# Set CWD to the project root.
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)

# Get global variable.
TRAIN_BIRDCALLS: Path = Path(get_env("TRAIN_BIRDCALLS"))
assert (
    TRAIN_BIRDCALLS.exists()
), "You must configure the TRAIN_BIRDCALLS environment variable in a .env file!"

TRAIN_SOUNDSCAPES: Path = Path(get_env("TRAIN_SOUNDSCAPES"))
assert (
    TRAIN_SOUNDSCAPES.exists()
), "You must configure the TRAIN_SOUNDSCAPES environment variable in a .env file!"

# Get global variables for vocabularies.
BIRD2IDX: Path = Path(get_env("BIRD2IDX"))
assert (
    BIRD2IDX.exists()
), "You must configure the BIRD2IDX environment variable in a .env file!"

IDX2BIRD: Path = Path(get_env("IDX2BIRD"))
assert (
    IDX2BIRD.exists()
), "You must configure the IDX2BIRD environment variable in a .env file!"

STATS_KEY: str = "stats"

# https://github.com/lucmos/nn-template/blob/main/src/common/utils.py#L56
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


def plot_audio_len(path: Union[str, Path]):
    """
    Plot the frequency of the audio files' length.
    :param path: Path of the directory containing data.
    :return:
    """
    counter = Counter()

    # Start exploring,
    dirs = [dir for dir in path.iterdir()]
    for dir in dirs:
        for file in dir.iterdir():
            # Get infos, we can express time as a function of number of frames and sample rate.
            metadata = torchaudio.info(file)

            sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames

            time = round(num_frames / sample_rate)
            minutes = time // 60
            # To get a decent approximation we care in terms of half minutes.
            seconds = 0 if time % 60 < 30 else 30
            # Just a graphic convention, I prefer to have 2.5 minutes instead of 2:30.
            if seconds == 30:
                minutes += 0.5

            # Update count.
            counter.update({minutes: 1})

    # Get an ordered list of the most frequent times.
    x = counter.keys()
    y = counter.values()

    # Show figure.
    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel("Time")
    plt.xticks(np.arange(min(x), max(x) + 1, 2), rotation=45)
    plt.tight_layout()
    fig.show()


# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#preparing-data-and-utility-functions-skip-this-section
def plot_waveform(
    waveform, sample_rate, title="Waveform", xlim=None, ylim=None
) -> None:
    """
    TODO: Maybe remove, is it necessary?
    :param waveform:
    :param sample_rate:
    :param title:
    :param xlim:
    :param ylim:
    :return:
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#preparing-data-and-utility-functions-skip-this-section
def plot_spectrogram(
    spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None
) -> None:
    """
    Plot a spectrogram.
    :param spec: Spectrogram tensor.
    :param title: Plot title.
    :param ylabel: Label on y-axis.
    :param aspect:
    :param xmax:
    :return:
    """
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def compute_spectrogram(
    audio: Union[Path, Tuple[torch.Tensor, int]],
    n_fft: int,
    win_length: Optional[int],
    hop_length: int,
    n_mels: int,
    mel: bool,
    time_window: Optional[Tuple[int, int]],
    **kwargs,
) -> torch.Tensor:
    """
    Get the spectrogram of an audio file.
    :param audio: Path of the audio file or a (waveform, sample_rate) tuple.
    :param n_fft:
    :param win_length:
    :param hop_length:
    :param n_mels:
    :param mel: If true we want melodic spectrograms.
    :param time_window: A tuple of two time values such we get the sliced spectrogram w.r.t. that window.
    :param kwargs:
    :return:
    """
    # See if we have to deal with an audio file or (waveform, sample rate).
    if isinstance(audio, Path):
        waveform, sample_rate = torchaudio.load(audio, format="ogg")
    elif isinstance(audio[0], torch.Tensor) and isinstance(audio[1], int):
        waveform = audio[0]
        sample_rate = audio[1]
    else:
        raise Exception(
            "Input audio worng, it must be either a path to an audio file or a (waveform, sample rate) tuple."
        )

    spectrogram: Callable

    if not mel:
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
    else:
        # Mel Spectrogram transform.
        spectrogram = T.MelSpectrogram(
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

    if time_window:
        # We convert the time window from seconds to frames.
        start, end = np.asarray(time_window) * sample_rate
        waveform = waveform[:, start:end]
    return spectrogram(waveform)


def get_spectrogram(
    audio: Union[Path, Tuple[torch.Tensor, int]],
    time_window: Optional[Tuple[int, int]] = None,
):
    """
    Returns the spectrogram of an audio file.
    :param audio: Path of the audio file or a (waveform, sample_rate) tuple.
    :param time_window: A tuple of two time values such we get the sliced spectrogram w.r.t. that window.
    :return:
    """
    cfg = hydra.compose(config_name="default")

    # Wrapper used to call compute_spectrogram without needing to specify too many parameters.
    def _get_spectrogram(cfg: DictConfig):
        return hydra.utils.call(
            cfg.data.spectrogram,
            audio=audio,
            time_window=time_window,
        )

    if time_window:
        s, e = time_window
        assert (
            s < e
        ), "To slice a tensor the first index has to be lower that the second."
    return _get_spectrogram(cfg)


def save_vocab(vocab: Dict, path: Union[str, Path]) -> None:
    """
    Save vocabulary to a JSON file.
    :param vocab: Dictionary object.
    :param path: Path to file.
    :return:
    """
    dump = json.dumps(vocab)
    f = open(path, "w")
    f.write(dump)
    f.close()


def load_vocab(path: Union[str, Path]) -> Dict:
    """
    Load vocabulary from a JSON file.
    :param path: Path to file.
    :return: Dictionary object i.e. the vocabulary.
    """
    f = open(path, "r")
    vocab = json.load(f)
    f.close()
    return vocab


def birdcall_vocabs(
    birdcalls_path: str, idx2bird_path: str, bird2idx_path: str, **kwargs
) -> None:
    """
    Save vocabularies mapping birds to their indexes and vice versa.
    :param birdcalls_path: Path of the parent directory under which calls are nested.
    :param idx2bird_path: Location where the vocabulary index to bird is saved.
    :param idx2bird_path: Location where the vocabulary bird to index is saved.
    :param kwargs:
    :return:
    """
    birdcalls_path = Path(birdcalls_path)

    dirs = sorted([d for d in birdcalls_path.iterdir() if d.is_dir()])

    # We build up two vocabularies: one from index to birdname and vice versa.
    idx2bird = {str(i): dir.name for i, dir in enumerate(dirs)}
    bird2idx = {dir.name: i for i, dir in enumerate(dirs)}

    # File I/O
    save_vocab(idx2bird, idx2bird_path)
    save_vocab(bird2idx, bird2idx_path)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    path = Path(cfg.data.birdcalls_datamodule.datasets.train.path)
    plot_audio_len(path=path)


if __name__ == "__main__":
    main()
