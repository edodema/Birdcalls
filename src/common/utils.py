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
    - random_oversampler

- NN counting:
    - cnn_size
    - multiple_cnn_size
    - pad_size
    - pool_size
    - fc_params
    - cnn_params
    - cnnxfc_params
    - cnn_kernel
"""

import os
from pathlib import Path
from typing import Optional, Dict, Union, Callable, Tuple
import dotenv
import hydra
import pandas as pd
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


def random_oversampler(
    df: pd.DataFrame,
    target: Tuple[str, Union[int, str]],
    n_samples: int,
    mode: str = "==",
):
    """
    A random oversampler for unbalanced datasets.
    :param df: DataFrame object.
    :param target: A tuple containing the column of the dataframe and its respective class we want to augment.
    :param n_samples: The number of samples we want to reach, often the length of the most represented class.
    :param mode: Defines how we want to filter data:
        - "==" Then we filter for samples whose value is equal to target[1].
        - "!=" Then we filter for samples whose value is different from target[1].
    :return: An augmented DataFrame.
    """
    target_class, target_value = target

    # Sample random points from the target distribution and append them to df.
    if mode == "==":
        distribution = df[df[target_class] == target_value]
    elif mode == "!=":
        distribution = df[df[target_class] != target_value]
    else:
        raise Exception("Select a valid mode.")

    n_distr = len(distribution)

    assert (
        n_samples > n_distr
    ), "The number of samples for the chosen class is higher than the number to be reached, change n_samples."

    samples = np.random.randint(0, n_distr, n_samples - n_distr, dtype=int)
    new_samples = distribution.iloc[samples]

    return df.append(other=new_samples, ignore_index=True)


def cnn_size(
    input: Tuple[int, int],
    kernel: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """
    Return the size of the output of a convolutional layer.
    :param input: Size of the input image.
    :param kernel: Kernel size, it is assumed to be a square.
    :param padding: Padding size.
    :param stride: Stride.
    :return: The output size.
    """
    if isinstance(kernel, int):
        kernel = (kernel, kernel)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    out_w = (input[0] - kernel[0] + 2 * padding[0]) / stride[0] + 1
    out_h = (input[1] - kernel[1] + 2 * padding[1]) / stride[1] + 1
    return int(out_w), int(out_h)


def multiple_cnn_size(
    input: Tuple[int, int],
    kernel: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
    n_convs: int = 1,
) -> Tuple[int, int]:
    """
    Return the size of the output of more than one convolutional layer. This function can replace cnn_size.
    :param input: Size of the input image.
    :param kernel: Kernel size, it is assumed to be a square.
    :param padding: Padding size.
    :param stride: Stride.
    :param n_convs: Number of convolutions.
    :return: The output size.
    """
    if isinstance(kernel, int):
        kernel = (kernel, kernel)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    # Function that computes our function.
    def f(i, k, p, s):
        count = 0
        for j in range(0, n_convs):
            s_j = s ** j
            count += -s_j * k + 2 * s_j * p + s_j * s
        count += i
        return count / (s ** n_convs)

    out_w = f(i=input[0], k=kernel[0], p=padding[0], s=stride[0])
    out_h = f(i=input[1], k=kernel[1], p=padding[1], s=stride[1])
    return int(out_w), int(out_h)


def pad_size(
    input: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """
    Return the size of the output of a convolutional layer.
    :param input: Size of the input image.
    :param padding: Pooling size.
    :param stride: Stride.
    :return: The output size.
    """
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    out_w = (input[0] - padding[0]) / stride[0] + 1
    out_h = (input[1] - padding[1]) / stride[1] + 1
    return int(out_w), int(out_h)


def pool_size(
    input: Union[int, Tuple[int, int]],
    pooling: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Return the size of the output of a convolutional layer.
    :param input: Size of the input image.
    :param pooling: Pooling size.
    :return: The output size.
    """
    if isinstance(pooling, int):
        pooling = (pooling, pooling)

    out_w = input[0] / pooling[0]
    out_h = input[1] / pooling[1]
    return int(out_w), int(out_h)


def fc_params(in_features: int, out_features: int, bias: bool = True):
    """
    Return the number of parameters in a linear layer.
    :param in_features: Size of input vector.
    :param out_features: Size of output vector.
    :param bias: If true count bias too.
    :return: The number of parameters.
    """
    m = out_features + 1 if bias else out_features
    return in_features * m


def cnn_params(kernel: int, in_channels: int, out_channels: int, bias: bool = True):
    """
    Return the number of parameters in a CNN.
    :param kernel: Kernel size, it is assumed to be squared.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels i.e. number of kernels.
    :param bias: If true count bias as well.
    :return: The number of parameters.
    """
    w = kernel * kernel * in_channels * out_channels
    b = out_channels if bias else 0
    return w + b


def cnnxfc_params(
    image_size: Tuple[int, int], n_channels: int, out_features: int, bias: bool = True
):
    """
    Return the number of parameters in a CNN followe by a linear layer.
    :param image_size: Size of the output of the CNN.
    :param n_channels: Number of the image's channels.
    :param out_features: Neurons in the linear layer.
    :param bias: If true count bias.
    :return: Number of parameters.
    """
    w, h = image_size
    weights = w * h * n_channels * out_features
    biases = out_features if bias else 0
    return weights + biases


def cnn_kernel(
    input: Tuple[int, int],
    output: Tuple[int, int],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
):
    """
    Given all the parameters of a convolution, except the filter's size, compute it.
    :param input: Size of the input image.
    :param output: Size of the feature map.
    :param padding: Padding size.
    :param stride: Stride.
    :return: The kernel size.
    """
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    kernel_w = input[0] + 2 * padding[0] - stride[0] * (output[0] - 1)
    kernel_h = input[1] + 2 * padding[1] - stride[1] * (output[1] - 1)
    return kernel_w, kernel_h
