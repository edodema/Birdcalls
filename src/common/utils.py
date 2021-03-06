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
    - plot_spectrogram
    - compute_spectrogram
    - get_spectrogram

- Preprocessing:
    - save_vocab
    - load_vocab
    - get_birds_names
    - get_ordered_vocab
    - get_most_common_class
    - random_oversampler
    - split_dataset

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
from typing import Optional, Dict, Union, Callable, Tuple, List
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

    Args:
        env_name: the name of the environment variable
        default: the default (optional) value for the environment variable

    Returns:
        The value of the environment variable
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

    Args:
        env_file: the file that defines the environment variables to use. If None
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

# Get global variables.
BIRDCALLS_DIR: Path = Path(get_env("BIRDCALLS_DIR"))
assert (
    BIRDCALLS_DIR.exists()
), "You must configure the BIRDCALLS_DIR environment variable in a .env file!"

SOUNDSCAPES_DIR: Path = Path(get_env("SOUNDSCAPES_DIR"))
assert (
    SOUNDSCAPES_DIR.exists()
), "You must configure the SOUNDSCAPES_DIR environment variable in a .env file!"

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
        - number of trainable models parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of models parameters
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
    # (this is just a trick to prevent trainer from logging hparams of models, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


def plot_audio_len(path: Union[str, Path]):
    """
    Plot the frequency of the audio files' length.
    Args:
        path: Path of the directory containing data.
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
def plot_spectrogram(
    spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None
) -> None:
    """
    Plot a spectrogram.
    Args:
        spec: Spectrogram tensor.
        title: Plot title.
        ylabel: Label on y-axis.
        aspect: Aspect ratio.
        xmax: Maximum value on the x-axis.
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
    Args:
        audio: Path of the audio file or a (waveform, sample_rate) tuple.
        n_fft:
        win_length:
        hop_length:
        n_mels:
        mel: If true we want melodic spectrograms.
        time_window: A tuple of two time values such we get the sliced spectrogram w.r.t. that window.
        kwargs:
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
    Args:
        audio: Path of the audio file or a (waveform, sample_rate) tuple.
        time_window: A tuple of two time values such we get the sliced spectrogram w.r.t. that window.
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
    Args:
        vocab: Dictionary object.
        path: Path to file.
    """
    dump = json.dumps(vocab)
    f = open(path, "w")
    f.write(dump)
    f.close()


def load_vocab(path: Union[str, Path]) -> Dict:
    """
    Load vocabulary from a JSON file.
    Args:
        path: Path to file.
    Returns:
        Dictionary object i.e. the vocabulary.
    """
    f = open(path, "r")
    vocab = json.load(f)
    f.close()
    return vocab


def get_birds_names(csv_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Get a dictionary associating the id of a bird with its scientific and common name.
    Args:
        csv_path: Path of the CSV file.

    Returns:
        A dictionary mapping a species' id with its scientific and common name.
    """
    df = pd.read_csv(csv_path)
    labels = df["primary_label"].values
    scientific = df["scientific_name"].values
    common = df["common_name"].values

    out = {}
    for label, s_name, c_name in zip(labels, scientific, common):
        if label in out:
            continue
        out[label] = [s_name, c_name]

    return out


def birdcall_vocabs(
    birdcalls_path: str, idx2bird_path: str, bird2idx_path: str, **kwargs
) -> None:
    """
    Save vocabularies mapping birds to their indexes and vice versa.
    Args:
        birdcalls_path: Path of the parent directory under which calls are nested.
        idx2bird_path: Location where the vocabulary index to bird is saved.
        idx2bird_path: Location where the vocabulary bird to index is saved.
        kwargs:
    """
    birdcalls_path = Path(birdcalls_path)

    dirs = sorted([d for d in birdcalls_path.iterdir() if d.is_dir()])

    # We build up two vocabularies: one from index to birdname and vice versa.
    idx2bird = {str(i): dir.name for i, dir in enumerate(dirs)}
    bird2idx = {dir.name: i for i, dir in enumerate(dirs)}

    # File I/O
    save_vocab(idx2bird, idx2bird_path)
    save_vocab(bird2idx, bird2idx_path)


def get_most_common_class(df: pd.DataFrame, col: str):
    """
    Get some insights on the most common class of a column in a dataframe.
    Args:
        df: Dataframe object.
        col: The column we are interested in.

    Returns:
        - class: Most frequent class.
        - len: Number of samples belonging to the most frequent class.
        - function: A vectorized function to filter primary labels.
        - counter: Counter object.
    """
    col = df[col].values

    # If there are elements split by spaces (as in birds) we only keep the primary label.
    if isinstance(col[0], str):

        def f(x: str):
            return x.split()[0]

    else:

        def f(_):
            pass

    fv = np.vectorize(f)
    col = fv(col)

    # Take the most frequent class.
    count = Counter(col)
    most_common_class, most_common_n = count.most_common(1)[0]
    return {
        "class": most_common_class,
        "len": most_common_n,
        "function": fv,
        "counter": count,
    }


def random_oversampler(
    df: pd.DataFrame,
    target: Tuple[str, Union[int, str, None]],
    n_samples: Optional[int] = None,
    mode: Optional[str] = None,
):
    """
    A random oversampler for unbalanced datasets.
    Args:
        df: DataFrame object.
        target: A tuple containing the column of the dataframe and its respective class we want to augment.
            If no target is specified then all classes are automatically balanced to the most frequent one.
        n_samples: The number of samples we want to reach, often the length of the most represented class.
        mode: Defines how we want to filter data:
            - "==" Then we filter for samples whose value is equal to target[1].
            - "!=" Then we filter for samples whose value is different from target[1].
    Returns:
        An augmented DataFrame.
    """

    # - If target value is None n_samples and mode must be too.
    # - If target value is specified n_samples and mode must be too.
    phi = bool(target[1]) or not mode
    psi = not target[1] or bool(mode)

    assert (
        phi and psi
    ), "Either target, n_samples and mode are specified or none of them is."

    if target[1]:
        # We specified a number to oversample to.

        target_col, target_value = target

        # Sample random points from the target distribution and append them to df.
        if mode == "==":
            distribution = df[df[target_col] == target_value]
        elif mode == "!=":
            distribution = df[df[target_col] != target_value]
        else:
            raise Exception("Select a valid mode.")

        n_distr = len(distribution)

        # If n_samples is not defined we consider the most frequent class.
        if not n_samples:
            n_samples = get_most_common_class(df=df, col=target_col)["len"]

        assert (
            n_samples > n_distr
        ), "The number of samples for the chosen class is higher than the number to be reached, change n_samples."

        samples = np.random.randint(0, n_distr, n_samples - n_distr, dtype=int)
        new_samples = distribution.iloc[samples]

        df = df.append(other=new_samples, ignore_index=True)
    else:
        # We balance all classes.
        target_col = target[0]

        most_common = get_most_common_class(df=df, col=target_col)
        count = most_common["counter"]
        most_common_n = most_common["len"]
        fv = most_common["function"]

        # Oversampling for each class.
        for k, v in count.items():
            # Data we want to sample from.
            distribution = df[fv(df[target_col]) == k]

            # Repeat samples.
            to_n = most_common_n - v
            if to_n == 0:
                continue

            samples = np.random.randint(0, v, to_n, dtype=int)
            new_samples = distribution.iloc[samples]
            df = df.append(other=new_samples, ignore_index=True)

    return df


def split_dataset(
    csv: Union[str, Path, pd.DataFrame],
    save_path_train: Union[str, Path, None] = None,
    save_path_eval: Union[str, Path, None] = None,
    p: float = 0.8,
    autosave: bool = False,
):
    """
    Split a CSV dataset in train and evaluation according to percentage p.
    Args:
        csv_path: Path of the dataset as a CSV file.
        save_path_train: Path to save to the train CSV file.
        save_path_eval: Path to save to the eval CSV file.
        p: Percentage of the train set expressed as a float.
        autosave: If true save the files, save_path_train and save_path_eval must be specified.
    Returns:
        Train and eval dataframes.
    """
    assert (
        0 <= p and p <= 1
    ), "The probability of a sample being in the train set must be between 0 and 1!"

    if isinstance(csv, str) or isinstance(csv, Path):
        df = pd.read_csv(csv)
    else:
        df = csv

    df = df.sample(frac=1)

    idx = int(len(df) * p)
    train_df = df[:idx]
    eval_df = df[idx:]

    if autosave:
        train_df.to_csv(path_or_buf=save_path_train, index=False)
        eval_df.to_csv(path_or_buf=save_path_eval, index=False)

    return train_df, eval_df


def cnn_size(
    input: Tuple[int, int],
    kernel: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """
    Return the size of the output of a convolutional layer.
    Args:
        input: Size of the input image.
        kernel: Kernel size, it is assumed to be a square.
        padding: Padding size.
        stride: Stride.
    Returns:
        The output size.
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
    Args:
        input: Size of the input image.
        kernel: Kernel size, it is assumed to be a square.
        padding: Padding size.
        stride: Stride.
        n_convs: Number of convolutions.
    Returns:
        The output size.
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
    Args:
        input: Size of the input image.
        padding: Pooling size.
        stride: Stride.
    Returns:
        The output size.
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
    Args:
        input: Size of the input image.
        pooling: Pooling size.
    Returns:
        The output size.
    """
    if isinstance(pooling, int):
        pooling = (pooling, pooling)

    out_w = input[0] / pooling[0]
    out_h = input[1] / pooling[1]
    return int(out_w), int(out_h)


def fc_params(in_features: int, out_features: int, bias: bool = True):
    """
    Return the number of parameters in a linear layer.
    Args:
        in_features: Size of input vector.
        out_features: Size of output vector.
        bias: If true count bias too.
    Returns:
        The number of parameters.
    """
    m = out_features + 1 if bias else out_features
    return in_features * m


def cnn_params(kernel: int, in_channels: int, out_channels: int, bias: bool = True):
    """
    Return the number of parameters in a CNN.
    Args:
        kernel: Kernel size, it is assumed to be squared.
        in_channels: Number of input channels.
        out_channels: Number of output channels i.e. number of kernels.
        bias: If true count bias as well.
    Returns:
        The number of parameters.
    """
    w = kernel * kernel * in_channels * out_channels
    b = out_channels if bias else 0
    return w + b


def cnnxfc_params(
    image_size: Tuple[int, int], n_channels: int, out_features: int, bias: bool = True
):
    """
    Return the number of parameters in a CNN followe by a linear layer.
    Args:
        image_size: Size of the output of the CNN.
        n_channels: Number of the image's channels.
        out_features: Neurons in the linear layer.
        bias: If true count bias.
    Returns:
        Number of parameters.
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
    Args:
        input: Size of the input image.
        output: Size of the feature map.
        padding: Padding size.
        stride: Stride.
    Returns:
        The kernel size.
    """
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride)

    kernel_w = input[0] + 2 * padding[0] - stride[0] * (output[0] - 1)
    kernel_h = input[1] + 2 * padding[1] - stride[1] * (output[1] - 1)
    return kernel_w, kernel_h
