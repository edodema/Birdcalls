"""
- Environment setup:
    - get_hydra_cfg

- UI elements:
    - draw_spectrogram
    - translate_detection

- Data manipulation:
    - get_sample
    - get_tensor
    - get_predictions
"""
from typing import Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import hydra
import torch
import librosa
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from src.common.utils import (
    PROJECT_ROOT,
    SOUNDSCAPES_DIR,
    IDX2BIRD,
    load_vocab,
)
from src.pl_data.dataset import JointDataset

idx2bird = load_vocab(IDX2BIRD)

# https://github.com/lucmos/nn-template/blob/main/src/ui/ui_utils.py#L121
def get_hydra_cfg(config_name: str = "default") -> DictConfig:
    """
    Instantiate and return the hydra config -- streamlit and jupyter compatible.
    Args:
        config_name: .yaml configuration name, without the extension

    Returns:
        The desired omegaconf.DictConfig
    """
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"))
    return hydra.compose(config_name=config_name)


cfg = get_hydra_cfg()


def draw_spectrogram(spectrogram):
    """
    Plot a spectrogram tensor in the demo.
    Args:
        spectrogram: The spectrogram tensor.

    Returns:
    """
    spec = np.transpose(librosa.power_to_db(spectrogram[0].numpy()), axes=(1, 2, 0))
    fig, axs = plt.subplots(1, 1)
    axs.set_title("Spectrogram (db)")
    axs.set_ylabel("freq")
    axs.set_xlabel("frame")
    axs.imshow(spec, origin="lower", aspect="auto")
    return fig


def translate_detection(x):
    """
    Just get the detection prediction in a human readable way.
    Args:
        x: What we want to translate.

    Returns:
        The human readable prediction.
    """
    if x == "nocall":
        x = "No"
    else:
        scientific, common = load_vocab(path=cfg.demo.bird_names)[x]
        x = common + " (" + scientific + ")"
    return x


def get_sample(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Function used to get a random sample from a CSV file.
    Args:
        csv_path: Path of the CSV file.

    Returns:
        A random sample from the CSV.
    """
    df = pd.read_csv(csv_path)
    idx = np.random.randint(0, len(df), 1).item()
    # Doing this way we get an array and we can reuse Dataset's functions.
    sample = df.iloc[idx : idx + 1]
    return sample


def get_tensor(
    sample: pd.DataFrame,
) -> Tuple[Union[str, Path], int, torch.Tensor, torch.Tensor]:
    """
    Get a tensor out of a DataFrame sample.
    Args:
        sample: The sample obtained by get_sample.

    Returns:
        - The path of the audio file.
        - Playback audio start time, for Birdcalls 0 by default.
        - The spectrogram tensor.
        - The target class.
    """
    idx2bird["397"] = "nocall"

    row_id, site, audio_id, seconds, birds = JointDataset.get_columns(df=sample)
    spectrogram, target = JointDataset.get_tensor_data(
        audio_id=audio_id, seconds=seconds, birds=birds
    )

    # Get audio path.
    path = list(SOUNDSCAPES_DIR.glob(audio_id[0] + "*"))[0]

    return path, seconds[0] - 5, spectrogram, idx2bird[str(target[0].item())]


def get_prediction(pred):
    """
    Outputs a prediction in a way consistent to the selected mode.
    Args:
        pred: The output prediction of a model.

    Returns:
        The output value, it depends on the mode.
    """
    idx2bird["397"] = "nocall"
    return idx2bird[str(pred.item())]
