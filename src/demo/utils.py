"""
- Environment setup:
    - get_hydra_cfg

- UI elements:
    - draw_spectrogram

- Helper functions:
    - get_csv_path
    - get_sample
    - get_tensor
    - translate_detection
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
from src.common.utils import PROJECT_ROOT, SOUNDSCAPES_DIR, IDX2BIRD, load_vocab
from src.pl_data.dataset import SoundscapeDataset, BirdcallDataset, JointDataset

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

MODES = {
    "empty": cfg.demo.mode.empty,
    "soundscapes": cfg.demo.mode.soundscapes,
    "birdcalls": cfg.demo.mode.birdcalls,
    "split": cfg.demo.mode.split,
    "joint": cfg.demo.mode.joint,
}


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


def get_csv_path(mode: str):
    """
    Just a wrapper to get the correct CSV path.
    Args:
        mode: The selected mode.

    Returns:
        The corresponding filepath.
    """
    if mode == MODES["soundscapes"]:
        return cfg.demo.csv.soundscapes
    elif mode == MODES["birdcalls"]:
        return cfg.demo.csv.birdcalls
    elif mode == MODES["split"]:
        return cfg.demo.csv.soundscapes
    elif mode == MODES["joint"]:
        return cfg.demo.csv.joint
    else:
        return None


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
    sample: pd.DataFrame, mode: str
) -> Tuple[Union[str, Path], int, torch.Tensor, torch.Tensor]:
    """
    Get a tensor out of a DataFrame sample.
    Args:
        sample: The sample obtained by get_sample.
        mode: The mode we are in.

    Returns:
        - The path of the audio file.
        - Playback audio start time, for Birdcalls 0 by default.
        - The spectrogram tensor.
        - The target class.
    """
    if mode == MODES["soundscapes"] or mode == MODES["split"]:
        row_id, site, audio_id, seconds, birds = SoundscapeDataset.get_columns(
            df=sample
        )
        spectrogram, target = SoundscapeDataset.get_tensor_data(
            audio_id=audio_id, seconds=seconds, birds=birds
        )

        # Get audio path.
        path = list(SOUNDSCAPES_DIR.glob(audio_id[0] + "*"))[0]

        return path, seconds[0] - 5, spectrogram, target.to(int).item()

    elif mode == MODES["birdcalls"]:
        # We need to instantiate a dummy dataset object due to BirdcallDataset.get_tensor_data not being static.
        # To do that we set both online and load to True, thus the dataset does nothing.
        birdcalls = BirdcallDataset(
            csv_path=None,
            standard_len=cfg.data.birdcalls_datamodule.datasets.val.standard_len,
            online=True,
            debug=-1,
            load=True,
        )

        (
            primary_label,
            scientific_name,
            common_name,
            filename,
            rating,
        ) = BirdcallDataset.get_columns(df=sample)
        spectrogram, target = birdcalls.get_tensor_data(
            primary_labels=primary_label, filenames=filename
        )

        # Get the audio file path.
        path = cfg.demo.audio_dir.birdcalls + "/" + primary_label[0] + "/" + filename[0]

        return path, 0, spectrogram, idx2bird[str(target[0].item())]

    if mode == MODES["joint"]:
        idx2bird["397"] = "nocall"

        row_id, site, audio_id, seconds, birds = JointDataset.get_columns(df=sample)
        spectrogram, target = JointDataset.get_tensor_data(
            audio_id=audio_id, seconds=seconds, birds=birds
        )

        # Get audio path.
        path = list(SOUNDSCAPES_DIR.glob(audio_id[0] + "*"))[0]

        return path, seconds[0] - 5, spectrogram, idx2bird[str(target[0].item())]


def translate_detection(x, mode: str):
    """
    Just get the detection prediction in a human readable way.
    Args:
        x: What we want to translate.
        mode: The selected mode.

    Returns:
        The human readable prediction.
    """
    if mode == MODES["soundscapes"]:
        x = "No" if x == 0 else "Yes"
    return x


def get_prediction(pred, mode: str):
    """
    Outputs a prediction in a way consistent to the selected mode.
    Args:
        pred: The output prediction of a model.
        mode: The selected mode.

    Returns:
        The output value, it depends on the mode.
    """
    if mode == MODES["soundscapes"] or mode == MODES["split"]:
        return translate_detection(pred, mode=mode)

    elif mode == MODES["birdcalls"]:
        return idx2bird[str(pred.item())]

    if mode == MODES["joint"]:
        idx2bird["397"] = "nocall"
        return idx2bird[str(pred.item())]
