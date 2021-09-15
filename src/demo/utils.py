from typing import Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from src.common.utils import PROJECT_ROOT
from src.pl_data.dataset import SoundscapeDataset, BirdcallDataset, JointDataset

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


def get_csv_path(mode: str):
    """
    Just a wrapper to get the correct CSV path.
    Args:
        mode: The mode we are using.

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


def get_tensor(sample: pd.DataFrame, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a tensor out of a DataFrame sample.
    Args:
        sample: The sample obtained by get_sample.
        mode: The mode we are in.

    Returns:
        A spectrogram and a target tensor objects.
    """
    if mode == MODES["soundscapes"] or mode == MODES["split"]:
        _, _, audio_id, seconds, birds = SoundscapeDataset.get_columns(df=sample)
        spectrogram, target = SoundscapeDataset.get_tensor_data(
            audio_id=audio_id, seconds=seconds, birds=birds
        )
        return spectrogram, target

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

        primary_label, _, _, filename, _ = BirdcallDataset.get_columns(df=sample)
        spectrogram, target = birdcalls.get_tensor_data(
            primary_labels=primary_label, filenames=filename
        )
        return spectrogram, target

    if mode == MODES["joint"]:
        _, _, audio_id, seconds, birds = JointDataset.get_columns(df=sample)
        spectrogram, target = JointDataset.get_tensor_data(
            audio_id=audio_id, seconds=seconds, birds=birds
        )
        return spectrogram, target
