from typing import Optional, Tuple, List, Union

import hydra.utils
import omegaconf
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from src.common.utils import PROJECT_ROOT
from pathlib import Path


class BirdcallDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.birds, self.calls = None, None

    def init(
        self,
        path: str,
        n_fft: int,
        win_length: Optional[int],
        hop_length: int,
        n_mels: int,
        **kwargs
    ):
        """
        Initialize the dataset.
        :param path: Path of the directory containing data.
        :param n_fft:
        :param win_length:
        :param hop_length:
        :param n_mels:
        :param kwargs:
        :return:
        """
        path = Path(path)
        # We have one directory for each bird.
        dirs = [d for d in path.iterdir() if d.is_dir()]

        birds, calls = [], []
        for dir in dirs:
            for call in dir.iterdir():
                waveform, sample_rate = torchaudio.load(call, format="ogg")

                # Mel Spectrogram transform.
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

                # MelSpectrogram.
                melspec = mel_spectrogram(waveform)

                birds.append(dir.name)
                calls.append(melspec)
        self.birds = birds
        self.calls = calls

    def __len__(self):
        return len(self.birds)

    def __getitem__(self, item):
        return {"bird": self.birds[item], "call": self.calls[item]}

    def save(self, bird_path: Union[str, Path], call_path: Union[str, Path]):
        """
        Save a dataset.
        :param bird_path: Path of the file that will keep bird names.
        :param call_path: Path of the file that will keep tensor bird calls.
        :return:
        """
        torch.save(obj=self.birds, f=bird_path)
        torch.save(obj=self.calls, f=call_path)

    def load(self, bird_path: Union[str, Path], call_path: Union[str, Path]):
        """
        Load a dataset.
        :param bird_path: Path of the file keeping bird names.
        :param call_path: Path of the file keeping tensor bird calls.
        :return:
        """
        self.birds = torch.load(f=bird_path)
        self.calls = torch.load(f=call_path)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    birdcalls_dataset: BirdcallDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train.birdcalls, _recursive_=False
    )

    mode = cfg.data.datamodule.datasets.train.birdcalls.mode
    if mode == "init":
        # Init data.
        hydra.utils.call(
            cfg.data.datamodule.datasets.train.birdcalls.init, self=birdcalls_dataset
        )

        # Save data.
        hydra.utils.call(
            cfg.data.datamodule.datasets.train.birdcalls.save, self=birdcalls_dataset
        )
    elif mode == "load":
        hydra.utils.call(
            cfg.data.datamodule.datasets.train.birdcalls.load, self=birdcalls_dataset
        )


if __name__ == "__main__":
    main()
