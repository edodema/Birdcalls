from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from src.common.utils import (
    TRAIN_BIRDCALLS,
    TRAIN_SOUNDSCAPES,
    BIRD2IDX,
    get_spectrogram,
    load_vocab,
    PROJECT_ROOT,
)
import hydra
from omegaconf import DictConfig


class SoundscapeDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path, None],
        online: bool,
        debug: int,
        load: bool,
        **kwargs
    ):
        """
        :param csv_path: Path of the training CSV file.
        :param online: If true tensors are computed on-the-fly by the dataloader, otherwise they are all precomputed.
        :param debug: Defines the size of the reduced dataset (it is shuffled beforehand) we want to use, any number
        below or equal to 0 means that we keep the whole dataset.
        :param load: If true we do not compute anything and will load values from a file.
        :param kwargs:
        """
        super(SoundscapeDataset, self).__init__()

        self.online = online
        self.len: int

        self.spectrograms: torch.Tensor
        self.targets: torch.Tensor

        # We do something only if load is False.
        if not load:
            df = pd.read_csv(csv_path)

            if debug and debug > 0:
                # Shuffle the dataframe and take the first debug elements.
                df = df.sample(frac=1).iloc[:debug]

            self._init_data(df)

    @staticmethod
    def _get_columns(df: pandas.DataFrame):
        """
        Extract the columns from a dataframe, we can have the original whole CSV's df or a preprocessed split.
        :param df: DataFrame.
        :return: The columns of our CSV as numpy arrays.
        """
        row_id = df["row_id"].values.tolist()
        site = df["site"].values.tolist()
        audio_id = df["audio_id"].values.astype(str).tolist()
        seconds = df["seconds"].values

        # The "birds" column can have different type values in the CSV.
        if "train" not in df.columns:
            # Having a call corresponds to 1, otherwise it is 0.
            birds = df["birds"].values
            mask = birds == "nocall"
            birds = np.ones(shape=birds.shape, dtype=np.int)
            birds[mask] = 0
        else:
            # If there is a train column then the CSV is a train/eval split thus has already been preprocessed.
            birds = df["birds"].values

        return row_id, site, audio_id, seconds, birds

    @staticmethod
    def _get_tensor_data(audio_id, seconds, birds):
        """
        Compute tensors from the data.
        :param audio_id: audio_id column.
        :param seconds: seconds column, defining the second term of the time interval we are dealing with.
        :param birds: birds column, containing if we had a call or not.
        :return: The spectrograms and targets as tensors.
        """
        golds = []
        spectrograms = []
        for id, s, b in zip(audio_id, seconds, birds):
            golds.append(b)

            # Get the soundscape file, * is a wildcard and we only use the id.
            file_path = list(TRAIN_SOUNDSCAPES.glob(id + "*"))[0]
            spec = get_spectrogram(file_path, time_window=(s - 5, s))
            spectrograms.append(spec)

        spectrograms_t = torch.stack(spectrograms)
        golds_t = torch.tensor(golds, dtype=torch.float).unsqueeze(1)
        return spectrograms_t, golds_t

    def _init_data(self, df: pandas.DataFrame):
        """
        Initialize data.
        :param df: Dataframe.
        :return:
        """
        if self.online:
            # On-the-fly dataset
            (
                self.row_id,
                self.site,
                self.audio_id,
                self.seconds,
                self.birds,
            ) = SoundscapeDataset._get_columns(df)

            self.len = len(self.birds)
        else:
            # The online initialization is a base step, we need to do something more.
            row_id, site, audio_id, seconds, birds = SoundscapeDataset._get_columns(df)
            self.spectrograms, self.targets = SoundscapeDataset._get_tensor_data(
                audio_id=audio_id, seconds=seconds, birds=birds
            )

            self.len = len(self.targets)

    def save(
        self,
        dir_path: Union[str, Path],
        spectrograms_name: str = "spectrograms.pt",
        targets_name: str = "targets.pt",
    ):
        """
        Save spectrograms and targets' tensors to a file, this makes sense only for offline datasets.
        :param dir_path: Path of the directory in which the tensors will be saved.
        :param spectrograms_name: Name of the spectrograms' tensor file.
        :param targets_name: Name of the targets' tensor file.
        :return:
        """
        assert (
            not self.online
        ), "The dataset computes tensors on-the-fly, change it to offline."

        torch.save(f=dir_path + "/" + spectrograms_name, obj=self.spectrograms)
        torch.save(f=dir_path + "/" + targets_name, obj=self.targets)

    @staticmethod
    def load(
        spectrograms_path: Union[str, Path], targets_path: Union[str, Path], **kwargs
    ):
        """
        Load a dataset whose spectorgrams and targets are loaded from .pt files.
        :param spectrograms_path: Path of the spectrograms tensor file.
        :param targets_path: Path of the targets tensor file.
        :param kwargs:
        :return: A SoundscapeDataset object with populated tensors.
        """
        ds = SoundscapeDataset(csv_path=None, online=False, debug=-1, load=True)

        ds.spectrograms = torch.load(spectrograms_path)
        ds.targets = torch.load(targets_path)
        ds.len = len(ds.targets)

        return ds

    def __len__(self):
        """
        :return: Length of the dataset.
        """
        return self.len

    def __getitem__(self, item):
        """
        :param item: Index of the item to retrieve.
        :return: The item-th entry.
        """
        if self.online:
            return {
                "row_id": self.row_id[item],
                "site": self.site[item],
                "audio_id": self.audio_id[item],
                "seconds": self.seconds[item],
                "birds": self.birds[item],
            }
        else:
            return {
                "spectrograms": self.spectrograms[item],
                "targets": self.targets[item],
            }

    @staticmethod
    def collate_online(data):
        """
        DataLoader's collate function, it computes the spectrogram on the fly.
        :param data: Input data.
        :return: A batch.
        """
        golds = []
        spectrograms = []

        for obj in data:
            audio_id = obj["audio_id"]
            seconds = obj["seconds"]
            birds = obj["birds"]
            golds.append(birds)

            # We consider 5 seconds audio clips.
            file_path = list(TRAIN_SOUNDSCAPES.glob(audio_id + "*"))[0]
            spec = get_spectrogram(file_path, time_window=(seconds - 5, seconds))
            spectrograms.append(spec)

        return {
            "spectrograms": torch.stack(spectrograms),
            "targets": torch.tensor(golds, dtype=torch.float).unsqueeze(1),
        }

    @staticmethod
    def collate_fn(online: bool):
        """
        A wrapper for the collate function.
        :param online: If true we have an on-the-fly dataset.
        :return: A collate function.
        """
        return SoundscapeDataset.collate_online if online else None


class BirdcallDataset(Dataset):
    # Static attributes
    bird2idx = load_vocab(BIRD2IDX)
    n_classes = len(bird2idx)

    def __init__(
        self,
        csv_path: Union[str, Path, None],
        standard_len: Optional[int],
        online: bool,
        debug: int,
        load: bool,
        **kwargs
    ):
        """
        :param csv_path: Path of the training CSV file.
        :param standard_len: To use tensors we standardize them to a common dimension, that's it.
        :param online: If true tensors are computed on-the-fly by the dataloader, otherwise they are all precomputed.
        :param debug: Defines the size of the reduced dataset (it is shuffled beforehand) we want to use, any number
        below or equal to 0 means that we keep the whole dataset.
        :param load: If true we do not compute anything and will load values from a file.
        :param kwargs:
        """
        self.online = online
        self.len: int

        self.spectrograms: torch.Tensor
        self.targets: torch.Tensor

        # We do something only if load is False.
        if not load:
            # Set the standard length.
            self.standard_len = standard_len * 60

            df = pd.read_csv(csv_path)

            if debug and debug > 0:
                # Shuffle the dataframe and take the first debug elements.
                df = df.sample(frac=1).iloc[:debug]

            self._init_data(df)

    def _init_data(self, df: pandas.DataFrame):
        """
        Initialize data.
        :param df: Dataframe.
        :return:
        """
        if self.online:
            # We do not need to compute anything.
            self.primary_label = df["primary_label"].values
            self.scientific_name = df["scientific_name"].values
            self.common_name = df["common_name"].values
            self.filename = df["filename"].values
            self.rating = df["rating"].values

            self.len = len(self.rating)
        else:
            # We directly compute our tensors.
            primary_labels = df["primary_label"].values
            filenames = df["filename"].values

            self.spectrograms, self.targets = self._get_tensor_data(
                primary_labels=primary_labels, filenames=filenames
            )

            self.len = len(self.targets)

    def _get_tensor_data(self, primary_labels, filenames):
        """
        Compute tensors from the data.
        :param primary_labels: Label of each bird, used to retrieve filepath.
        :param filenames: The name of the audio file, is a relative path inside each primary_label directory.
        :return: The spectrograms and targets as tensors.
        """
        targets = []
        spectrograms = []

        for primary_label, filename in zip(primary_labels, filenames):
            # Get the path for each audio file.
            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))

            # Directly load the audio file since we need to do some light preprocessing on the waveform.
            raw_waveform, sample_rate = torchaudio.load(audio_file)
            waveform = self.waveform_resize(
                waveform=raw_waveform, sample_rate=sample_rate
            )

            # Compute the spectrogram.
            spec = get_spectrogram(audio=(waveform, sample_rate))
            spectrograms.append(spec)

            # List of targets that will build up one-hot vector for classification.
            target = BirdcallDataset.bird2idx[primary_label]
            targets.append(target)

        spectrograms_t = torch.stack(spectrograms)
        targets_t = torch.tensor(targets)

        return spectrograms_t, targets_t

    def waveform_resize(self, waveform: torch.Tensor, sample_rate: int):
        """
        Pad the waveform to a standard size one, if it is too long then gets truncated while if it is
        too short gets padded in reflection mode.
        :param waveform: A waveform.
        :param sample_rate: Its sample rate.
        :return: A waveform of standardized size.
        """
        # Get length of the waveform and length of the standard waveform.
        waveform_len = waveform.shape[1]
        standard_len = int(self.standard_len * sample_rate)

        if waveform_len > standard_len:
            # It gets truncated
            standard_waveform = waveform[:, :standard_len]
        elif waveform_len < standard_len:
            # It gets padded using reflection.

            # Get the padding dimension.
            offset = standard_len - waveform_len
            m = offset // 2
            pad = (m, m) if offset % 2 == 0 else (m + 1, m)

            # Forced to use numpy due to torch.nn.functional.pad having some issues
            # with 'reflect', 'replicate' and 'circular' mode.
            standard_waveform = torch.from_numpy(
                np.pad(array=waveform.numpy().squeeze(), pad_width=pad, mode="reflect")
            ).unsqueeze(0)
        else:
            # It is unchanged.
            standard_waveform = waveform

        return standard_waveform

    def collate_online(self, data):
        """
        DataLoader's collate function, it computes the spectrogram on the fly.
        :param data: Input data
        :return: A batch.
        """
        targets = []
        spectrograms = []

        for obj in data:
            primary_label = obj["primary_label"]
            filename = obj["filename"]

            # Compute the spectrogram.
            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))

            # Directly load the audio file since we need to do some light preprocessing on the waveform.
            raw_waveform, sample_rate = torchaudio.load(audio_file)
            waveform = self.waveform_resize(
                waveform=raw_waveform, sample_rate=sample_rate
            )

            # Get the spectrograms.
            spec = get_spectrogram(audio=(waveform, sample_rate))
            spectrograms.append(spec)

            # List of targets that will build up one-hot vector for classification.
            target = BirdcallDataset.bird2idx[primary_label]
            targets.append(target)

        return {
            "targets": torch.tensor(targets),
            "spectrograms": torch.stack(spectrograms),
        }

    def save(
        self,
        dir_path: Union[str, Path],
        spectrograms_name: str = "spectrograms.pt",
        targets_name: str = "targets.pt",
    ):
        """
        Save spectrograms and targets' tensors to a file, this makes sense only for offline datasets.
        :param dir_path: Path of the directory in which the tensors will be saved.
        :param spectrograms_name: Name of the spectrograms' tensor file.
        :param targets_name: Name of the targets' tensor file.
        :return:
        """
        assert (
            not self.online
        ), "The dataset computes tensors on-the-fly, change it to offline."

        torch.save(f=dir_path + "/" + spectrograms_name, obj=self.spectrograms)
        torch.save(f=dir_path + "/" + targets_name, obj=self.targets)

    @staticmethod
    def load(
        spectrograms_path: Union[str, Path], targets_path: Union[str, Path], **kwargs
    ):
        """
        Load a dataset whose spectorgrams and targets are loaded from .pt files.
        :param spectrograms_path: Path of the spectrograms tensor file.
        :param targets_path: Path of the targets tensor file.
        :param kwargs:
        :return: A SoundscapeDataset object with populated tensors.
        """
        ds = BirdcallDataset(
            csv_path=None, standard_len=None, online=False, debug=-1, load=True
        )
        ds.spectrograms = torch.load(spectrograms_path)
        ds.targets = torch.load(targets_path)
        ds.len = len(ds.targets)

        return ds

    def __len__(self) -> int:
        """
        :return: Length of the dataset.
        """
        return self.len

    def __getitem__(self, item) -> Dict:
        """
        :param item: Index of the item to retrieve.
        :return: The item-th entry.
        """
        if self.online:
            return {
                "primary_label": self.primary_label[item],
                "scientific_name": self.scientific_name[item],
                "common_name": self.common_name[item],
                "filename": self.filename[item],
                "rating": self.rating[item],
            }
        else:
            return {
                "spectrograms": self.spectrograms[item],
                "targets": self.targets[item],
            }

    def collate_fn(self, online: bool):
        """
        Wrapper for collate functions.
        :param online: If true we have an on-the-fly dataset.
        :return: A collate function.
        """
        return self.collate_online if online else None


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    # Train
    print("Train")
    spectrograms = cfg.data.soundscapes_datamodule.datasets.train.spectrograms_path
    targets = cfg.data.soundscapes_datamodule.datasets.train.targets_path

    path = Path(
        "/home/edo/Documents/Code/Birdcalls/out/split_datasets/train/soundscapes_balanced.csv"
    )
    out = "/home/edo/Documents/Code/Birdcalls/out/debug_datasets/train/soundscapes"

    ds = SoundscapeDataset(csv_path=path, online=False, debug=-1, load=False)
    print(ds.targets.shape)
    print(ds.spectrograms.shape)

    ds.save(
        dir_path=out,
        spectrograms_name="spectrograms_balanced.pt",
        targets_name="targets_balanced.pt",
    )

    ds = SoundscapeDataset.load(spectrograms_path=spectrograms, targets_path=targets)

    print(ds.targets.shape)
    print(ds.spectrograms.shape)

    # Val
    print("Val")

    spectrograms = cfg.data.soundscapes_datamodule.datasets.val.spectrograms_path
    targets = cfg.data.soundscapes_datamodule.datasets.val.targets_path
    path = Path(
        "/home/edo/Documents/Code/Birdcalls/out/split_datasets/val/soundscapes_balanced.csv"
    )
    out = "/home/edo/Documents/Code/Birdcalls/out/debug_datasets/val/soundscapes"

    ds = SoundscapeDataset(csv_path=path, online=False, debug=-1, load=False)
    print(ds.targets.shape)
    print(ds.spectrograms.shape)

    ds.save(
        dir_path=out,
        spectrograms_name="spectrograms_balanced.pt",
        targets_name="targets_balanced.pt",
    )

    ds = SoundscapeDataset.load(spectrograms_path=spectrograms, targets_path=targets)
    print(ds.targets.shape)
    print(ds.spectrograms.shape)


if __name__ == "__main__":
    main()
