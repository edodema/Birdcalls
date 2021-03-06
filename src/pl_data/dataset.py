from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from src.common.utils import (
    BIRDCALLS_DIR,
    SOUNDSCAPES_DIR,
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
    def get_columns(df: pandas.DataFrame):
        """
        Extract the columns from a dataframe, we can have the original whole CSV's df or a preprocessed split.
        :param df: DataFrame.
        :return: The columns of our CSV as numpy arrays.
        """
        row_id = df["row_id"].values.tolist()
        site = df["site"].values.tolist()
        audio_id = df["audio_id"].values.astype(str).tolist()
        seconds = df["seconds"].values

        # Having a call corresponds to 1, otherwise it is 0.
        birds = df["birds"].values
        mask = birds == "nocall"
        birds = np.ones(shape=birds.shape, dtype=np.int)
        birds[mask] = 0

        return row_id, site, audio_id, seconds, birds

    @staticmethod
    def get_tensor_data(audio_id, seconds, birds):
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
            file_path = list(SOUNDSCAPES_DIR.glob(id + "*"))[0]
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
            ) = SoundscapeDataset.get_columns(df)

            self.len = len(self.birds)
        else:
            # The online initialization is a base step, we need to do something more.
            row_id, site, audio_id, seconds, birds = SoundscapeDataset.get_columns(df)
            self.spectrograms, self.targets = SoundscapeDataset.get_tensor_data(
                audio_id=audio_id, seconds=seconds, birds=birds
            )

            self.len = len(self.targets)

    def save(
        self,
        spectrograms_path: str,
        targets_path: str,
    ):
        """
        Save spectrograms and targets' tensors to a file, this makes sense only for offline datasets.
        :param spectrograms_path: Path of the spectrograms' tensor file.
        :param targets_path: Path of the targets' tensor file.
        :return:
        """
        assert (
            not self.online
        ), "The dataset computes tensors on-the-fly, change it to offline."

        torch.save(f=spectrograms_path, obj=self.spectrograms)
        torch.save(f=targets_path, obj=self.targets)

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
            file_path = list(SOUNDSCAPES_DIR.glob(audio_id + "*"))[0]
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
        online: bool,
        debug: int,
        load: bool,
        standard_len: Optional[int] = None,
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
        # If there is no standard len then we are loading.
        assert bool(standard_len) or load

        self.online = online
        self.len: int

        self.spectrograms: torch.Tensor
        self.targets: torch.Tensor

        # Set the standard length, this is useful only for the UI.
        if standard_len:
            self.standard_len = standard_len * 60

        # We do something only if load is False.
        if not load:
            df = pd.read_csv(csv_path)

            if debug and debug > 0:
                # Shuffle the dataframe and take the first debug elements.
                df = df.sample(frac=1).iloc[:debug]

            self._init_data(df)

    @staticmethod
    def get_columns(df: pd.DataFrame):
        """
        Extract the columns from a dataframe, we can have the original whole CSV's df or a preprocessed split.
        :param df: DataFrame.
        :return: The columns of our CSV as numpy arrays.
        """
        primary_label = df["primary_label"].values
        scientific_name = df["scientific_name"].values
        common_name = df["common_name"].values
        filename = df["filename"].values
        rating = df["rating"].values

        return primary_label, scientific_name, common_name, filename, rating

    def _init_data(self, df: pandas.DataFrame):
        """
        Initialize data.
        :param df: Dataframe.
        :return:
        """
        if self.online:
            # We do not need to compute anything.
            (
                self.primary_label,
                self.scientific_name,
                self.common_name,
                self.filename,
                self.rating,
            ) = BirdcallDataset.get_columns(df)

            self.len = len(self.rating)
        else:
            # We directly compute our tensors.
            (
                primary_label,
                scientific_name,
                common_name,
                filename,
                rating,
            ) = BirdcallDataset.get_columns(df)

            self.spectrograms, self.targets = self.get_tensor_data(
                primary_labels=primary_label, filenames=filename
            )

            self.len = len(self.targets)

    def get_tensor_data(self, primary_labels, filenames):
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
            audio_file = Path(str(BIRDCALLS_DIR / primary_label / filename))

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
            audio_file = Path(str(BIRDCALLS_DIR / primary_label / filename))

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
        spectrograms_path: str,
        targets_path: str,
    ):
        """
        Save spectrograms and targets' tensors to a file, this makes sense only for offline datasets.
        :param spectrograms_path: Path of the spectrograms' tensor file.
        :param targets_path: Path of the targets' tensor file.
        :return:
        """
        assert (
            not self.online
        ), "The dataset computes tensors on-the-fly, change it to offline."

        torch.save(f=spectrograms_path, obj=self.spectrograms)
        torch.save(f=targets_path, obj=self.targets)

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


class JointDataset(Dataset):
    # Static attributes
    bird2idx = load_vocab(BIRD2IDX)
    n_classes = len(bird2idx)

    bird2idx["nocall"] = n_classes
    n_classes += 1

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
        super(JointDataset, self).__init__()

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
    def get_columns(df: pandas.DataFrame):
        """
        Extract the columns from a dataframe, we can have the original whole CSV's df or a preprocessed split.
        :param df: DataFrame.
        :return: The columns of our CSV as numpy arrays.
        """
        row_id = df["row_id"].values.tolist()
        site = df["site"].values.tolist()
        audio_id = df["audio_id"].values.astype(str).tolist()
        seconds = df["seconds"].values

        # For our purposes we only care about the primary label.
        def get_primary_label(birds: str):
            """
            Get the first bird of a string of birds, separated by a space.
            :param birds: Array of birds.
            :return: The first bird.
            """
            return birds.split()[0]

        get_primary_label_v = np.vectorize(get_primary_label)
        birds = get_primary_label_v(df["birds"].values)

        # Get the index of each class i.e. birds + nocall.
        birds = np.vectorize(lambda s: JointDataset.bird2idx[s])(birds)

        return row_id, site, audio_id, seconds, birds

    @staticmethod
    def get_tensor_data(audio_id, seconds, birds):
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
            file_path = list(SOUNDSCAPES_DIR.glob(id + "*"))[0]
            spec = get_spectrogram(file_path, time_window=(s - 5, s))
            spectrograms.append(spec)

        spectrograms_t = torch.stack(spectrograms)
        golds_t = torch.tensor(golds)
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
            ) = JointDataset.get_columns(df)

            self.len = len(self.birds)
        else:
            # The online initialization is a base step, we need to do something more.
            row_id, site, audio_id, seconds, birds = JointDataset.get_columns(df)
            self.spectrograms, self.targets = JointDataset.get_tensor_data(
                audio_id=audio_id, seconds=seconds, birds=birds
            )
            self.len = len(self.targets)

    def save(
        self,
        spectrograms_path: str,
        targets_path: str,
    ):
        """
        Save spectrograms and targets' tensors to a file, this makes sense only for offline datasets.
        :param spectrograms_path: Path of the spectrograms' tensor file.
        :param targets_path: Path of the targets' tensor file.
        :return:
        """
        assert (
            not self.online
        ), "The dataset computes tensors on-the-fly, change it to offline."

        torch.save(f=spectrograms_path, obj=self.spectrograms)
        torch.save(f=targets_path, obj=self.targets)

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
        ds = JointDataset(csv_path=None, online=False, debug=-1, load=True)

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
            file_path = list(SOUNDSCAPES_DIR.glob(audio_id + "*"))[0]
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    # print("TRAIN")
    #
    # spectrograms_path = cfg.data.joint_datamodule.datasets.train.spectrograms_path
    # targets_path = cfg.data.joint_datamodule.datasets.train.targets_path
    #
    # print(spectrograms_path)
    # print(targets_path)
    #
    # print("save")
    # ds = hydra.utils.instantiate(
    #     cfg.data.joint_datamodule.datasets.train, _recursive_=False
    # )
    #
    # print(ds.spectrograms.shape)
    # print(ds.targets.shape)
    #
    # ds.save(spectrograms_path=spectrograms_path, targets_path=targets_path)
    #
    # print("load")
    # ds = JointDataset.load(
    #     spectrograms_path=spectrograms_path, targets_path=targets_path
    # )
    #
    # print(ds.spectrograms.shape)
    # print(ds.targets.shape)
    #
    # print("\nVAL")
    #
    # spectrograms_path = cfg.data.joint_datamodule.datasets.val.spectrograms_path
    # targets_path = cfg.data.joint_datamodule.datasets.val.targets_path
    #
    # print(spectrograms_path)
    # print(targets_path)
    #
    # print("save")
    # ds = hydra.utils.instantiate(
    #     cfg.data.joint_datamodule.datasets.val, _recursive_=False
    # )
    #
    # print(ds.spectrograms.shape)
    # print(ds.targets.shape)
    #
    # ds.save(spectrograms_path=spectrograms_path, targets_path=targets_path)
    #
    # print("load")
    # ds = JointDataset.load(
    #     spectrograms_path=spectrograms_path, targets_path=targets_path
    # )
    #
    # print(ds.spectrograms.shape)
    # print(ds.targets.shape)

    print("\nTEST")

    spectrograms_path = cfg.data.birdcalls_datamodule.datasets.test.spectrograms_path
    targets_path = cfg.data.birdcalls_datamodule.datasets.test.targets_path

    print(spectrograms_path)
    print(targets_path)

    print("save")
    ds = hydra.utils.instantiate(
        cfg.data.birdcalls_datamodule.datasets.test, _recursive_=False
    )

    print(ds.spectrograms.shape)
    print(ds.targets.shape)

    ds.save(spectrograms_path=spectrograms_path, targets_path=targets_path)

    print("load")
    ds = BirdcallDataset.load(
        spectrograms_path=spectrograms_path, targets_path=targets_path
    )

    print(ds.spectrograms.shape)
    print(ds.targets.shape)


if __name__ == "__main__":
    main()
