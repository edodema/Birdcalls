from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torchaudio
import pandas as pd
from src.common.utils import (
    TRAIN_BIRDCALLS,
    TRAIN_SOUNDSCAPES,
    BIRD2IDX,
    get_spectrogram,
    load_vocab,
)


class SoundscapeDataset(Dataset):
    def __init__(self, csv_path: str, **kwargs):
        """
        :param csv_path: Path of the training CSV file.
        :param kwargs:
        """
        super(SoundscapeDataset, self).__init__()
        df = pd.read_csv(csv_path)

        # This is needed here due to the birds column having different type values in preprocessed CSV.
        if "train" not in df.columns:
            self.row_id = df["row_id"].values.tolist()
            self.site = df["site"].values.tolist()
            self.audio_id = df["audio_id"].values.astype(np.str).tolist()
            self.seconds = df["seconds"].values

            # Having a call corresponds to 1, otherwise it is 0.
            birds = df["birds"].values
            mask = birds == "nocall"
            self.birds = np.ones(shape=birds.shape, dtype=np.int)
            self.birds[mask] = 0
        else:
            # If there is a train column then the CSV is a train/eval split thus has already been preprocessed.
            self.row_id = df["row_id"].values.tolist()
            self.site = df["site"].values.tolist()
            self.audio_id = df["audio_id"].values.astype(np.str).tolist()
            self.seconds = df["seconds"].values
            self.birds = df["birds"].values

    def __len__(self):
        """
        :return: Length of the dataset.
        """
        return len(self.seconds)

    def __getitem__(self, item):
        """
        :param item: Index of the item to retrieve.
        :return: The item-th entry.
        """
        return {
            "row_id": self.row_id[item],
            "site": self.site[item],
            "audio_id": self.audio_id[item],
            "seconds": self.seconds[item],
            "birds": self.birds[item],
        }

    @staticmethod
    def collate_fn(data):
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


class BirdcallDataset(Dataset):
    # Static attributes
    bird2idx = load_vocab(BIRD2IDX)
    n_classes = len(bird2idx)
    # The fixed length of an audio file in seconds.
    standard_len = 3.5 * 60

    def __init__(self, csv_path: str, **kwargs):
        """
        :param csv_path: Path of the training CSV file.
        :param kwargs:
        """
        df = pd.read_csv(csv_path)
        self.primary_label = df["primary_label"]
        self.scientific_name = df["scientific_name"]
        self.common_name = df["common_name"]
        self.filename = df["filename"]
        self.rating = df["rating"]

    def __len__(self) -> int:
        """
        :return: Length of the dataset.
        """
        return len(self.rating)

    def __getitem__(self, item) -> Dict:
        """
        :param item: Index of the item to retrieve.
        :return: The item-th entry.
        """
        return {
            "primary_label": self.primary_label[item],
            "scientific_name": self.scientific_name[item],
            "common_name": self.common_name[item],
            "filename": self.filename[item],
            "rating": self.rating[item],
        }

    @staticmethod
    def collate(data):
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
            waveform = BirdcallDataset.waveform_resize(
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

    @staticmethod
    def collate_weighted(data):
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
            rating = obj["rating"]

            # Compute the spectrogram.
            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))

            # Directly load the audio file since we need to do some light preprocessing on the waveform.
            raw_waveform, sample_rate = torchaudio.load(audio_file)
            waveform = BirdcallDataset.waveform_resize(
                waveform=raw_waveform, sample_rate=sample_rate
            )

            # Get the spectrograms.
            spec = get_spectrogram(audio=(waveform, sample_rate))
            spectrograms.append(spec * rating)

            # List of targets that will build up one-hot vector for classification.
            target = BirdcallDataset.bird2idx[primary_label]
            targets.append(target)
        return {
            "targets": torch.tensor(targets),
            "spectrograms": torch.stack(spectrograms),
        }

    @staticmethod
    def collate_fn(weighting: bool):
        """
        Wrapper for collate functions.
        :param weighting: If true we weight each spectrogram for its rating.
        :return: A collate function.
        """
        if weighting:
            return BirdcallDataset.collate_weighted
        else:
            return BirdcallDataset.collate

    @staticmethod
    def waveform_resize(waveform: torch.Tensor, sample_rate: int):
        """
        Pad the waveform to a standard size one, if it is too long then gets truncated while if it is
        too short gets padded in reflection mode.
        :param waveform: A waveform.
        :param sample_rate: Its sample rate.
        :return: A waveform of standardized size.
        """
        # Get length of the waveform and length of the standard waveform.
        waveform_len = waveform.shape[1]
        standard_len = int(BirdcallDataset.standard_len * sample_rate)

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
