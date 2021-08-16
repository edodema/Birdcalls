from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
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
        self.row_id = df["row_id"].values.tolist()
        self.site = df["site"].values.tolist()
        self.audio_id = df["audio_id"].values.astype(np.str).tolist()
        self.seconds = df["seconds"].values

        # Having a call corresponds to 1, otherwise it is 0.
        birds = df["birds"].values
        mask = birds == "nocall"
        self.birds = np.ones(shape=birds.shape, dtype=np.int)
        self.birds[mask] = 0

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

        return {"spectrograms": spectrograms, "targets": golds}


class BirdcallDataset(Dataset):
    # Static attributes
    bird2idx = load_vocab(BIRD2IDX)
    n_classes = len(bird2idx)

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
            spec = get_spectrogram(audio_file)
            spectrograms.append(spec)

            # One hot vector for classification.
            target = torch.zeros(BirdcallDataset.n_classes, dtype=torch.long)
            target[BirdcallDataset.bird2idx[primary_label]] = 1
            targets.append(target)

        return {"targets": targets, "spectrograms": spectrograms}

    @staticmethod
    def collate_weighted(data):
        """
        DataLoader's collate function, it computes the spectrogram on the fly weighting a spectrogram according to the
         recording's quality.
        :param data: Input data
        :return: A batch.
        """
        targets = []
        spectrograms = []

        for obj in data:
            primary_label = obj["primary_label"]
            filename = obj["filename"]
            rating = obj["rating"]

            # Compute spectrograms.
            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))
            spec = get_spectrogram(audio_file)
            spectrograms.append(spec * rating)

            # One hot vector for classification.
            target = torch.zeros(BirdcallDataset.n_classes, dtype=torch.long)
            target[BirdcallDataset.bird2idx[primary_label]] = 1
            targets.append(target)

        return {"targets": targets, "spectrograms": spectrograms}

    @staticmethod
    def collate_fn(weighting: bool = False):
        """
        Wrapper for collate functions.
        :param weighting: If true we weight each spectrogram for its rating.
        :return: A collate function.
        """
        if weighting:
            return BirdcallDataset.collate_weighted
        else:
            return BirdcallDataset.collate
