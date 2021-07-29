from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import hydra
import omegaconf
from src.common.utils import (
    PROJECT_ROOT,
    TRAIN_BIRDCALLS,
    TRAIN_SOUNDSCAPES,
    BIRD2IDX,
    get_spectrogram,
    load_vocab,
)


class SoundscapeDataset(Dataset):
    def __init__(self, csv_path: str, **kwargs):
        super().__init__()
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
        return len(self.seconds)

    def __getitem__(self, item):
        return {
            "row_id": self.row_id[item],
            "site": self.site[item],
            "audio_id": self.audio_id[item],
            "seconds": self.seconds[item],
            "birds": self.birds[item],
        }

    @staticmethod
    def collate_fn_on(data):
        """
        Dataloader's collate function, it computes the spectrogram on the fly.
        :param data: Input data.
        :return:
        """
        golds = []
        spectrograms = []
        for obj in data:
            audio_id = obj["audio_id"]
            site = obj["site"]
            seconds = obj["seconds"]
            row_id = obj["row_id"]
            birds = obj["birds"]
            golds.append(birds)

            file_path = list(TRAIN_SOUNDSCAPES.glob(audio_id + "*"))[0]

            spec = get_spectrogram(file_path, time_window=(seconds - 5, seconds))
            spectrograms.append(spec)

        return {"spectrograms": spectrograms, "targets": golds}


class BirdcallDataset(Dataset):
    bird2idx = load_vocab(BIRD2IDX)
    n_classes = len(bird2idx)

    def __init__(self, csv_path: str, **kwargs):
        df = pd.read_csv(csv_path)
        self.primary_label = df["primary_label"]
        self.scientific_name = df["scientific_name"]
        self.common_name = df["common_name"]
        self.filename = df["filename"]
        self.rating = df["rating"]

    def __len__(self) -> int:
        return len(self.rating)

    def __getitem__(self, item) -> Dict:
        return {
            "primary_label": self.primary_label[item],
            "scientific_name": self.scientific_name[item],
            "common_name": self.common_name[item],
            "filename": self.filename[item],
            "rating": self.rating[item],
        }

    @staticmethod
    def collate_fn_on(data):
        """
        Dataloader's collate function, it computes the spectrogram on the fly.
        :param data: Input data
        :return:
        """
        targets = []
        spectrograms = []

        for obj in data:
            primary_label = obj["primary_label"]
            filename = obj["filename"]
            rating = obj["rating"]

            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))
            spec = get_spectrogram(audio_file)
            spectrograms.append(spec)

            target = torch.zeros(BirdcallDataset.n_classes, dtype=torch.long)
            target[BirdcallDataset.bird2idx[primary_label]] = 1
            targets.append(target)

        return {"targets": targets, "spectrograms": spectrograms}

    @staticmethod
    def collate_fn_weighted_on(data):
        """
        Dataloader's collate function, it computes the spectrogram on the fly.
        :param data: Input data
        :param weighting: If true we weight each recording for its rating.
        :return:
        """
        targets = []
        spectrograms = []

        for obj in data:
            primary_label = obj["primary_label"]
            filename = obj["filename"]
            rating = obj["rating"]

            audio_file = Path(str(TRAIN_BIRDCALLS / primary_label / filename))
            spec = get_spectrogram(audio_file)

            # To weight each spectrogram for its rating uncomment that line and comment the one before.
            spectrograms.append(spec * rating)

            target = torch.zeros(BirdcallDataset.n_classes, dtype=torch.long)
            target[BirdcallDataset.bird2idx[primary_label]] = 1
            targets.append(target)

        return {"targets": targets, "spectrograms": spectrograms}

    @staticmethod
    def collate_fn(weighting: bool = False, online: bool = False):
        """
        Wrapper returning a collate function.
        :param weighting: If true we weight each spectrogram for its rating.
        :param online: If true we compute spectrograms on the fly.
        :return:
        """
        if weighting:
            if online:
                return BirdcallDataset.collate_fn_weighted_on
            else:
                # return BirdcallDataset.collate_fn_weighted_off
                pass
        else:
            if online:
                return BirdcallDataset.collate_fn_on
            else:
                # return BirdcallDataset.collate_fn_off
                pass


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    birdcalls_ds = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train.birdcalls, _recursive_=False
    )
    birdcalls_dl = DataLoader(
        dataset=birdcalls_ds,
        batch_size=4,
        collate_fn=BirdcallDataset.collate_fn_on,
        shuffle=True,
    )

    soundscapes_ds = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train.soundscapes, _recursive_=False
    )

    soundscapes_dl = DataLoader(
        dataset=soundscapes_ds,
        batch_size=20,
        collate_fn=SoundscapeDataset.collate_fn_on,
        shuffle=True,
    )

    # print("Soundscapes")
    # for xb in soundscapes_dl:
    #     targets = xb["targets"]
    #     specs = xb["spectrograms"]
    #     # print(targets)
    #     # print(specs)
    #     for spec, target in zip(specs, targets):
    #         call = "nocall" if target == 0 else "Call"
    #         plot_spectrogram(spec[0], title=call)
    #     break

    print("Birdcalls")
    for xb in birdcalls_dl:
        targets = xb["targets"]
        specs = xb["spectrograms"]
        print(targets)
        print(specs)
        # for spec in specs:
        #     plot_spectrogram(spec[0], title="Birdcall")

        break


if __name__ == "__main__":
    main()
