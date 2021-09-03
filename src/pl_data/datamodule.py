from typing import Optional, Dict, List, Union
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from src.pl_data.dataset import BirdcallDataset, SoundscapeDataset


class SoundscapesDataModule(pl.LightningModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        shuffle: DictConfig,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle

        # These attributes will be populated after self.setup() call.
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            # Train
            if self.datasets.train.load:
                # Load tensors if the dataset wants to.
                self.train_ds = SoundscapeDataset.load(
                    spectrograms_path=self.datasets.train.spectrograms_path,
                    targets_path=self.datasets.train.targets_path,
                )
            else:
                self.train_ds = hydra.utils.instantiate(self.datasets.train)

            # Val
            if self.datasets.val.load:
                # Load tensors if the dataset wants to.
                self.val_ds = SoundscapeDataset.load(
                    spectrograms_path=self.datasets.val.spectrograms_path,
                    targets_path=self.datasets.val.targets_path,
                )
            else:
                self.val_ds = hydra.utils.instantiate(self.datasets.val)

        if stage is None or stage == "test":
            if self.datasets.test.load:
                # Load tensors if the dataset wants to.
                self.test_ds = SoundscapeDataset.load(
                    spectrograms_path=self.datasets.test.spectrograms_path,
                    targets_path=self.datasets.test.targets_path,
                )
            else:
                self.test_ds = hydra.utils.instantiate(self.datasets.test)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        batch_size = self.batch_size["train"]
        shuffle = self.shuffle["train"]

        dl = DataLoader(
            dataset=self.train_ds,
            batch_size=batch_size,
            collate_fn=SoundscapeDataset.collate_fn(online=self.train_ds.online),
            shuffle=shuffle,
        )

        return dl

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        batch_size = self.batch_size["val"]
        shuffle = self.shuffle["val"]

        dl = DataLoader(
            dataset=self.val_ds,
            batch_size=batch_size,
            collate_fn=SoundscapeDataset.collate_fn(online=self.val_ds.online),
            shuffle=shuffle,
        )

        return dl

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        batch_size = self.batch_size["test"]
        shuffle = self.shuffle["test"]

        dl = DataLoader(
            dataset=self.test_ds,
            batch_size=batch_size,
            collate_fn=SoundscapeDataset.collate_fn(online=self.test_ds.online),
            shuffle=shuffle,
        )

        return dl

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f" {self.datasets=},\n"
            f" {self.num_workers=},\n"
            f" {self.batch_size=}\n)"
        )


class BirdcallsDataModule(pl.LightningModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        shuffle: DictConfig,
        weighting: DictConfig,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weighting = weighting

        # These attributes will be populated after self.setup() call.
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            # Train
            if self.datasets.train.load:
                # Load tensors if the dataset wants to.
                self.train_ds = BirdcallDataset.load(
                    spectrograms_path=self.datasets.train.spectrograms_path,
                    targets_path=self.datasets.train.targets_path,
                )
            else:
                self.train_ds = hydra.utils.instantiate(self.datasets.train)

            # Val
            if self.datasets.val.load:
                # Load tensors if the dataset wants to.
                self.val_ds = BirdcallDataset.load(
                    spectrograms_path=self.datasets.val.spectrograms_path,
                    targets_path=self.datasets.val.targets_path,
                )
            else:
                self.val_ds = hydra.utils.instantiate(self.datasets.val)

        if stage is None or stage == "test":
            if self.datasets.test.load:
                # Load tensors if the dataset wants to.
                self.test_ds = BirdcallDataset.load(
                    spectrograms_path=self.datasets.test.spectrograms_path,
                    targets_path=self.datasets.test.targets_path,
                )
            else:
                self.test_ds = hydra.utils.instantiate(self.datasets.test)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        batch_size = self.batch_size["train"]
        weighting = self.weighting["train"]
        shuffle = self.shuffle["train"]

        dl = DataLoader(
            dataset=self.train_ds,
            batch_size=batch_size,
            collate_fn=self.train_ds.collate_fn(online=self.train_ds.online),
            shuffle=shuffle,
        )

        return dl

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        batch_size = self.batch_size["val"]
        weighting = self.weighting["val"]
        shuffle = self.shuffle["val"]

        dl = DataLoader(
            dataset=self.val_ds,
            batch_size=batch_size,
            collate_fn=self.val_ds.collate_fn(online=self.val_ds.online),
            shuffle=shuffle,
        )

        return dl

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        batch_size = self.batch_size["test"]
        weighting = self.weighting["test"]
        shuffle = self.shuffle["test"]

        dl = DataLoader(
            dataset=self.test_ds,
            batch_size=batch_size,
            collate_fn=self.test_ds.collate_fn(online=self.test_ds.online),
            shuffle=shuffle,
        )

        return dl

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f" {self.datasets=},\n"
            f" {self.num_workers=},\n"
            f" {self.batch_size=},\n"
            f" {self.weighting=}\n)"
        )
