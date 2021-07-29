from typing import Optional, Dict, List, Union
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from src.common.utils import PROJECT_ROOT
from src.pl_data.dataset import BirdcallDataset, SoundscapeDataset


class SplitDataModule(pl.LightningModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        weighting: bool,
        online: bool,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.weighting = weighting
        self.online = online

        self.train_soundscapes_ds: Optional[Dataset] = None
        self.val_soundscapes_ds: Optional[Dataset] = None
        self.test_soundscapes_ds: Optional[Dataset] = None

        self.train_birdcalls_ds: Optional[Dataset] = None
        self.val_birdcalls_ds: Optional[Dataset] = None
        self.test_birdcalls_ds: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # TODO: Do not train on the whole dataset since we have no val set. Fit on smaller sets (on Colab) train on the whole set only in the end.
        # TODO: Function that preprocesses data (See README.md) to make easier loading/splitting it.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            self.train_soundscapes_ds = hydra.utils.instantiate(
                self.datasets.train.soundscapes
            )
            self.train_birdcalls_ds = hydra.utils.instantiate(
                self.datasets.train.birdcalls
            )

            # self.val_soundscapes_ds = hydra.utils.instantiate(
            #     self.datasets.val.soundscapes
            # )
            # self.val_birdcalls_ds = hydra.utils.instantiate(self.datasets.val.birdcalls)

        # if stage is None or stage == "test":
        #     self.test_soundscapes_ds = hydra.utils.instantiate(
        #         self.datasets.test.soundscapes
        #     )
        #     self.test_birdcalls_ds = hydra.utils.instantiate(
        #         self.datasets.test.birdcalls
        #     )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        soundscapes_dl = DataLoader(
            dataset=self.train_soundscapes_ds,
            batch_size=self.batch_size,
            collate_fn=SoundscapeDataset.collate_fn_on,
            shuffle=True,
        )

        birdcalls_dl = DataLoader(
            dataset=self.train_birdcalls_ds,
            batch_size=self.batch_size,
            collate_fn=BirdcallDataset.collate_fn(
                weighting=self.weighting, online=self.online
            ),
            shuffle=True,
        )

        return {
            "train_soundscapes_dataloader": soundscapes_dl,
            "train_birdcalls_dataloader": birdcalls_dl,
        }

    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     soundscapes_dl = DataLoader(
    #         dataset=self.val_soundscapes_ds,
    #         batch_size=self.batch_size,
    #         collate_fn=SoundscapeDataset.collate_fn_on,
    #         shuffle=True,
    #     )
    #
    #     birdcalls_dl = DataLoader(
    #         dataset=self.val_birdcalls_ds,
    #         batch_size=self.batch_size,
    #         collate_fn=BirdcallDataset.collate_fn_on,
    #         shuffle=True,
    #     )
    #
    #     return [soundscapes_dl, birdcalls_dl]
    #
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     soundscapes_dl = DataLoader(
    #         dataset=self.test_soundscapes_ds,
    #         batch_size=self.batch_size,
    #         collate_fn=SoundscapeDataset.collate_fn_on,
    #         shuffle=True,
    #     )
    #
    #     birdcalls_dl = DataLoader(
    #         dataset=self.test_birdcalls_ds,
    #         batch_size=self.batch_size,
    #         collate_fn=BirdcallDataset.collate_fn_on,
    #         shuffle=True,
    #     )
    #
    #     return [soundscapes_dl, birdcalls_dl]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f" {self.datasets=},\n"
            f" {self.num_workers=},\n"
            f" {self.batch_size=}\n)"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    split = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    split.setup()
    print(split)


if __name__ == "__main__":
    main()
