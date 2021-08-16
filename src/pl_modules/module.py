from typing import Dict, Any

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
import src.pl_modules.model as model
from src.common.utils import PROJECT_ROOT


class Detection(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Detection, self).__init__()


class Classification(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Classification, self).__init__()
        self.save_hyperparameters()

        self.model = model.Classification()

        self.loss = nn.CrossEntropyLoss

        metric = Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, xb):
        logits = self.classification(xb)
        preds = torch.argmax(logits, dim=-1)
        return logits, preds

    def step(self, x: torch.Tensor, y: torch.Tensor):
        logits, preds = self(x)
        loss = self.loss(logits, y)
        return {"logits": logits, "preds": preds, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.train_accuracy(out_step["preds"], targets)
        self.log_dict(
            {"train_loss": out_step["loss"], "train_acc": self.train_accuracy.compute()}
        )
        return out_step["loss"]

    # def validation_step(self, *args, **kwargs):
    #     pass
    #
    # def test_step(self, *args, **kwargs):
    #     pass

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )

        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, optimizer=opt
            )
            return [opt, scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    classification = hydra.utils.instantiate(
        cfg.model.classification, _recursive_=False
    )


if __name__ == "__main__":
    main()
