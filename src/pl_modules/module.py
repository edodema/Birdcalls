from typing import Any

import hydra
import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
import src.pl_modules.model as model


class SoundscapeDetection(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SoundscapeDetection, self).__init__()
        self.save_hyperparameters()

        self.model = model.Detection()

        self.loss = nn.BCEWithLogitsLoss()

        metric = Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, xb):
        logits = self.model(xb)
        # The loss function does implement the sigmoid by itself.
        preds = torch.sigmoid(logits).squeeze().ge(0.5).to(torch.long)
        return logits, preds

    def step(self, x: torch.Tensor, y: torch.Tensor):
        logits, preds = self(x)
        loss = self.loss(logits, y)
        return {"logits": logits, "preds": preds, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        targets = batch["targets"]
        specs = batch["spectrograms"]

        out_step = self.step(x=specs, y=targets)

        self.train_accuracy(out_step["preds"], targets.squeeze().to(torch.long))
        self.log_dict(
            {"train_loss": out_step["loss"], "train_acc": self.train_accuracy.compute()}
        )
        return out_step["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.val_accuracy(out_step["preds"], targets.squeeze().to(torch.long))
        self.log_dict(
            {"val_loss": out_step["loss"], "val_acc": self.val_accuracy.compute()}
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.test_accuracy(out_step["preds"], targets.squeeze().to(torch.long))
        self.log_dict(
            {"test_loss": out_step["loss"], "test_acc": self.test_accuracy.compute()}
        )
        return out_step["loss"]

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.soundscapes.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )

        if not self.hparams.optim.soundscapes.use_lr_scheduler:
            return {"optimizer": opt}
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.soundscapes.lr_scheduler, optimizer=opt
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}


class BirdcallClassification(pl.LightningModule):
    def __init__(self, out_features: int, **kwargs):
        super(BirdcallClassification, self).__init__()
        self.save_hyperparameters()

        self.model = model.Classification(out_features=out_features)

        self.loss = nn.CrossEntropyLoss()

        metric = Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, xb):
        logits = self.model(xb)
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

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.val_accuracy(out_step["preds"], targets)
        self.log_dict(
            {"val_loss": out_step["loss"], "val_acc": self.val_accuracy.compute()}
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.test_accuracy(out_step["preds"], targets)
        self.log_dict(
            {"test_loss": out_step["loss"], "test_acc": self.test_accuracy.compute()}
        )
        return out_step["loss"]

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.birdcalls.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )

        if not self.hparams.optim.birdcalls.use_lr_scheduler:
            return {"optimizer": opt}
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.birdcalls.lr_scheduler, optimizer=opt
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}


class JointClassification(pl.LightningModule):
    def __init__(self, out_features: int, **kwargs):
        super(JointClassification, self).__init__()
        self.save_hyperparameters()
        self.model = model.Classification(out_features=out_features)

        self.loss = nn.CrossEntropyLoss()

        metric = Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, xb):
        logits = self.model(xb)
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

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.val_accuracy(out_step["preds"], targets)
        self.log_dict(
            {"val_loss": out_step["loss"], "val_acc": self.val_accuracy.compute()}
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        self.test_accuracy(out_step["preds"], targets)
        self.log_dict(
            {"test_loss": out_step["loss"], "test_acc": self.test_accuracy.compute()}
        )
        return out_step["loss"]

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.birdcalls.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )

        if not self.hparams.optim.birdcalls.use_lr_scheduler:
            return {"optimizer": opt}
        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.birdcalls.lr_scheduler, optimizer=opt
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}
