from typing import Any
import wandb
import hydra
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from src.pl_modules.detection import SoundscapeModel
from src.pl_modules.classification import BirdcallModel
from src.pl_modules.joint import JointModel
from src.common.utils import load_vocab, BIRD2IDX


class SoundscapeDetection(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SoundscapeDetection, self).__init__()
        self.save_hyperparameters()

        self.model = SoundscapeModel()

        self.loss = nn.BCEWithLogitsLoss()

        accuracy = torchmetrics.Accuracy()
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

        precision = torchmetrics.Precision()
        self.train_precision = precision.clone()
        self.val_precision = precision.clone()
        self.test_precision = precision.clone()

        recall = torchmetrics.Recall()
        self.train_recall = recall.clone()
        self.val_recall = recall.clone()
        self.test_recall = recall.clone()

        self.conf_mat = torchmetrics.ConfusionMatrix(num_classes=2)

    def forward(self, xb):
        logits = self.model(xb)
        # The loss function does implement the sigmoid by itself.
        preds = torch.sigmoid(logits).squeeze(1).ge(0.5).to(torch.long)
        return logits, preds

    def step(self, x: torch.Tensor, y: torch.Tensor):
        logits, preds = self(x)
        loss = self.loss(logits, y)
        return {"logits": logits, "preds": preds, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        targets = batch["targets"]
        specs = batch["spectrograms"]

        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets.squeeze(1).to(torch.long)

        self.train_accuracy(x, y)
        self.train_precision(x, y)
        self.train_recall(x, y)

        self.log_dict(
            {
                "train_loss": out_step["loss"],
                "train_acc": self.train_accuracy.compute(),
                "train_prec": self.train_precision.compute(),
                "train_rec": self.train_recall.compute(),
            }
        )
        return out_step["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets.squeeze(1).to(torch.long)

        self.val_accuracy(x, y)
        self.val_precision(x, y)
        self.val_recall(x, y)

        self.log_dict(
            {
                "val_loss": out_step["loss"],
                "val_acc": self.val_accuracy.compute(),
                "val_prec": self.val_precision.compute(),
                "val_rec": self.val_recall.compute(),
            }
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets.squeeze(1).to(torch.long)

        self.test_accuracy(x, y)
        self.test_precision(x, y)
        self.test_recall(x, y)

        self.log_dict(
            {
                "test_acc": self.test_accuracy.compute(),
                "test_prec": self.test_precision.compute(),
                "test_rec": self.test_recall.compute(),
            }
        )

        self.logger.experiment.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=x.cpu(),
                    y_true=y.cpu(),
                    class_names=["nocall", "call"],
                )
            }
        )

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

        self.model = BirdcallModel(out_features=out_features)

        self.loss = nn.CrossEntropyLoss()

        accuracy = torchmetrics.Accuracy()
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

        precision = torchmetrics.Precision()
        self.train_precision = precision.clone()
        self.val_precision = precision.clone()
        self.test_precision = precision.clone()

        recall = torchmetrics.Recall()
        self.train_recall = recall.clone()
        self.val_recall = recall.clone()
        self.test_recall = recall.clone()

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

        x = out_step["preds"]
        y = targets

        self.train_accuracy(x, y)
        self.train_precision(x, y)
        self.train_recall(x, y)

        self.log_dict(
            {
                "train_loss": out_step["loss"],
                "train_acc": self.train_accuracy.compute(),
                "train_prec": self.train_precision.compute(),
                "train_rec": self.train_recall.compute(),
            }
        )
        return out_step["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets

        self.val_accuracy(x, y)
        self.val_precision(x, y)
        self.val_recall(x, y)

        self.log_dict(
            {
                "val_loss": out_step["loss"],
                "val_acc": self.val_accuracy.compute(),
                "val_prec": self.val_precision.compute(),
                "val_rec": self.val_recall.compute(),
            }
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets

        self.test_accuracy(x, y)
        self.test_precision(x, y)
        self.test_recall(x, y)

        self.log_dict(
            {
                "test_acc": self.test_accuracy.compute(),
                "test_prec": self.test_precision.compute(),
                "test_rec": self.test_recall.compute(),
            }
        )

        # Get the list of classes.
        ordered = sorted(load_vocab(BIRD2IDX).items(), key=lambda item: int(item[1]))
        classes = [c for c, _ in ordered]

        self.logger.experiment.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=x.cpu(),
                    y_true=y.cpu(),
                    class_names=classes,
                )
            }
        )

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
        self.model = JointModel(out_features=out_features)

        self.loss = nn.CrossEntropyLoss()

        accuracy = torchmetrics.Accuracy()
        self.train_accuracy = accuracy.clone()
        self.val_accuracy = accuracy.clone()
        self.test_accuracy = accuracy.clone()

        precision = torchmetrics.Precision()
        self.train_precision = precision.clone()
        self.val_precision = precision.clone()
        self.test_precision = precision.clone()

        recall = torchmetrics.Recall()
        self.train_recall = recall.clone()
        self.val_recall = recall.clone()
        self.test_recall = recall.clone()

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

        x = out_step["preds"]
        y = targets

        self.train_accuracy(x, y)
        self.train_precision(x, y)
        self.train_recall(x, y)

        self.log_dict(
            {
                "train_loss": out_step["loss"],
                "train_acc": self.train_accuracy.compute(),
                "train_prec": self.train_precision.compute(),
                "train_rec": self.train_recall.compute(),
            }
        )
        return out_step["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets

        self.val_accuracy(x, y)
        self.val_precision(x, y)
        self.val_recall(x, y)

        self.log_dict(
            {
                "val_loss": out_step["loss"],
                "val_acc": self.val_accuracy.compute(),
                "val_prec": self.val_precision.compute(),
                "val_rec": self.val_recall.compute(),
            }
        )
        return out_step["loss"]

    def test_step(self, batch: Any, batch_idx: int):
        targets = batch["targets"]
        specs = batch["spectrograms"]
        out_step = self.step(x=specs, y=targets)

        x = out_step["preds"]
        y = targets

        self.test_accuracy(x, y)
        self.test_precision(x, y)
        self.test_recall(x, y)

        self.log_dict(
            {
                "test_acc": self.test_accuracy.compute(),
                "test_prec": self.test_precision.compute(),
                "test_rec": self.test_recall.compute(),
            }
        )

        # Get the list of classes.
        ordered = sorted(load_vocab(BIRD2IDX).items(), key=lambda item: int(item[1]))
        classes = [c for c, _ in ordered] + ["nocall"]

        self.logger.experiment.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=x.cpu(),
                    y_true=y.cpu(),
                    class_names=classes,
                )
            }
        )

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
