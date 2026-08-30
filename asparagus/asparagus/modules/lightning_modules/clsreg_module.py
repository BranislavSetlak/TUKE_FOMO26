import logging
import os
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import wandb
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.paths.write import save_json
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchvision import transforms


class ClsRegBase(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        log_image_every_n_epochs: int = 50,
        test_output_path: str = None,
        load_decoder: bool = True,
        repeat_stem_weights: bool = True,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            decoder_warmup_epochs=decoder_warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            load_decoder=load_decoder,
            repeat_stem_weights=repeat_stem_weights,
        )
        self.loss = None
        self.task_type = None
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.test_output_path = test_output_path
        self.ignore_index_in_metrics = -1
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_test_metrics()

    @abstractmethod
    def configure_test_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def configure_metrics(self, prefix: str):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.train_metrics.update(pred, y)
        self._maybe_log_images(batch_idx, x, y, pred, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["CLSREG_label"]
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        self.val_metrics.update(pred, y)
        self._maybe_log_images(batch_idx, x, y, pred, "val")

    def _maybe_log_images(self, batch_idx, x, y, pred, split):
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().float().numpy(),
                    "target": y.detach().cpu().float().numpy(),
                    "output": pred.detach().cpu().float().numpy(),
                },
                log_key=split,
                task_type=self.task_type,
            )

    def on_train_epoch_end(self):
        results = format_multilabel_metrics(
            self.train_metrics.compute(), ignore_index=self.ignore_index_in_metrics
        )
        self.log_dict(results, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        results = format_multilabel_metrics(
            self.val_metrics.compute(), ignore_index=self.ignore_index_in_metrics
        )
        self.log_dict(results, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.results = {}
        self.predictions = []
        self.labels = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.model(batch["image"])

    def predict_step(self, batch, batch_idx):
        return self.model(batch["image"])

    def on_test_epoch_end(self):
        predictions = torch.cat([value.reshape(-1) for value in self.predictions])
        labels = torch.cat([value.reshape(-1) for value in self.labels])
        averages = self.test_metrics(predictions, labels)
        averages = {key: value.cpu().numpy().tolist() for key, value in averages.items()}
        self.results["metrics"] = averages
        os.makedirs(os.path.dirname(self.test_output_path), exist_ok=True)
        save_json(self.results, self.test_output_path)
        logging.info(f"Aggregated test results for {len(self.predictions)} files: {averages}")


class ClassificationModule(ClsRegBase):
    def __init__(self, label_smoothing: float = 0.0, loss_weight: list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(loss_weight) if loss_weight else None,
            label_smoothing=label_smoothing,
        )
        self.task_type = "classification"

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "Precision": MulticlassPrecision(num_classes=self.num_classes, average=None),
                "Recall": MulticlassRecall(num_classes=self.num_classes, average=None),
            }
        )

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/acc": MulticlassAccuracy(num_classes=self.num_classes, average=None),
                f"{prefix}/auroc": MulticlassAUROC(num_classes=self.num_classes, average=None),
                f"{prefix}/acc_macro": MulticlassAccuracy(num_classes=self.num_classes, average="macro"),
                f"{prefix}/auroc_macro": MulticlassAUROC(num_classes=self.num_classes, average="macro"),
            }
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.predicting:
            batch["CLSREG_label"] = batch["CLSREG_label"].view(-1).long()
        return batch

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(1).long()
        label = batch["CLSREG_label"].reshape(-1).long()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
            "probabilities": probabilities[0].detach().cpu().tolist(),
        }
        self.predictions.append(prediction.detach())
        self.labels.append(label.detach())


class RegressionModule(ClsRegBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss()
        self.task_type = "regression"

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/MSE": MeanSquaredError(num_outputs=self.num_classes),
                f"{prefix}/MAE": MeanAbsoluteError(num_outputs=self.num_classes),
            }
        )

    def configure_test_metrics(self):
        return MetricCollection(
            {
                "MSE": MeanSquaredError(num_outputs=self.num_classes),
                "MAE": MeanAbsoluteError(num_outputs=self.num_classes),
            }
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.predicting:
            # Prevent MSELoss from broadcasting [B,1] predictions against [B]
            # labels into a B-by-B loss matrix.
            batch["CLSREG_label"] = batch["CLSREG_label"].view(-1, self.num_classes).float()
        return batch

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prediction = outputs.reshape(-1).float()
        label = batch["CLSREG_label"].reshape(-1).float()
        self.results[batch["file_path"]] = {
            "prediction": prediction.item(),
            "label": label.item(),
        }
        self.predictions.append(prediction.detach())
        self.labels.append(label.detach())
