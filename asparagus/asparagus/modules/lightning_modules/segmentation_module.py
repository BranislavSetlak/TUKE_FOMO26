import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torchmetrics.functional
import wandb
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.functional.reverse_preprocessing import reverse_preprocessing
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.metrics import (
    FN,
    FP,
    TP,
    dice,
    f1,
    jaccard,
    precision,
    sensitivity,
    specificity,
    total_pos_gt,
    total_pos_pred,
    volume_similarity,
)
from gardening_tools.functional.paths.write import save_json
from gardening_tools.modules.losses.deep_supervision import DeepSupervisionLoss
from gardening_tools.modules.losses.DiceCE import DiceCE
from gardening_tools.modules.metrics import GeneralizedDiceScore
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score
from torchvision import transforms
from typing import Optional


class SegmentationModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        deep_supervision: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "SGD",
        inference_patch_size: list = [],
        test_output_path: str = None,
        log_image_every_n_epochs: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        load_decoder: bool = True,
        repeat_stem_weights: bool = True,
        sliding_window_validation: bool = False,
        inference_overlap: float = 0.5,
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
        self.inference_patch_size = inference_patch_size
        self.sliding_window_validation = bool(sliding_window_validation)
        self.inference_overlap = float(inference_overlap)
        if not 0.0 <= self.inference_overlap < 1.0:
            raise ValueError("inference_overlap must be in [0, 1)")
        self.test_output_path = test_output_path
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.deep_supervision = deep_supervision

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")

        self.train_loss = DiceCE()
        self.val_loss = DiceCE()

        if self.deep_supervision:
            self.train_loss = DeepSupervisionLoss(loss=self.train_loss, weights=None)

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/dice": GeneralizedDiceScore(
                    num_classes=self.num_classes,
                    weight_type="linear",
                    per_class=True,
                    input_format="index",
                ),
                f"{prefix}/F1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    average=None,
                ),
            },
        )

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.train_loss(pred, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of
            # (downsampled) tensors. We only need the original ground truth and
            # its corresponding prediction which is always the first entry in each list.
            pred = pred[0]
            y = y[0]

        self._log_foreground_behavior("train", pred, y)

        metrics = self.train_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="train",
                task_type="segmentation",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        if self.sliding_window_validation:
            pred = self.model.sliding_window_predict(
                data=x,
                patch_size=self.inference_patch_size,
                overlap=self.inference_overlap,
            )
        else:
            pred = self.model(x)
        loss = self.val_loss(pred, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        self._update_validation_epoch_metrics(pred, y)

        metrics = self.val_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="val",
                task_type="segmentation",
            )

    def _log_foreground_behavior(self, prefix: str, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Log collapse diagnostics alongside the existing per-class Dice."""

        probabilities = torch.softmax(logits.float(), dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        target_index = target.squeeze(1).long()
        predicted_foreground = prediction > 0
        target_foreground = target_index > 0
        intersection = (predicted_foreground & target_foreground).float().sum()
        denominator = predicted_foreground.float().sum() + target_foreground.float().sum()
        foreground_dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
        foreground_class_dice = []
        for class_id in range(1, self.num_classes):
            predicted_class = prediction == class_id
            target_class = target_index == class_id
            class_intersection = (predicted_class & target_class).float().sum()
            class_denominator = predicted_class.float().sum() + target_class.float().sum()
            foreground_class_dice.append(
                (2.0 * class_intersection + 1e-8) / (class_denominator + 1e-8)
            )
        minimum_foreground_class_dice = torch.stack(foreground_class_dice).min()
        foreground_probability = probabilities[:, 1:].sum(dim=1)
        if target_foreground.any():
            target_foreground_probability = foreground_probability[target_foreground].mean()
        else:
            target_foreground_probability = foreground_probability.new_zeros(())

        values = {
            f"{prefix}/foreground_dice": foreground_dice,
            f"{prefix}/min_foreground_class_dice": minimum_foreground_class_dice,
            f"{prefix}/pred_foreground_fraction": predicted_foreground.float().mean(),
            f"{prefix}/target_foreground_fraction": target_foreground.float().mean(),
            f"{prefix}/target_foreground_probability": target_foreground_probability,
            f"{prefix}/max_foreground_probability": foreground_probability.max(),
        }
        self.log_dict(
            values,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

    def on_validation_epoch_start(self):
        n_foreground = self.num_classes - 1
        self._val_class_dice_sum = torch.zeros(n_foreground, device=self.device, dtype=torch.float64)
        self._val_class_dice_count = torch.zeros(n_foreground, device=self.device, dtype=torch.float64)
        self._val_binary_dice_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_binary_dice_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_predicted_foreground = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_target_foreground = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_false_positive = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_background = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_voxels = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_target_probability_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_target_probability_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_max_foreground_probability = torch.zeros((), device=self.device, dtype=torch.float64)

    def _update_validation_epoch_metrics(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate exact full-fold metrics without treating absent classes as Dice 1."""

        probabilities = torch.softmax(logits.float(), dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        target_index = target.squeeze(1).long()
        predicted_foreground = prediction > 0
        target_foreground = target_index > 0

        for sample_index in range(logits.shape[0]):
            predicted_sample = predicted_foreground[sample_index]
            target_sample = target_foreground[sample_index]
            denominator = predicted_sample.sum() + target_sample.sum()
            if denominator > 0:
                intersection = (predicted_sample & target_sample).sum()
                self._val_binary_dice_sum += (2.0 * intersection / denominator).double()
                self._val_binary_dice_count += 1.0

            for class_id in range(1, self.num_classes):
                predicted_class = prediction[sample_index] == class_id
                target_class = target_index[sample_index] == class_id
                class_denominator = predicted_class.sum() + target_class.sum()
                if class_denominator > 0:
                    class_intersection = (predicted_class & target_class).sum()
                    self._val_class_dice_sum[class_id - 1] += (
                        2.0 * class_intersection / class_denominator
                    ).double()
                    self._val_class_dice_count[class_id - 1] += 1.0

        self._val_predicted_foreground += predicted_foreground.sum().double()
        self._val_target_foreground += target_foreground.sum().double()
        self._val_false_positive += (predicted_foreground & ~target_foreground).sum().double()
        self._val_background += (~target_foreground).sum().double()
        self._val_voxels += torch.tensor(target_index.numel(), device=self.device, dtype=torch.float64)

        foreground_probability = probabilities[:, 1:].sum(dim=1)
        if target_foreground.any():
            self._val_target_probability_sum += foreground_probability[target_foreground].sum().double()
            self._val_target_probability_count += target_foreground.sum().double()
        self._val_max_foreground_probability = torch.maximum(
            self._val_max_foreground_probability,
            foreground_probability.max().double(),
        )

    def on_validation_epoch_end(self):
        sum_tensors = (
            self._val_class_dice_sum,
            self._val_class_dice_count,
            self._val_binary_dice_sum,
            self._val_binary_dice_count,
            self._val_predicted_foreground,
            self._val_target_foreground,
            self._val_false_positive,
            self._val_background,
            self._val_voxels,
            self._val_target_probability_sum,
            self._val_target_probability_count,
        )
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            for value in sum_tensors:
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                self._val_max_foreground_probability,
                op=torch.distributed.ReduceOp.MAX,
            )

        valid_classes = self._val_class_dice_count > 0
        class_dice = torch.zeros_like(self._val_class_dice_sum)
        class_dice[valid_classes] = (
            self._val_class_dice_sum[valid_classes] / self._val_class_dice_count[valid_classes]
        )
        minimum_class_dice = class_dice[valid_classes].min() if valid_classes.any() else class_dice.new_zeros(())
        macro_class_dice = class_dice[valid_classes].mean() if valid_classes.any() else class_dice.new_zeros(())
        binary_dice = self._val_binary_dice_sum / self._val_binary_dice_count.clamp_min(1.0)

        values = {
            "val/foreground_dice": binary_dice.float(),
            "val/macro_foreground_class_dice": macro_class_dice.float(),
            "val/min_foreground_class_dice": minimum_class_dice.float(),
            "val/pred_foreground_fraction": (
                self._val_predicted_foreground / self._val_voxels.clamp_min(1.0)
            ).float(),
            "val/target_foreground_fraction": (
                self._val_target_foreground / self._val_voxels.clamp_min(1.0)
            ).float(),
            "val/false_positive_fraction": (
                self._val_false_positive / self._val_background.clamp_min(1.0)
            ).float(),
            "val/pred_to_target_volume_ratio": (
                self._val_predicted_foreground / self._val_target_foreground.clamp_min(1.0)
            ).float(),
            "val/target_foreground_probability": (
                self._val_target_probability_sum / self._val_target_probability_count.clamp_min(1.0)
            ).float(),
            "val/max_foreground_probability": self._val_max_foreground_probability.float(),
        }
        for class_id, dice_value in enumerate(class_dice, start=1):
            if valid_classes[class_id - 1]:
                values[f"val/exact_dice_{class_id}"] = dice_value.float()

        self.log_dict(values, on_step=False, on_epoch=True, sync_dist=False)

    def on_test_epoch_start(self):
        self.test_metrics = [
            dice,
            f1,
            jaccard,
            precision,
            sensitivity,
            specificity,
            TP,
            FP,
            FN,
            total_pos_gt,
            total_pos_pred,
            volume_similarity,
        ]
        self.results = {}
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x = batch["image"]

        logits = self.model.sliding_window_predict(
            data=x,
            patch_size=self.inference_patch_size,
            overlap=self.inference_overlap,
        )

        src_logits = reverse_preprocessing(logits, batch["properties"])
        src_label = batch["src_label"]
        self.results[batch["file_path"]] = self.compute_metrics_from_confusion_matrix(src_logits, src_label)

    def on_test_epoch_end(self):
        avg_results = {}
        first_file = list(self.results.keys())[0]
        logging.info(f"Test results for {len(self.results)} files:")
        for label in self.results[first_file].keys():
            avg_results[label] = {}
            for metric in self.results[first_file][label].keys():
                avg_results[label][metric] = round(
                    np.nanmean([self.results[path][label][metric] for path in self.results]),
                    4,
                )
                logging.info(f"{label} {metric}: {avg_results[label][metric]}")
        self.results["mean"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        logits = self.model.sliding_window_predict(
            data=x,
            patch_size=self.inference_patch_size,
            overlap=self.inference_overlap,
        )
        src_logits = reverse_preprocessing(logits, batch["properties"])
        return src_logits, batch["properties"]

    def compute_metrics_from_confusion_matrix(self, logits, label):
        metrics = {}
        labels = logits.shape[1]
        cmat = torchmetrics.functional.confusion_matrix(
            logits, label.squeeze(1), task="multiclass", num_classes=logits.shape[1]
        )
        for label in range(labels):
            metrics_for_label = {}
            tp = cmat[label, label]
            fp = torch.sum(cmat[:, label]) - tp
            fn = torch.sum(cmat[label, :]) - tp
            tn = torch.sum(cmat) - tp - fp - fn
            for metric in self.test_metrics:
                metrics_for_label[metric.__name__] = float(metric(tp, fp, tn, fn))
            metrics[str(label)] = metrics_for_label
        return metrics
