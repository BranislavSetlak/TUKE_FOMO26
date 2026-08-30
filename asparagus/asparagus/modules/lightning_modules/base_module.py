import lightning as L
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from asparagus.functional.lr_scheduling import (
    cosine_decay_schedule,
    sawtooth_warmup_cosine_decay_schedule,
    separate_encoder_decoder_weights,
    simple_warmup_cosine_decay_schedule,
)
from asparagus.functional.pos_embed import resize_pos_embed_3d
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from torch.optim import SGD, AdamW
from torchvision import transforms
from typing import Optional


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        warmup_epochs: int = None,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        load_decoder: bool = True,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        repeat_stem_weights: bool = True,
        pretrained_target_size: Optional[tuple] = None,
        target_size: Optional[tuple] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.pretrained_target_size = pretrained_target_size
        self.target_size = target_size

        self.loss = None
        self.train_metrics = None
        self.val_metrics = None
        self.warmup_epochs = warmup_epochs
        self.decoder_warmup_epochs = decoder_warmup_epochs
        self.ignore_index_in_metrics = 0
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.repeat_stem_weights = repeat_stem_weights
        assert 0 < cosine_period_ratio <= 1

        self.save_hyperparameters(ignore=["model", "weights", "train_transforms", "val_transforms", "test_transforms"])
        self.model = model

        if weights is not None:
            self.load_state_dict(weights, load_decoder=load_decoder, strict=False)

        self.model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # Separate encoder and decoder parameters for different warmup schedules
        if self.decoder_warmup_epochs > 0:
            param_groups = separate_encoder_decoder_weights(self.named_parameters())
        else:
            param_groups = self.parameters()

        if self.optimizer == "SGD":
            optimizer = SGD(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            optimizer = AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        print(f"Using optimizer {optimizer.__class__.__name__} with learning rate {self.learning_rate}")

        # Calculate steps per epoch based on trainer configuration
        # if max_epochs is *not* set (i.e., set to -1), we are probably using max_steps
        # if max_epochs is set, we can calculate steps per epoch based on estimated_stepping_batches
        if self.trainer.max_epochs <= 0:
            optimizer_steps_per_epoch = self.trainer.limit_train_batches // self.trainer.accumulate_grad_batches
        else:
            optimizer_steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # Scheduler option 1: Three-phase schedule with separate decoder/joint warmup
        if self.decoder_warmup_epochs > 0:
            scheduler = sawtooth_warmup_cosine_decay_schedule(
                optimizer,
                self.decoder_warmup_epochs,
                self.warmup_epochs,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
            )
        # Scheduler option 2: Two-phase schedule with joint warmup
        elif self.warmup_epochs > 0:
            scheduler = simple_warmup_cosine_decay_schedule(
                optimizer,
                self.warmup_epochs,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
            )
        # Scheduler option 3: Just cosine annealing
        else:
            scheduler = cosine_decay_schedule(
                optimizer,
                optimizer_steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,  # may be -1, if using max_steps
                self.trainer.max_steps,  # may be -1, if using max_epochs
            )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def load_state_dict(self, state_dict, strict=True, assign=False, *, load_decoder=True):
        """Load either transfer weights or a native Lightning checkpoint safely.

        The return value intentionally matches ``torch.nn.Module.load_state_dict``.
        Lightning consumes that value when it restores ``best.ckpt`` or resumes
        training.  A checkpoint is considered loaded when keys were accepted,
        even when its values equal the model's current values.
        """

        if not state_dict:
            raise ValueError("Cannot load an empty state_dict")

        state_dict = dict(state_dict)
        target_params = self.state_dict()
        target_compiled = any(key.startswith("model._orig_mod.") for key in target_params)
        source_compiled = any(key.startswith("model._orig_mod.") for key in state_dict)
        print(f"Target compiled: {target_compiled}, source compiled: {source_compiled}")

        if source_compiled and not target_compiled:
            print("Removing _orig_mod from compiled source state_dict keys.")
            state_dict = {key.replace("model._orig_mod.", "model.", 1): value for key, value in state_dict.items()}
        elif target_compiled and not source_compiled:
            print("Adding _orig_mod to source state_dict keys for the compiled target.")
            state_dict = {
                (key.replace("model.", "model._orig_mod.", 1) if key.startswith("model.") else key): value
                for key, value in state_dict.items()
            }

        stem_weight_names = getattr(self.model, "stem_weight_names", None)
        if stem_weight_names is None:
            stem_weight_name = getattr(self.model, "stem_weight_name", None)
            stem_weight_names = [] if stem_weight_name is None else [stem_weight_name]
        if self.repeat_stem_weights:
            prefix = "model._orig_mod." if target_compiled else "model."
            for stem_weight_name in stem_weight_names:
                stem_name = f"{prefix}{stem_weight_name}"
                if stem_name not in state_dict or stem_name not in target_params:
                    raise KeyError(f"Configured stem weight is absent: {stem_name}")
                source_channels = state_dict[stem_name].shape[1]
                target_channels = target_params[stem_name].shape[1]
                if source_channels == target_channels:
                    continue
                if source_channels != 1 or target_channels <= 1:
                    raise ValueError(
                        f"Cannot adapt {stem_name} from {source_channels} to {target_channels} input channels"
                    )
                print(f"Repeating stem weights from {source_channels} to {target_channels} channels for {stem_name}.")
                repeats = [1] * state_dict[stem_name].ndim
                repeats[1] = target_channels
                state_dict[stem_name] = state_dict[stem_name].repeat(*repeats) / target_channels

        if self.pretrained_target_size is not None and self.target_size is not None:
            for key in list(state_dict):
                if key not in target_params or target_params[key].shape == state_dict[key].shape:
                    continue
                if key.endswith("pos_embed"):
                    num_prefix_tokens = getattr(self.model.eva, "num_prefix_tokens", 0)
                    patch_embed_size = tuple(self.model.encoder.proj.weight.shape[2:])
                    print(f"Interpolating {key}: {state_dict[key].shape} -> {target_params[key].shape}")
                    state_dict[key] = resize_pos_embed_3d(
                        state_dict[key],
                        target_params[key],
                        num_prefix_tokens=num_prefix_tokens,
                        pretrained_target_size=self.pretrained_target_size,
                        target_size=self.target_size,
                        patch_embed_size=patch_embed_size,
                    )

        decoder_prefixes = tuple(getattr(self.model, "decoder_weight_prefixes", ("decoder",)))

        def is_decoder_key(key):
            relative_key = key
            for prefix in ("model._orig_mod.", "model."):
                if relative_key.startswith(prefix):
                    relative_key = relative_key[len(prefix) :]
                    break
            return any(relative_key.startswith(prefix) for prefix in decoder_prefixes)

        rejected_new = []
        rejected_shape = []
        rejected_decoder = []
        accepted = {}
        for key, value in state_dict.items():
            if key not in target_params:
                rejected_new.append(key)
            elif target_params[key].shape != value.shape:
                rejected_shape.append(key)
            elif not load_decoder and is_decoder_key(key):
                rejected_decoder.append(key)
            else:
                accepted[key] = value

        if not accepted:
            raise RuntimeError("No compatible tensors were found in the supplied state_dict")

        incompatible = super().load_state_dict(accepted, strict=strict, assign=assign)
        print(f"Transferred {len(accepted)}/{len(target_params)} target tensors")
        print(
            "Rejected checkpoint keys:\n"
            f"Not in target: {rejected_new}.\n"
            f"Wrong shape: {rejected_shape}.\n"
            f"Decoder disabled: {rejected_decoder}."
        )
        if not load_decoder:
            print("Decoder weights were not loaded, as requested.")
        else:
            print("Decoder loading was enabled.")
        return incompatible

    def _log_dict_of_images_to_wandb(self, imagedict: dict, log_key: str, task_type: str = ""):
        """
        Log a random image from the imagedict to wandb
        """
        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        image, output, target = get_logger_compatible_image_output_target(
            image=imagedict["input"][batch_idx],
            output=imagedict["output"][batch_idx],
            target=imagedict["target"][batch_idx],
            task_type=task_type,
        )
        for logger in self.trainer.loggers:
            if "WandbLogger" in logger.__class__.__name__:
                log_image_output_target_to_wandb(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )
            if "MLFlowLogger" in logger.__class__.__name__:
                log_image_output_target_to_mlflow(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        if (self.trainer.validating or self.trainer.sanity_checking) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        if (self.trainer.testing or self.trainer.predicting) and self.test_transforms is not None:
            batch = self.test_transforms(batch)
        return super().on_after_batch_transfer(batch, dataloader_idx)
