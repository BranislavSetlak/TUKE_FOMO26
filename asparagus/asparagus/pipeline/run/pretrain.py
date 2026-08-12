import glob
import os
import random

import hydra
import lightning as pl
from asparagus.functional.sequence_labels import effective_number_class_weights, sequence_class_counts
from asparagus.functional.versioning import generate_unused_run_id
from asparagus.modules.hydra.plugins.searchpath_plugins import PretrainSearchpathPlugin
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_ssl_plugins, prepare_standard_experiment
from asparagus.pipeline.auto_configuration.logging import logging
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf

load_dotenv()


OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver(
    "version",
    lambda resume_training, run_dir: generate_unused_run_id(resume_training=resume_training, run_dir=run_dir),
    use_cache=True,
)
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(PretrainSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    plugins = prepare_ssl_plugins(cfg)

    assert cfg.task is not None, "Config file is not set up correctly."

    loggers = logging(
        ckpt_wandb_id=version_store.wandb_id,
        ckpt_mlflow_id=version_store.mlflow_id,
        log_file_name=HydraConfig.get().job.name,
        run_dir=path_store.run_dir,
        version=version_store.version,
        wandb_config=logging_safe_cfg,
        wandb_experiment=HydraConfig.get().job.config_name,
        wandb_project=cfg.logger.wandb_project,
        wandb_logging=cfg.logger.wandb_logging,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_to_stdout=cfg.logger.log_to_stdout,
    )

    callbacks = [
        TQDMProgressBar(refresh_rate=cfg.logger.log_every_n_steps),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
    ] + plugins

    checkpoint_frequency = {}
    if cfg.model.ckpt_every_n_train_steps is not None:
        checkpoint_frequency["every_n_train_steps"] = int(cfg.model.ckpt_every_n_train_steps)
    else:
        checkpoint_frequency["every_n_epochs"] = int(cfg.model.ckpt_every_n_epoch)
    callbacks.insert(
        1,
        ModelCheckpoint(
            dirpath=path_store.ckpt_save_dir,
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            filename="step={step:09d}",
            auto_insert_metric_name=False,
            enable_version_counter=False,
            **checkpoint_frequency,
        ),
    )

    if cfg.profiler.enabled:
        callbacks.append(instantiate(cfg.profiler._callback))

    cpu_tr_transforms = instantiate(
        cfg.transforms._cpu_tr_transforms,
        patch_size=cfg.training.patch_size,
    )
    cpu_val_transforms = instantiate(
        cfg.transforms._cpu_val_transforms,
        patch_size=cfg.training.patch_size,
    )
    gpu_tr_transforms = instantiate(
        cfg.transforms._gpu_tr_transforms,
        cfg.transforms.masking,
        ndim=len(cfg.training.patch_size),
        mask_ratio=cfg.training.mask_ratio,
    )
    gpu_val_transforms = instantiate(
        cfg.transforms._gpu_val_transforms,
        cfg.transforms.masking,
        mask_ratio=cfg.training.mask_ratio,
    )

    model = instantiate(
        cfg.model._pretrain_net,
    )

    sequence_kwargs = {}
    sequence_module_kwargs = {}
    sequence_cfg = cfg.get("sequence")
    if sequence_cfg is not None and sequence_cfg.get("enabled", False):
        raw_to_class = OmegaConf.to_container(sequence_cfg.raw_to_class, resolve=True)
        ignored_sequences = OmegaConf.to_container(sequence_cfg.ignored_sequences, resolve=True)
        class_counts = sequence_class_counts(
            file_store.splits["train"],
            raw_to_class=raw_to_class,
            ignored_sequences=ignored_sequences,
            other_class_id=sequence_cfg.other_class_id,
            num_classes=sequence_cfg.num_classes,
        )
        class_weights = effective_number_class_weights(
            class_counts,
            beta=sequence_cfg.effective_number_beta,
        )
        sequence_kwargs = {
            "sequence_raw_to_class": raw_to_class,
            "sequence_ignored": ignored_sequences,
            "sequence_other_class_id": sequence_cfg.other_class_id,
        }
        sequence_module_kwargs = {
            "sequence_loss_weight": sequence_cfg.sequence_loss_weight,
            "sequence_class_weights": class_weights,
            "sequence_ignore_index": sequence_cfg.ignore_index,
        }
        if int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))) == 0:
            print("Sequence classification targets (training split):")
            for class_id, (class_name, count, weight) in enumerate(
                zip(sequence_cfg.class_names, class_counts, class_weights)
            ):
                print(f"  {class_id:2d} {class_name:12s} count={count:8d} weight={weight:.6f}")

    data_module = instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        **sequence_kwargs,
    )

    model_module = instantiate(
        cfg.lightning._lightning_module,
        model=model,
        learning_rate=cfg.model.pretrain_lr,
        warmup_epochs=cfg.training.warmup_epochs,
        train_transforms=gpu_tr_transforms,
        val_transforms=gpu_val_transforms,
        rec_loss_masked_only=cfg.training.rec_loss_masked_only,
        optimizer=cfg.model.pretrain_optim,
        mlflow_logging=cfg.logger.mlflow_logging,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        log_images_every_n_epoch=cfg.logger.log_images_every_n_epoch,
        **sequence_module_kwargs,
    )

    trainer = instantiate(
        cfg.lightning._trainer,
        callbacks=callbacks,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        logger=loggers,
        default_root_dir=path_store.run_dir,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_steps=cfg.training.steps,
        limit_train_batches=cfg.training.steps_per_epoch,
        limit_val_batches=cfg.training.val_steps_per_epoch,
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )

    if trainer.is_global_zero:
        print("Training duration configured as:")
        print(f"  - Steps: {cfg.training.steps}")
        print(f"  - Global batch size: {cfg.training.global_batch_size}")
        print(f"  - Steps per pseudo epoch: {cfg.training.steps_per_epoch}")
        print(f"  - Validation steps per pseudo epoch: {cfg.training.val_steps_per_epoch}")
        print(
            "  - Pseudo Epochs: {:.1f}".format(
                cfg.training.steps * cfg.training.accumulate_grad_batches / cfg.training.steps_per_epoch
            )
        )
        print(f"  - Warmup Pseudo Epochs: {cfg.training.warmup_epochs} (ratio {cfg.training.warmup_ratio})")

    resume_candidates = [os.path.join(path_store.ckpt_save_dir, "last.ckpt")]
    resume_candidates.extend(glob.glob(os.path.join(path_store.run_dir, "hpc_ckpt_*.ckpt")))
    resume_candidates = [path for path in resume_candidates if os.path.isfile(path)]
    ckpt_path = None
    if cfg.resume_training and resume_candidates:
        ckpt_path = max(resume_candidates, key=os.path.getmtime)
    if trainer.is_global_zero:
        print(f"Checkpoint restart: {ckpt_path if ckpt_path is not None else 'starting a new run'}")

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
