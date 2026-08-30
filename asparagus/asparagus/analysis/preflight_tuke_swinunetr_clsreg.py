"""GPU forward/backward and checkpoint-transfer preflight for TUKE cls/reg."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from asparagus.modules.datasets.TrainDataset import ClsRegDataset
from asparagus.modules.lightning_modules.clsreg_module import ClassificationModule, RegressionModule
from asparagus.modules.networks.swinunetr_hybrid import SwinUNETRClsReg
from asparagus.modules.transforms.presets import (
    CPU_clsreg_train_transforms_crop,
    GPU_all_train_transforms,
    GPU_all_train_transforms_gin,
)
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--task-type", choices=("classification", "regression"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_files(task_root):
    split = read_json(task_root / "split_5fold_cv.json")
    fold = split[0] if isinstance(split, list) else split.get("0", split.get(0))
    files = [str(path) for path in fold["train"] if Path(path).is_file()]
    if len(files) < 2:
        raise RuntimeError(f"Need two existing training files, found {len(files)}")
    return files


def gradient_total(model, prefix):
    values = [
        parameter.grad.detach().float().abs().sum()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    total = torch.stack(values).sum() if values else torch.zeros((), device="cuda")
    if not torch.isfinite(total) or total <= 0:
        raise RuntimeError(f"No finite non-zero gradient for {prefix}")
    return float(total.cpu())


def run_variant(module, samples, transform, task_type, num_classes):
    module.zero_grad(set_to_none=True)
    batch = default_collate(samples)
    batch["image"] = batch["image"].cuda(non_blocking=True)
    batch = transform(batch)
    if task_type == "classification":
        target = batch["CLSREG_label"].view(-1).long().cuda(non_blocking=True)
    else:
        target = batch["CLSREG_label"].view(-1, num_classes).float().cuda(non_blocking=True)
    prediction = module.model(batch["image"])
    if tuple(prediction.shape) != (len(samples), num_classes):
        raise RuntimeError(f"Unexpected prediction shape {tuple(prediction.shape)}")
    loss = module.loss(prediction, target)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    return {
        "image_shape": list(batch["image"].shape),
        "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "encoder_gradient": gradient_total(module.model, "swin_unetr.swinViT."),
        "head_gradient": gradient_total(module.model, "downstream_head."),
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    task_root = Path(args.task_root)
    metadata = read_json(task_root / "dataset.json")["metadata"]
    num_classes = int(metadata["n_classes"])
    dataset = ClsRegDataset(
        train_files(task_root),
        transforms=CPU_clsreg_train_transforms_crop((96, 96, 96)),
    )
    torch.manual_seed(262700)
    samples = [dataset[0], dataset[1]]
    model = SwinUNETRClsReg(
        input_channels=int(metadata["n_modalities"]),
        output_channels=num_classes,
        num_sequence_classes=14,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=True,
        downstream_dropout_rate=0.1,
    )
    module_class = ClassificationModule if args.task_type == "classification" else RegressionModule
    module = module_class(
        model=model,
        weights=load_checkpoint_state_dict(args.checkpoint),
        load_decoder=False,
        repeat_stem_weights=True,
        optimizer="AdamW",
        learning_rate=1e-4,
        warmup_epochs=10,
        decoder_warmup_epochs=0,
    ).cuda()
    module.train()
    torch.manual_seed(262800)
    result = {
        "task_root": str(task_root),
        "task_type": args.task_type,
        "checkpoint": str(args.checkpoint),
        "normal": run_variant(
            module, samples, GPU_all_train_transforms(ndim=3), args.task_type, num_classes
        ),
        "gin": run_variant(
            module, samples, GPU_all_train_transforms_gin(ndim=3), args.task_type, num_classes
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"TUKE_SWINUNETR_CLSREG_PREFLIGHT_OK output={output}")


if __name__ == "__main__":
    main()
