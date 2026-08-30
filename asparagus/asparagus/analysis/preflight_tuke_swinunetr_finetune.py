"""GPU preflight for the controlled TUKE SwinUNETR segmentation variants."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from asparagus.modules.datasets.TrainDataset import SegDataset
from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from asparagus.modules.networks.swinunetr_hybrid import SwinUNETRSegmentation
from asparagus.modules.transforms.presets import (
    CPU_seg_train_transforms,
    GPU_all_train_transforms,
    GPU_all_train_transforms_gin,
)
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=(96, 96, 96))
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def training_files(task_root):
    split_path = task_root / "split_5fold_cv.json"
    split = read_json(split_path)
    fold = split[0] if isinstance(split, list) else split.get("0", split.get(0))
    if not isinstance(fold, dict) or "train" not in fold:
        raise ValueError(f"Could not read fold 0 train files from {split_path}")
    files = [str(path) for path in fold["train"] if Path(path).is_file()]
    if len(files) < 2:
        raise RuntimeError(f"Need at least two existing training files, found {len(files)}")
    return files


def batch_sample(sample, device):
    return {
        "image": sample["image"].clone().unsqueeze(0).to(device),
        "label": sample["label"].clone().unsqueeze(0).long().to(device),
        "file_path": [sample["file_path"]],
        "transforms_applied": dict(sample.get("transforms_applied", {})),
    }


def find_carvemix_sample(files, cpu_transforms):
    dataset = SegDataset(
        files,
        transforms=cpu_transforms,
        carvemix_probability=1.0,
        carvemix_donor_attempts=8,
    )
    for attempt in range(min(16, len(files) * 2)):
        torch.manual_seed(261200 + attempt)
        sample = dataset[attempt % len(dataset)]
        metadata = sample.get("transforms_applied", {}).get("carvemix", {})
        if metadata.get("applied"):
            return sample, metadata
    raise RuntimeError("CarveMix did not produce a foreground paste in 16 deterministic attempts")


def gradient_summary(module):
    groups = {
        "encoder": "swin_unetr.swinViT.",
        "decoder": "swin_unetr.decoder",
        "output": "swin_unetr.out.",
    }
    result = {}
    for group, prefix in groups.items():
        gradients = [
            parameter.grad.detach().float().abs().sum()
            for name, parameter in module.model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        total = torch.stack(gradients).sum() if gradients else torch.zeros((), device=module.device)
        if not torch.isfinite(total) or total <= 0:
            raise RuntimeError(f"No finite non-zero gradient reached the {group}")
        result[group] = float(total.cpu())
    return result


def run_variant(module, sample, gpu_transforms, device):
    module.zero_grad(set_to_none=True)
    batch = batch_sample(sample, device)
    if gpu_transforms is not None:
        batch = gpu_transforms(batch)
    image = batch["image"]
    label = batch["label"]
    logits = module.model(image)
    if logits.shape[0] != 1 or logits.shape[2:] != label.shape[2:]:
        raise RuntimeError(f"Logit/label shape mismatch: {tuple(logits.shape)} vs {tuple(label.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits")
    loss = module.train_loss(logits, label)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    return {
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradients": gradient_summary(module),
        "labels": sorted(int(value) for value in torch.unique(label).cpu().tolist()),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preflight")
    if any(size <= 0 or size % 32 for size in args.patch_size):
        raise ValueError("Every SwinUNETR patch dimension must be a positive multiple of 32")

    task_root = Path(args.task_root)
    checkpoint = Path(args.checkpoint)
    dataset_json = read_json(task_root / "dataset.json")
    metadata = dataset_json["metadata"]
    files = training_files(task_root)
    cpu_transforms = CPU_seg_train_transforms(args.patch_size)

    torch.manual_seed(261100)
    normal_sample = SegDataset(files, transforms=cpu_transforms)[0]
    carvemix_sample, carvemix_metadata = find_carvemix_sample(files, cpu_transforms)

    # Exercise the exact production collation path with one unmixed and one
    # mixed item.  This guards the nested-mapping schema that caused job 77542
    # to fail before its first optimizer step.
    collated = default_collate([normal_sample, carvemix_sample])
    applied_flags = collated["transforms_applied"]["carvemix"]["applied"]
    if applied_flags.numel() != 2 or bool(applied_flags[0]) or not bool(applied_flags[1]):
        raise RuntimeError(f"Unexpected CarveMix collation flags: {applied_flags}")

    model = SwinUNETRSegmentation(
        input_channels=int(metadata["n_modalities"]),
        output_channels=int(metadata["n_classes"]),
        num_sequence_classes=14,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=True,
    )
    module = SegmentationModule(
        model=model,
        weights=load_checkpoint_state_dict(checkpoint),
        load_decoder=True,
        repeat_stem_weights=True,
        optimizer="AdamW",
        learning_rate=3e-4,
        warmup_epochs=10,
        decoder_warmup_epochs=0,
        inference_patch_size=list(args.patch_size),
    ).to("cuda")
    module.train()

    torch.manual_seed(261300)
    results = {
        "task_root": str(task_root),
        "checkpoint": str(checkpoint),
        "carvemix": carvemix_metadata,
        "carvemix_collation_flags": applied_flags.tolist(),
        "variants": {
            "normal": run_variant(
                module,
                normal_sample,
                GPU_all_train_transforms(ndim=3, deep_supervision=False),
                torch.device("cuda"),
            ),
            "gin": run_variant(
                module,
                normal_sample,
                GPU_all_train_transforms_gin(ndim=3, deep_supervision=False),
                torch.device("cuda"),
            ),
            "gin_carvemix": run_variant(
                module,
                carvemix_sample,
                GPU_all_train_transforms_gin(ndim=3, deep_supervision=False),
                torch.device("cuda"),
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"TUKE_SWINUNETR_FINETUNE_PREFLIGHT_OK output={output}")


if __name__ == "__main__":
    main()
