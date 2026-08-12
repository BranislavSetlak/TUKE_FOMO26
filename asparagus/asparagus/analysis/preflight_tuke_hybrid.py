#!/usr/bin/env python3
"""Preflight the TUKE hybrid dataset, labels, and SwinUNETR forward pass."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from asparagus.analysis.inventory_fomo_sequences import current_path, extract_split_paths, select_fold
from asparagus.functional.loading import load_image_file
from asparagus.functional.sequence_labels import (
    effective_number_class_weights,
    sequence_and_variant,
    sequence_class_counts,
    sequence_class_id,
)
from asparagus.modules.networks.swinunetr_hybrid import SwinUNETRHybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--mapping-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--expected-train", type=int, default=303_144)
    parser.add_argument("--expected-validation", type=int, default=3_063)
    parser.add_argument("--expected-raw-labels", type=int, default=25)
    parser.add_argument("--check-files", type=int, default=28)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-forward", action="store_true")
    return parser.parse_args()


def evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0 or not values:
        return []
    if count == 1:
        return [values[0]]
    if count >= len(values):
        return values
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    required = {
        "num_classes",
        "other_class_id",
        "class_names",
        "raw_to_class",
        "ignored_sequences",
        "ignore_index",
        "effective_number_beta",
        "sequence_loss_weight",
    }
    missing = sorted(required - mapping.keys())
    if missing:
        raise KeyError(f"Sequence mapping is missing keys: {missing}")
    if len(mapping["class_names"]) != mapping["num_classes"]:
        raise ValueError("class_names length does not equal num_classes")
    if set(mapping["ignored_sequences"].values()) != {mapping["ignore_index"]}:
        raise ValueError("Every ignored sequence must map to ignore_index")
    return mapping


def inspect_tensors(paths: list[str]) -> list[dict[str, Any]]:
    inspected = []
    for original_path in paths:
        path = current_path(original_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        tensor = load_image_file(path)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"Expected [1, D, H, W] tensor at {path}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor values at {path}")
        inspected.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "minimum": float(tensor.min()),
                "maximum": float(tensor.max()),
            }
        )
        del tensor
    return inspected


def check_model(args: argparse.Namespace, num_classes: int) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = SwinUNETRHybrid(
        input_channels=1,
        output_channels=1,
        num_sequence_classes=num_classes,
        feature_size=args.feature_size,
        # Checkpointing is a backward-pass memory optimization. Disabling it
        # avoids wrapping an inference-only preflight in autograd checkpointing;
        # the smoke job exercises the configured use_checkpoint=true path.
        use_checkpoint=False,
    ).to(device)
    model.eval()
    x = torch.randn((1, 1, *args.patch_size), device=device)
    autocast_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        reconstruction, features, logits = model.forward_with_features(x)
    if reconstruction.shape != x.shape:
        raise ValueError(f"Reconstruction shape {tuple(reconstruction.shape)} != input shape {tuple(x.shape)}")
    if logits.shape != (1, num_classes):
        raise ValueError(f"Sequence-logit shape is {tuple(logits.shape)}, expected {(1, num_classes)}")
    result = {
        "input_shape": list(x.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "feature_shape": list(features.shape),
        "sequence_logits_shape": list(logits.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "autocast_bfloat16": autocast_enabled,
    }
    del model, x, reconstruction, features, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.fold < 0:
        raise ValueError("--fold must be non-negative")
    for path in (args.split_json, args.mapping_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping_yaml)
    with args.split_json.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    fold_data, fold_description = select_fold(document, args.fold)
    split_paths = extract_split_paths(fold_data)
    train_paths = sorted(split_paths.get("train", set()))
    validation_paths = sorted(split_paths.get("validation", set()))

    if len(train_paths) != args.expected_train:
        raise ValueError(f"Expected {args.expected_train} train scans, found {len(train_paths)}")
    if len(validation_paths) != args.expected_validation:
        raise ValueError(f"Expected {args.expected_validation} validation scans, found {len(validation_paths)}")
    overlap = set(train_paths).intersection(validation_paths)
    if overlap:
        raise ValueError(f"Train/validation overlap contains {len(overlap)} paths")

    all_paths = train_paths + validation_paths
    raw_counts = Counter(sequence_and_variant(path)[0] for path in all_paths)
    if len(raw_counts) != args.expected_raw_labels:
        raise ValueError(f"Expected {args.expected_raw_labels} raw sequence labels, found {len(raw_counts)}")

    counts = sequence_class_counts(
        train_paths,
        raw_to_class=mapping["raw_to_class"],
        ignored_sequences=mapping["ignored_sequences"],
        other_class_id=mapping["other_class_id"],
        num_classes=mapping["num_classes"],
    )
    weights = effective_number_class_weights(counts, beta=mapping["effective_number_beta"])

    sample_paths = evenly_spaced(all_paths, args.check_files)
    tensor_summary = inspect_tensors(sample_paths)
    sample_targets = Counter(
        sequence_class_id(
            path,
            raw_to_class=mapping["raw_to_class"],
            ignored_sequences=mapping["ignored_sequences"],
            other_class_id=mapping["other_class_id"],
        )
        for path in sample_paths
    )

    model_summary = None if args.skip_forward else check_model(args, mapping["num_classes"])
    summary = {
        "status": "ok",
        "fold_selection": fold_description,
        "train_scans": len(train_paths),
        "validation_scans": len(validation_paths),
        "unique_scans": len(set(all_paths)),
        "raw_sequence_counts": dict(raw_counts.most_common()),
        "class_names": mapping["class_names"],
        "class_counts": counts,
        "class_weights": weights,
        "sequence_loss_weight": mapping["sequence_loss_weight"],
        "sample_targets": {str(key): value for key, value in sorted(sample_targets.items())},
        "tensors_checked": tensor_summary,
        "model": model_summary,
        "torch_version": torch.__version__,
    }
    summary_path = args.output_dir / "preflight_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"PREFLIGHT_OK={summary_path}")


if __name__ == "__main__":
    main()
