"""Create true five-fold CV splits while preserving the existing held-out test set."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import KFold, StratifiedKFold


TASK_TYPES = {
    "CLS002_FOMO26_Infarct": "classification",
    "SEG009_FOMO26_Meningioma": "segmentation",
    "REGR002_FOMO26_BrainAge": "regression",
    "SEG010_FOMO26_TrigeminalNeuralgia": "segmentation",
    "CLS003_FOMO26_Polymicrogyria": "classification",
}


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def classification_labels(files: list[str]) -> list[int]:
    labels = []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=False)
        label = data[1]
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError(f"Expected one classification label in {file}, got shape {label.shape}")
            label = label.item()
        labels.append(int(label))
    return labels


def make_task_splits(task_dir: Path, task_type: str, seed: int, overwrite: bool) -> None:
    source_path = task_dir / "split_80_10_10.json"
    test_path = task_dir / "TEST_80_10_10.json"
    output_path = task_dir / "split_5fold_cv.json"

    source_folds = read_json(source_path)
    if not source_folds:
        raise ValueError(f"No folds found in {source_path}")

    trainval = sorted(set(source_folds[0]["train"]) | set(source_folds[0]["val"]))
    trainval_set = set(trainval)
    for fold_index, fold in enumerate(source_folds):
        fold_pool = set(fold["train"]) | set(fold["val"])
        if fold_pool != trainval_set:
            raise ValueError(f"Source fold {fold_index} does not use the same train/validation pool")

    test_files = set(read_json(test_path))
    leakage = trainval_set & test_files
    if leakage:
        raise ValueError(f"Found {len(leakage)} samples in both train/validation and test")
    if len(trainval) < 5:
        raise ValueError(f"Five-fold CV requires at least five samples, got {len(trainval)}")

    if task_type == "classification":
        labels = classification_labels(trainval)
        counts = Counter(labels)
        if min(counts.values()) < 5:
            raise ValueError(
                f"Five stratified folds require at least five samples per class; counts={dict(counts)}"
            )
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        split_indices = splitter.split(trainval, labels)
    else:
        labels = None
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        split_indices = splitter.split(trainval)

    output_folds = []
    validation_files = []
    for fold_index, (train_indices, val_indices) in enumerate(split_indices):
        train_files = [trainval[index] for index in train_indices]
        val_files = [trainval[index] for index in val_indices]
        output_folds.append({"train": train_files, "val": val_files})
        validation_files.extend(val_files)

        message = (
            f"task={task_dir.name} fold={fold_index} train={len(train_files)} val={len(val_files)}"
        )
        if labels is not None:
            train_counts = Counter(labels[index] for index in train_indices)
            val_counts = Counter(labels[index] for index in val_indices)
            message += f" train_classes={dict(train_counts)} val_classes={dict(val_counts)}"
        print(message)

    if len(validation_files) != len(trainval) or set(validation_files) != trainval_set:
        raise RuntimeError("Validation folds do not partition the train/validation pool exactly once")

    if output_path.exists() and not overwrite:
        if read_json(output_path) != output_folds:
            raise FileExistsError(
                f"{output_path} exists with different contents; set OVERWRITE_SPLITS=true to replace it"
            )
        print(f"FIVEFOLD_SPLIT_ALREADY_OK task={task_dir.name} output={output_path}")
        return

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(output_folds, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    print(
        f"FIVEFOLD_SPLIT_OK task={task_dir.name} output={output_path} "
        f"trainval={len(trainval)} held_out_test={len(test_files)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for task, task_type in TASK_TYPES.items():
        make_task_splits(args.data_root / task, task_type, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
