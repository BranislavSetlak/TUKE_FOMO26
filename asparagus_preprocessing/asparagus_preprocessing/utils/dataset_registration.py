import logging
import os
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.detect import recursive_find_files
from asparagus_preprocessing.utils.metadata_generation import generate_dataset_json
from asparagus_preprocessing.utils.path import prepare_target_dir
from asparagus_preprocessing.utils.saving import enhanced_save_json
from asparagus_preprocessing.utils.splitting import non_stratified_split


def register_dataset(
    task_name: str,
    n_classes: int = 1,
    n_modalities: int = 1,
    # --- Directory mode ---
    data_dir: str | None = None,
    extension: str = ".nii.gz",
    # --- Explicit mode ---
    train: list[str] | None = None,
    val: list[str] | None = None,
    test: list[str] | None = None,
    # --- Split options (directory mode only) ---
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    n_folds: int = 5,
    seed: int = 42,
    # --- Output ---
    output_dir: str | None = None,
    split_name: str | None = None,
) -> str:
    """Register a preprocessed dataset for asparagus training.

    Generates the metadata JSONs (dataset.json, paths.json, split_*.json)
    required by the asparagus training pipeline. Supports two modes:

    Directory mode: provide ``data_dir`` + ``extension`` to auto-discover
    files and generate random train/val/test splits.

    Explicit mode: provide ``train`` + ``val`` (and optionally ``test``)
    file lists directly.

    Args:
        task_name: Dataset name (e.g. "PT100_MyBrains").
        n_classes: Number of output classes.
        n_modalities: Number of input modalities/channels.
        data_dir: Path to directory with preprocessed files (directory mode).
        extension: File extension to search for (directory mode).
        train: List of absolute training file paths (explicit mode).
        val: List of absolute validation file paths (explicit mode).
        test: List of absolute test file paths (optional, both modes).
        train_ratio: Training ratio 0-1 (directory mode).
        val_ratio: Validation ratio 0-1 (directory mode).
        test_ratio: Test ratio 0-1 (directory mode).
        n_folds: Number of cross-validation folds.
        seed: Random seed for splitting.
        output_dir: Where to write JSONs. Default: $ASPARAGUS_DATA/{task_name}.
        split_name: Custom split name. Default: auto from ratios / "split_manual".

    Returns:
        The output directory path.
    """
    _validate_inputs(data_dir, train, val, extension, train_ratio, val_ratio, test_ratio)

    if output_dir is None:
        output_dir = os.path.join(get_data_path(), task_name)

    save_as_tensor = extension == ".pt"
    prepare_target_dir(output_dir, save_as_tensor)

    _warn_existing_files(output_dir)

    if data_dir is not None:
        all_files, splits, test_files, split_label = _directory_mode(
            data_dir, extension, train_ratio, val_ratio, test_ratio, n_folds, seed
        )
    else:
        all_files, splits, test_files, split_label = _explicit_mode(train, val, test, n_folds)

    if split_name is None:
        split_name = split_label

    # Write dataset.json
    generate_dataset_json(
        output_file=os.path.join(output_dir, "dataset.json"),
        dataset_name=task_name,
        metadata={"n_classes": n_classes, "n_modalities": n_modalities},
        dataset_config={"task_name": task_name, "n_classes": n_classes, "n_modalities": n_modalities, "split": split_name},
        saving_config={"save_as_tensor": save_as_tensor},
    )

    # Write paths.json
    enhanced_save_json(sorted(all_files), os.path.join(output_dir, "paths.json"))

    # Write split JSON
    enhanced_save_json(splits, os.path.join(output_dir, f"{split_name}.json"))

    # Write test JSON (optional)
    if test_files:
        test_split_name = split_name.replace("split_", "TEST_")
        enhanced_save_json(test_files, os.path.join(output_dir, f"{test_split_name}.json"))

    _print_summary(output_dir, split_name, splits, test_files, all_files)
    return output_dir


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(data_dir, train, val, extension, train_ratio, val_ratio, test_ratio):
    if data_dir is not None and train is not None:
        raise ValueError("Provide either data_dir (directory mode) or train/val (explicit mode), not both.")
    if data_dir is None and train is None:
        raise ValueError("Provide either data_dir (directory mode) or train/val (explicit mode).")
    if train is not None and val is None:
        raise ValueError("When using explicit mode, both train and val must be provided.")
    if not extension.startswith("."):
        raise ValueError(f"Extension must start with '.', got: '{extension}'")
    if data_dir is not None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.4f} ({train_ratio} + {val_ratio} + {test_ratio})")
    if train is not None:
        for path in train + val:
            if not os.path.isabs(path):
                raise ValueError(f"All paths must be absolute. Got relative path: {path}")


def _warn_existing_files(output_dir):
    existing = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    if existing:
        logging.warning(f"Output directory already contains JSON files: {existing}. They may be overwritten.")


def _directory_mode(data_dir, extension, train_ratio, val_ratio, test_ratio, n_folds, seed):
    files = sorted(recursive_find_files(os.path.abspath(data_dir), [extension]))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 files to create a split, found {len(files)} in {data_dir}")

    splits, test_files = _make_splits(files, train_ratio, val_ratio, test_ratio, n_folds, seed)
    all_files = files

    train_pct = round(train_ratio * 100)
    val_pct = round(val_ratio * 100)
    test_pct = round(test_ratio * 100)
    split_name = f"split_{train_pct:02d}_{val_pct:02d}_{test_pct:02d}"

    return all_files, splits, test_files, split_name


def _explicit_mode(train, val, test, n_folds):
    splits = [{"train": list(train), "val": list(val)} for _ in range(n_folds)]
    test_files = list(test) if test else None
    all_files = list(train) + list(val) + (list(test) if test else [])
    return all_files, splits, test_files, "split_manual"


def _make_splits(files, train_ratio, val_ratio, test_ratio, n_folds, seed):
    trainval, test_files = non_stratified_split(files, train_ratio, val_ratio, test_ratio, test=True, base_seed=seed)
    splits = []
    for i in range(n_folds):
        train, val = non_stratified_split(
            trainval, train_ratio, val_ratio, test_ratio, test=False, seed_increment=i, base_seed=seed
        )
        splits.append({"train": train, "val": val})
    if not test_files:
        test_files = None
    return splits, test_files


def _print_summary(output_dir, split_name, splits, test_files, all_files):
    print(f"\nDataset registered successfully at: {output_dir}")
    print(f"  Total files:    {len(all_files)}")
    print(f"  Train files:    {len(splits[0]['train'])} (fold 0)")
    print(f"  Val files:      {len(splits[0]['val'])} (fold 0)")
    if test_files:
        print(f"  Test files:     {len(test_files)}")
    print(f"  Folds:          {len(splits)}")
    print(f"  Split name:     {split_name}")
    print("\nTo use with asparagus:")
    print(f"  asp_pretrain task={os.path.basename(output_dir)} data.train_split={split_name}")
