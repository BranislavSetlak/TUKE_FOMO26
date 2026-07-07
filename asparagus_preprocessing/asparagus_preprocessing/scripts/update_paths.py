import argparse
import os
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.detect import find_processed_dataset, update_paths
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Update paths in a dataset after changing directory structure or compute environment."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Update all datasets found in the data directory.",
    )
    parser.add_argument(
        "dataset",
        nargs="*",
        type=str,
        help="Dataset ID/full name to update (e.g., SEG001, CLS002, or CLS002_SLIM)",
    )

    args = parser.parse_args()
    if args.all:
        print("Updating all datasets...")
        datasets_to_update = [p.name for p in Path(get_data_path()).iterdir() if p.is_dir()]
    else:
        print("Updating specified datasets...")
        datasets_to_update = [find_processed_dataset(ds) for ds in args.dataset]

    for dataset in datasets_to_update:
        print(f"Updating dataset {dataset}...")

        dataset_dir = os.path.join(get_data_path(), dataset)
        update_paths(dataset_dir)


if __name__ == "__main__":
    main()
