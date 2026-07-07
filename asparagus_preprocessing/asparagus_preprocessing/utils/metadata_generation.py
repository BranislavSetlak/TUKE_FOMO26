import asparagus_preprocessing
import os
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.dataclasses import (
    DatasetConfig,
    PreprocessingConfig,
    SavingConfig,
)
from asparagus_preprocessing.utils.detect import (
    find_and_add_test_splits,
    find_and_add_train_splits,
    find_processed_dataset,
    recursive_find_and_group_files,
    recursive_find_files,
)
from asparagus_preprocessing.utils.loading import load_json
from asparagus_preprocessing.utils.saving import enhanced_save_json
from asparagus_preprocessing.utils.splitting import split


def generate_dataset_json(
    output_file: str,
    dataset_name: str,
    metadata: dict = {},
    dataset_config: DatasetConfig = None,
    saving_config: SavingConfig = None,
    preprocessing_config: PreprocessingConfig = None,
) -> None:
    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["metadata"] = metadata
    json_dict["preprocessing_config"] = preprocessing_config
    json_dict["saving_config"] = saving_config
    json_dict["dataset_config"] = dataset_config

    enhanced_save_json(json_dict, os.path.join(output_file))


def simple_postprocess_standard_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    saving_config: SavingConfig,
    target_dir: str,
    source_files_standard: list[str],
    source_files_excluded: list[str],
    processes: int = 12,
) -> None:
    postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=source_files_standard,
        source_files_DWI=[],
        source_files_PET=[],
        source_files_Perf=[],
        source_files_excluded=source_files_excluded,
        processes=processes,
    )


def postprocess_standard_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    saving_config: SavingConfig,
    target_dir: str,
    source_files_standard: list[str],
    source_files_DWI: list[str],
    source_files_PET: list[str],
    source_files_Perf: list[str],
    source_files_excluded: list[str],
    processes: int = 12,
) -> None:
    source_all_files = (
        len(source_files_standard)
        + len(source_files_DWI)
        + len(source_files_PET)
        + len(source_files_Perf)
        + len(source_files_excluded)
    )
    target_all_files = recursive_find_files(target_dir, extensions=dataset_config.in_extensions + [".pt"])

    files_delta = len(target_all_files) - source_all_files

    target_files_standard, target_files_DWI, target_files_PET, target_files_Perf, _ = recursive_find_and_group_files(
        base_path=target_dir,
        extensions=dataset_config.in_extensions + [".pt"],
        patterns_dwi=dataset_config.patterns_DWI,
        patterns_pet=dataset_config.patterns_PET,
        patterns_perfusion=dataset_config.patterns_perfusion,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )
    generate_dataset_json(
        os.path.join(target_dir, "dataset.json"),
        dataset_name=dataset_config.task_name,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        dataset_config=dataset_config,
        metadata={
            "files_source_directory_total": source_all_files,
            "files_source_directory_standard": len(source_files_standard),
            "files_source_directory_DWI": len(source_files_DWI),
            "files_source_directory_PET": len(source_files_PET),
            "files_source_directory_Perfusion": len(source_files_Perf),
            "files_source_directory_excluded": len(source_files_excluded),
            "files_target_directory_total": len(target_all_files),
            "files_target_directory_standard": len(target_files_standard),
            "files_target_directory_DWI": len(target_files_DWI),
            "files_target_directory_PET": len(target_files_PET),
            "files_target_directory_Perfusion": len(target_files_Perf),
            "files_delta_after_processing": files_delta,
            "n_classes": dataset_config.n_classes,
            "n_modalities": dataset_config.n_modalities,
        },
    )
    enhanced_save_json(target_all_files, os.path.join(target_dir, "paths.json"))

    if dataset_config.split is not None:
        save_path = os.path.join(target_dir, dataset_config.split + ".json")
        split(
            files=target_all_files,
            fn=getattr(asparagus_preprocessing.utils.splitting, dataset_config.split),
            save_path=save_path,
        )


def combine_datasets_with_splits(
    dataset_collection: list[str],
) -> tuple[dict[str, dict], list[str], list[dict[str, list]], list[dict]]:
    all_dataset_json = {}
    all_files = []
    all_train_splits = [
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
        {"train": [], "val": []},
    ]
    all_test_splits = []

    for dataset in dataset_collection:
        dataset = find_processed_dataset(dataset)
        dataset_dir = os.path.join(get_data_path(), dataset)
        dataset_json = load_json(os.path.join(dataset_dir, "dataset.json"))
        paths_json = load_json(os.path.join(dataset_dir, "paths.json"))
        all_dataset_json[dataset] = dataset_json
        all_files += paths_json
        all_train_splits = find_and_add_train_splits(dataset_dir, all_train_splits)
        all_test_splits = find_and_add_test_splits(dataset_dir, all_test_splits)
    return all_dataset_json, all_files, all_train_splits, all_test_splits


def combine_datasets_without_splits(
    dataset_collection: list[str],
) -> tuple[dict[str, dict], list[str]]:
    all_dataset_json = {}
    all_files = []

    for dataset in dataset_collection:
        dataset = find_processed_dataset(dataset)
        dataset_dir = os.path.join(get_data_path(), dataset)
        dataset_json = load_json(os.path.join(dataset_dir, "dataset.json"))
        paths_json = load_json(os.path.join(dataset_dir, "paths.json"))
        all_dataset_json[dataset] = dataset_json
        all_files += paths_json
    return all_dataset_json, all_files
