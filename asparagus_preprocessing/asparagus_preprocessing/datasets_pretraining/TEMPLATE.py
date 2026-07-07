"""
Pretraining dataset preprocessing template.
Copy this file, rename it to PT<NNN>_<DatasetName>.py, and fill in every TODO.
"""

import os
from asparagus_preprocessing.configs.preprocessing_presets import get_FOMO_saving_config, get_noresampling_preprocessing_config
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import simple_recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import simple_postprocess_standard_dataset
from asparagus_preprocessing.utils.mp import multiprocess_mri_dwi_pet_perf_cases
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir


def main(
    path: str = get_source_path(),
    subdir: str = "MyDataset",  # name of your dataset folder inside get_source_path()
    processes: int = 12,
    bidsify: bool = False,
    save_dset_metadata: bool = False,
    save_as_tensor: bool = True,
):
    dataset_config = DatasetConfig(
        task_name="PT0XX_MyDataset",  # unique name; used as the output folder name
        n_classes=1,
        n_modalities=1,
        in_extensions=[".nii.gz"],  # file extensions to look for, e.g. [".nii", ".nii.gz"]
        split=None,
        patterns_exclusion=["func", "fmap"],  # path substrings: matching files are skipped entirely
        patterns_bidsify=[r"sub-([0-9]{4})"],  # regex to extract subject ID from file path
        df_columns=["participant_id", "session_id", "age", "sex", "group"],
    )

    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config()

    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    prepare_target_dir(target_dir, saving_config.save_as_tensor)

    files_standard, files_excluded = simple_recursive_find_and_group_files(
        source_dir,
        extensions=dataset_config.in_extensions,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )

    files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, dataset_config.in_extensions)

    multiprocess_mri_dwi_pet_perf_cases(
        files_standard=files_standard,
        files_standard_out=files_standard_out,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        processes=processes,
        strict=False,
    )

    simple_postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=files_standard,
        source_files_excluded=files_excluded,
        processes=processes,
    )


if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )
