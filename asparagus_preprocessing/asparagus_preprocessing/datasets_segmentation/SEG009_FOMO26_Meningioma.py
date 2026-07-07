import logging
import nibabel as nib
import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import simple_recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import simple_postprocess_standard_dataset
from asparagus_preprocessing.utils.mp import process_dataset_without_table
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths
from asparagus_preprocessing.utils.process_case import preprocess_case_with_label
from asparagus_preprocessing.utils.saving import save_data_and_metadata, save_raw_label
from dataclasses import asdict


def process_sample(file_in, file_out, dataset_config, preprocessing_config, saving_config):
    # Use FLAIR as anchor so each subject is processed once.
    if os.path.basename(file_in) != "flair.nii.gz":
        return

    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return

    case_dir = os.path.dirname(file_in)

    flair_path = os.path.join(case_dir, "flair.nii.gz")
    dwi_path = os.path.join(case_dir, "dwi_b1000.nii.gz")
    label_path = file_in.replace("/preprocessed/", "/labels/").replace("flair.nii.gz", "seg.nii.gz")

    required = [flair_path, dwi_path, label_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        logging.warning(f"Skipping {case_dir}; missing files: {missing}")
        return

    flair = nib.load(flair_path)
    dwi = nib.load(dwi_path)
    label_raw = nib.load(label_path)

    image, label, image_props = preprocess_case_with_label(
        images=[flair, dwi],
        label=label_raw,
        **asdict(preprocessing_config),
        strict=False,
    )

    image_props["src_label_path"] = label_path
    image_props["src_image_paths"] = [flair_path, dwi_path]

    save_data_and_metadata(
        data=image + [label],
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
    )

    save_raw_label(file_out, label_path)


def main(
    path: str = get_source_path(),
    subdir: str = "Task_2/Task_2",
    processes: int = 12,
    bidsify: bool = False,
    save_dset_metadata: bool = False,
    save_as_tensor: bool = True,
):
    dataset_config = DatasetConfig(
        task_name="SEG009_FOMO26_Meningioma",
        in_extensions=[".nii.gz"],
        n_classes=2,
        split="split_80_10_10",
        n_modalities=2,
        patterns_exclusion=[
            "labels",
            "seg.nii.gz",
            "dwi_b1000.nii.gz",
            "swi.nii.gz",
            "t2s.nii.gz",
        ],
    )

    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config(n_modalities=dataset_config.n_modalities)

    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    os.makedirs(target_dir, exist_ok=True)

    files_standard, files_excluded = simple_recursive_find_and_group_files(
        source_dir,
        extensions=dataset_config.in_extensions,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )

    files_standard_out = get_image_output_paths(
        files_standard,
        source_dir,
        target_dir,
        dataset_config.in_extensions,
    )

    process_dataset_without_table(
        process_fn=process_sample,
        files_in=files_standard,
        files_out=files_standard_out,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        processes=processes,
    )

    if saving_config.bidsify or saving_config.save_dset_metadata:
        raise NotImplementedError

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
