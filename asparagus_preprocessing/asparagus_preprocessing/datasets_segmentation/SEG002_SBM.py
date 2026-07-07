import nibabel as nib
import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import (
    postprocess_standard_dataset,
)
from asparagus_preprocessing.utils.mp import process_dataset_without_table
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir
from asparagus_preprocessing.utils.process_case import preprocess_case_with_label
from asparagus_preprocessing.utils.saving import save_data_and_metadata, save_raw_label
from dataclasses import asdict


def process_sample(file_in, file_out, dataset_config, preprocess_config, saving_config):
    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return

    bravo_path = file_in
    label_path = bravo_path.replace("bravo.nii.gz", "seg.nii.gz")
    flair_path = bravo_path.replace("bravo.nii.gz", "flair.nii.gz")
    t1_gd_path = bravo_path.replace("bravo.nii.gz", "t1_gd.nii.gz")
    t1_pre_path = bravo_path.replace("bravo.nii.gz", "t1_pre.nii.gz")

    label_raw = nib.load(label_path)
    bravo = nib.load(bravo_path)
    flair = nib.load(flair_path)
    t1_gd = nib.load(t1_gd_path)
    t1_pre = nib.load(t1_pre_path)

    image, label, image_props = preprocess_case_with_label(
        images=[bravo, flair, t1_gd, t1_pre],
        label=label_raw,
        **asdict(preprocess_config),
        strict=False,
    )

    image_props["src_label_path"] = label_path

    save_data_and_metadata(
        data=image + [label],
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
    )

    save_raw_label(file_out, label_path)


def main(
    path: str = get_source_path(),
    subdir: str = "SBM-Stanford-Brain-Metastasis-3",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=True,
):
    dataset_config = DatasetConfig(
        task_name="SEG002_SBM",
        in_extensions=[".nii.gz"],
        n_classes=2,
        n_modalities=4,
        split="split_40_10_50",
        patterns_exclusion=[
            "flair",
            "t1_gd",
            "t1_pre",
            "seg",
            "test",
        ],  # the test folder does not have segmentations
        patterns_DWI=[],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[],
        df_columns=[],
    )
    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config(n_modalities=dataset_config.n_modalities)

    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    prepare_target_dir(target_dir, saving_config.save_as_tensor)

    files_standard, files_DWI, files_PET, files_Perf, files_excluded = recursive_find_and_group_files(
        source_dir,
        extensions=dataset_config.in_extensions,
        patterns_dwi=dataset_config.patterns_DWI,
        patterns_pet=dataset_config.patterns_PET,
        patterns_perfusion=dataset_config.patterns_perfusion,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )

    files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, dataset_config.in_extensions)

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

    postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=files_standard,
        source_files_DWI=files_DWI,
        source_files_PET=files_PET,
        source_files_Perf=files_Perf,
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
