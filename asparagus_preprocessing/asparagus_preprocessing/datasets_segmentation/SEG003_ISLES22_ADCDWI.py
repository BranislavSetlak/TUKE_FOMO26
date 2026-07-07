import nibabel as nib
import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import postprocess_standard_dataset
from asparagus_preprocessing.utils.mp import process_dataset_without_table
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir
from asparagus_preprocessing.utils.process_case import preprocess_case_with_label
from asparagus_preprocessing.utils.saving import save_data_and_metadata, save_raw_label
from dataclasses import asdict


def process_sample(file_in, file_out, dataset_config, preprocess_config, saving_config):
    file_out = file_out.replace("_FLAIR", "")
    flair_path = file_in

    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return
    if "_FLAIR" not in flair_path:
        return

    label_path = flair_path.replace("images", "labels_derivatives").replace("anat/", "").replace("_FLAIR", "_msk")
    adc_path = flair_path.replace("anat/", "dwi/").replace("_FLAIR", "_adc")
    dwi_path = flair_path.replace("anat/", "dwi/").replace("_FLAIR", "_dwi")

    source_label = nib.load(label_path)
    adc_file = nib.load(adc_path)
    dwi_file = nib.load(dwi_path)

    image, label, image_props = preprocess_case_with_label(
        images=[adc_file, dwi_file],
        label=source_label,
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
    subdir: str = "ISLES22",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=True,
):
    dataset_config = DatasetConfig(
        task_name="SEG003_ISLES22_ADCDWI",
        in_extensions=[".nii.gz"],
        n_classes=2,
        n_modalities=2,
        split="split_40_10_50",
        patterns_exclusion=["labels_derivatives", "_adc", "_dwi", ".json"],
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
