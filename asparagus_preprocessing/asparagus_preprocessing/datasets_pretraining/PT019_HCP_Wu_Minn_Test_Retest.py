import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO300K_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.bids import (
    extract_demographics,
    rename_files_with_mapping,
)
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import (
    get_bvals_and_bvecs_v1,
    recursive_find_and_group_files,
    recursive_find_files,
)
from asparagus_preprocessing.utils.metadata_generation import (
    postprocess_standard_dataset,
)
from asparagus_preprocessing.utils.mp import multiprocess_mri_dwi_pet_perf_cases
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir


def main(
    path: str = get_source_path(),
    subdir: str = "HCP_Test_Retest",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    saving_config = get_FOMO300K_saving_config(
        save_as_tensor=save_as_tensor,
        save_dset_metadata=save_dset_metadata,
        bidsify=bidsify,
    )
    preprocessing_config = get_noresampling_preprocessing_config()

    process(
        saving_config=saving_config,
        preprocessing_config=preprocessing_config,
        path=path,
        subdir=subdir,
        processes=processes,
    )


def process(
    saving_config,
    preprocessing_config,
    path: str = get_source_path(),
    subdir: str = "HCP_Test_Retest",
    processes=12,
    task_name: str = "PT019_HCP_Wu_Minn_Test_Retest",
):
    dataset_config = DatasetConfig(
        task_name=task_name,
        n_classes=1,
        n_modalities=1,
        in_extensions=[".nii.gz"],
        split=None,
        patterns_exclusion=["AFI", "BIAS", "FieldMap"],
        # Do not use directly DWI as we do not want the baseline files (3D) to enter DWI processing,
        # with trace/averaging and bval naming convention
        patterns_DWI=["LR.nii.gz", "RL.nii.gz"],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[r"(\d{6})"],
        df_columns=["participant_id", "session_id", "sex", "group"],
    )

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
    files_DWI_out = get_image_output_paths(files_DWI, source_dir, target_dir, dataset_config.in_extensions)
    files_PET_out = get_image_output_paths(files_PET, source_dir, target_dir, dataset_config.in_extensions)
    files_Perf_out = get_image_output_paths(files_Perf, source_dir, target_dir, dataset_config.in_extensions)
    bvals_DWI, bvecs_DWI = get_bvals_and_bvecs_v1(files_DWI, dataset_config.in_extensions)

    multiprocess_mri_dwi_pet_perf_cases(
        files_standard=files_standard,
        files_standard_out=files_standard_out,
        files_DWI=files_DWI,
        bvals_DWI=bvals_DWI,
        bvecs_DWI=bvecs_DWI,
        files_DWI_out=files_DWI_out,
        files_PET=files_PET,
        files_PET_out=files_PET_out,
        files_Perf=files_Perf,
        files_Perf_out=files_Perf_out,
        patterns_m0=dataset_config.patterns_m0,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        processes=processes,
        strict=False,
    )

    if saving_config.bidsify or saving_config.save_dset_metadata:
        hardcoded_metadata = {
            r"T1w_MPR": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM ConnectomS",
                "SoftwareVersions": "syngo MR D11",
                "SequenceName": "MPRAGE",
                "MRAcquisitionType": "3D",
                "SliceThickness": "0.7",
                "EchoTime": "2.14",
                "RepetitionTime": "2400",
                "InversionTime": "1000",
                "FlipAngle": "8",
            },
            r"T2w_SPC": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM ConnectomS",
                "SoftwareVersions": "syngo MR D11",
                "SequenceName": "T2-SPACE",
                "MRAcquisitionType": "3D",
                "SliceThickness": "0.7",
                "EchoTime": "565",
                "RepetitionTime": "3200",
            },
            r"DWI_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM ConnectomS",
                "SoftwareVersions": "syngo MR D11",
                "SequenceName": "Spin-echo EPI",
                "MRAcquisitionType": "3D",
                "SliceThickness": "1.25",
                "EchoTime": "89.50",
                "RepetitionTime": "5520",
            },
        }
        demographics_csv_path = os.path.join(source_dir, "unrestricted_stce_7_8_2025_13_46_17.csv")
        processed_files = recursive_find_files(target_dir, extensions=dataset_config.in_extensions + [".pt"])
        expanded_df, subjects_df, mri_info_df = extract_demographics(
            processed_files=processed_files,
            demographics_csv_path=demographics_csv_path,
            columns_to_keep=dataset_config.df_columns,
            custom_patterns=dataset_config.patterns_bidsify,
            default_group="Control",
            source_dir=source_dir,
            target_dir=target_dir,
            hardcoded_metadata=hardcoded_metadata,
        )

    if saving_config.bidsify:
        subjects_df.to_csv(os.path.join(target_dir, "participants.tsv"), sep="\t", index=False)

        rename_files_with_mapping(
            target_dir=target_dir,
            expanded_df=expanded_df,
        )

    if saving_config.bidsify or saving_config.save_dset_metadata:
        mri_info_df.to_csv(os.path.join(target_dir, "mri_info.tsv"), sep="\t", index=False)

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
