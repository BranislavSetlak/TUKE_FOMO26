import os
import pandas as pd
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
    subdir: str = "OASIS2/OASIS2_NIfTI",
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
    subdir: str = "OASIS2/OASIS2_NIfTI",
    processes=12,
    task_name: str = "PT029_OASIS2",
):
    dataset_config = DatasetConfig(
        task_name=task_name,
        n_classes=1,
        n_modalities=1,
        in_extensions=[".nii.gz"],
        split=None,
        patterns_exclusion=[],
        patterns_DWI=[],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[r"(OAS2_\d{4})_(MR\d{1})"],
        df_columns=["participant_id", "session_id", "sex", "age", "group"],
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
            r"mpr-": {
                "Modality": "MR",
                "MagneticFieldStrength": "1.5",
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "Vision",
                "SequenceName": "MP-RAGE",
                "SliceThickness": "1.25",
                "EchoTime": "4.0",
                "RepetitionTime": "9.7",
                "InversionTime": "20",
                "FlipAngle": "10",
            },
        }
        demographics_csv_path = os.path.join(target_dir, "tmp")
        # Read and preprocess the Excel file
        df = pd.read_excel(os.path.join(path, "OASIS2", "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"))
        # PREPROCESSING: Align demographics with file patterns
        # Extract subject ID from MRI ID to match file patterns
        df["Subject_Clean"] = df["MRI ID"].str.extract(r"(OAS2_\d{4})")
        # Create session IDs that match your file pattern (MR1, MR2, etc.)
        df["Session_Clean"] = df["MRI ID"].str.extract(r"OAS2_\d{4}_(MR\d+)")
        # Rename columns to match what extract_demographics expects
        df = df.rename(
            columns={
                "Subject_Clean": "Subject ID",  # This will be the subject identifier
                "Session_Clean": "Session",  # This will be the session identifier
                "M/F": "Sex",  # Standardize sex column name
                "Age": "Age",  # Keep age as is
                "Group": "Group",  # Keep group as is
            }
        )
        # Keep only the columns we need and remove any rows with missing essential data
        essential_columns = ["Subject ID", "Session", "Sex", "Age", "Group"]
        df = df[essential_columns].dropna(subset=["Subject ID", "Session"])
        # Save the preprocessed data
        df.to_csv(demographics_csv_path, index=False)
        processed_files = recursive_find_files(target_dir, extensions=dataset_config.in_extensions + [".pt"])
        expanded_df, subjects_df, mri_info_df = extract_demographics(
            processed_files=processed_files,
            demographics_csv_path=demographics_csv_path,
            columns_to_keep=dataset_config.df_columns,
            custom_patterns=dataset_config.patterns_bidsify,
            source_dir=source_dir,
            target_dir=target_dir,
            hardcoded_metadata=hardcoded_metadata,
        )

        os.remove(demographics_csv_path)

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
