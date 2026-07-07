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
    subdir: str = "ADHD_200",
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
    subdir: str = "ADHD_200",
    processes=12,
    task_name: str = "PT005_ADHD_200",
):
    dataset_config = DatasetConfig(
        task_name=task_name,
        n_classes=1,
        n_modalities=1,
        in_extensions=[".nii.gz"],
        split=None,
        patterns_exclusion=["rest_"],
        patterns_DWI=[],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[
            r"(Brown_\d{5})_(\d+)",  # Group 1: Brown_26001, Group 2: 1
            r"(KKI_\d{5})_(\d+)",
            r"(KKI_\d{7})_(\d+)",
            r"(NeuroIMAGE_\d{5})_(\d+)",
            r"(NeuroIMAGE_\d{7})_(\d+)",
            r"(NYU_\d{5})_(\d+)",
            r"(NYU_\d{7})_(\d+)",
            r"(OHSU_\d{5})_(\d+)",
            r"(OHSU_\d{7})_(\d+)",
            r"(Peking_\d{5})_(\d+)",
            r"(Peking_\d{7})_(\d+)",
            r"(Pittsburgh_\d{5})_(\d+)",
            r"(WashU_\d{5})_(\d+)",
        ],
        df_columns=[
            "participant_id",
            "session_id",
            "age",
            "sex",
            "handedness",
            "group",
        ],
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
            "Brown_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",  # TrioTim is 3T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM TrioTim",
                "SoftwareVersions": "syngo MR B13",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "t1_mprage",
                "ProtocolName": "t1_mprage",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "2.98",  # TE: 2.98 ms
                "SliceThickness": "1.0",  # 1.00 mm
                "RepetitionTime": "2250",  # TR: 2250 ms
                "InversionTime": "900",  # TI: 900 ms
                "FlipAngle": "9",  # 9 deg
            },
            "KKI_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",
                "Manufacturer": "Philips",
                "ManufacturersModelName": None,
                "SoftwareVersions": None,
                "MRAcquisitionType": "3D",
                "SeriesDescription": "T1_TFE_MPRAGE",  # Based on sequence type
                "ProtocolName": "T1_TFE_MPRAGE",
                "ScanningSequence": "GR",  # FFE = Fast Field Echo (gradient echo)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "TFE",  # Turbo Field Echo
                "EchoTime": "3.7",  # Act. TR/TE: 8.0/3.7
                "SliceThickness": "1.0",  # slice thickness: 1mm
                "RepetitionTime": "3500",
                "InversionTime": "1000",  # To Exc: 1000ms
                "FlipAngle": "8",  # Flip angle: 8 deg
            },
            "NeuroIMAGE_": {
                "Modality": "MR",
                "MagneticFieldStrength": "1.5",  # Avanto is 1.5T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM Avanto",
                "SoftwareVersions": "syngo MR B17",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "t1_structural",
                "ProtocolName": "t1_structural",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "2.95",  # TE: 2.95 ms
                "SliceThickness": "1.0",  # 1.00 mm
                "RepetitionTime": "2730",  # TR: 2730 ms
                "InversionTime": "1000",  # TI: 1000 ms
                "FlipAngle": "7",  # 7 deg
            },
            "NYU_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",  # Allegra is 3T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM Allegra",
                "SoftwareVersions": "syngo MR 2004A",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "HighResT1",
                "ProtocolName": "HighResT1",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "3.25",  # TE: 3.25 ms
                "SliceThickness": "1.33",  # 1.33 mm
                "RepetitionTime": "2530",  # TR: 2530 ms
                "InversionTime": "1100",  # TI: 1100 ms
                "FlipAngle": "7",  # 7 deg
            },
            "OHSU_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",  # TrioTim is 3T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM TrioTim",
                "SoftwareVersions": "syngo MR B17",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "T1Anatomical-1",
                "ProtocolName": "T1Anatomical-1",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "3.58",  # TE: 3.58 ms
                "SliceThickness": "1.1",  # 1.10 mm
                "RepetitionTime": "2300",  # TR: 2300 ms
                "InversionTime": "900",  # TI: 900 ms
                "FlipAngle": "10",  # 10 deg
            },
            "Pittsburgh_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",  # TrioTim is 3T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM TrioTim",
                "SoftwareVersions": "syngo MR B15",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "axial_mprage",
                "ProtocolName": "axial_mprage",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "3.43",  # TE: 3.43 ms
                "SliceThickness": "1.0",  # 1.00 mm
                "RepetitionTime": "2100",  # TR: 2100 ms
                "InversionTime": "1050",  # TI: 1050 ms
                "FlipAngle": "8",  # 8 deg
            },
            "WashU_": {
                "Modality": "MR",
                "MagneticFieldStrength": "3.0",  # TrioTim is 3T scanner
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "MAGNETOM TrioTim",
                "SoftwareVersions": "syngo MR B13",
                "MRAcquisitionType": "3D",
                "SeriesDescription": "Mprage",
                "ProtocolName": "Mprage",
                "ScanningSequence": "GR",  # Gradient Echo (tfl = turbo flash)
                "SequenceVariant": "SP\\MP",  # Spoiled, Magnetization Prepared
                "ScanOptions": None,
                "SequenceName": "tfl",  # turbo flash
                "EchoTime": "3.08",  # TE: 3.08 ms
                "SliceThickness": "1.0",  # 1.00 mm
                "RepetitionTime": "2400",  # TR: 2400 ms
                "InversionTime": "1000",  # TI: 1000 ms
                "FlipAngle": "8",  # 8 deg
            },
        }
        my_modality_overrides = {"rest": {"suffix": "T1w", "folder": "anat"}}  # For some reasons rest are actual T1w!
        demographics_csv_path = os.path.join(target_dir, "tmp.csv")
        df1 = pd.read_csv(os.path.join(source_dir, "allSubs_testSet_phenotypic_dx.csv"))
        df2 = pd.read_csv(os.path.join(source_dir, "stce93_6_28_2025_2_31_30.csv"))
        # Extract ID from Subject column in df2
        df2["ID"] = df2["Subject"].str.extract(r"(\d+)").astype(int)
        # Merge on ID, adding Age, Gender, Handedness from df1
        result = df2.merge(df1[["ID", "Age", "Handedness"]], on="ID", how="left")
        # Drop the temporary ID column and save
        result.drop("ID", axis=1).to_csv(demographics_csv_path, index=False)
        # Save to tmp.csv
        result.to_csv("tmp.csv", index=False)
        processed_files = recursive_find_files(target_dir, extensions=dataset_config.in_extensions + [".pt"])
        expanded_df, subjects_df, mri_info_df = extract_demographics(
            processed_files=processed_files,
            demographics_csv_path=demographics_csv_path,
            columns_to_keep=dataset_config.df_columns,
            custom_patterns=dataset_config.patterns_bidsify,
            hardcoded_metadata=hardcoded_metadata,
            source_dir=source_dir,
            target_dir=target_dir,
            modality_patterns=my_modality_overrides,
        )
        os.remove(demographics_csv_path)

    if saving_config.bidsify:
        subjects_df.to_csv(os.path.join(target_dir, "participants.tsv"), sep="\t", index=False)

        rename_files_with_mapping(
            target_dir=target_dir,
            expanded_df=expanded_df,
            modality_patterns=my_modality_overrides,
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
