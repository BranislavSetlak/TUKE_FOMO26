The dataset.json for the task SEG003_ISLES22_ADCDWI:
```
{
    "dataset_config": {
        "df_columns": [],
        "in_extensions": [
            ".nii.gz"
        ],
        "n_classes": 2,
        "n_modalities": 2,
        "patterns_DWI": [],
        "patterns_PET": [],
        "patterns_bidsify": [],
        "patterns_exclusion": [
            "labels_derivatives",
            "_adc",
            "_dwi",
            ".json"
        ],
        "patterns_m0": [],
        "patterns_perfusion": [],
        "split": "split_40_10_50",
        "task_name": "SEG003_ISLES22_ADCDWI"
    },
    "metadata": {
        "files_delta_after_processing": -750,
        "files_source_directory_DWI": 0,
        "files_source_directory_PET": 0,
        "files_source_directory_Perfusion": 0,
        "files_source_directory_excluded": 750,
        "files_source_directory_standard": 250,
        "files_source_directory_total": 1000,
        "files_target_directory_DWI": 0,
        "files_target_directory_PET": 0,
        "files_target_directory_Perfusion": 0,
        "files_target_directory_standard": 250,
        "files_target_directory_total": 250,
        "n_classes": 2,
        "n_modalities": 2
    },
    "name": "SEG003_ISLES22_ADCDWI",
    "preprocessing_config": {
        "background_pixel_value": 0,
        "crop_to_nonzero": false,
        "image_properties": {},
        "intensities": null,
        "keep_aspect_ratio_when_using_target_size": false,
        "min_slices": 15,
        "normalization_operation": [
            "no_norm",
            "no_norm"
        ],
        "remove_nans": true,
        "target_orientation": "RAS",
        "target_size": null,
        "target_spacing": [
            1.0,
            1.0,
            1.0
        ]
    },
    "saving_config": {
        "bidsify": false,
        "save_as_tensor": true,
        "save_dset_metadata": false,
        "save_file_metadata": true,
        "tensor_dtype": "float32"
    }
}
```