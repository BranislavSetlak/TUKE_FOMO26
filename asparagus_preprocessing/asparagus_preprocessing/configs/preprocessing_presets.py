from asparagus_preprocessing.utils.dataclasses import PreprocessingConfig, SavingConfig


def get_noresampling_preprocessing_config(n_modalities: int = 1) -> PreprocessingConfig:
    return PreprocessingConfig(
        normalization_operation=["no_norm"] * n_modalities,
        target_spacing=None,  # native spacing
        target_orientation="RAS",
        crop_to_nonzero=False,
        min_slices=15,
    )


def get_iso_preprocessing_config(n_modalities: int = 1) -> PreprocessingConfig:
    return PreprocessingConfig(
        normalization_operation=["no_norm"] * n_modalities,
        target_spacing=[1.0, 1.0, 1.0],
        target_orientation="RAS",
        crop_to_nonzero=False,
        min_slices=15,
    )


def get_FOMO300K_saving_config(save_as_tensor: bool, save_dset_metadata: bool, bidsify: bool) -> SavingConfig:
    if save_as_tensor == True:  # noqa: E712
        save_file_metadata = True
    else:
        save_file_metadata = False
    return SavingConfig(
        bidsify=bidsify,
        save_as_tensor=save_as_tensor,
        tensor_dtype="float32",
        save_dset_metadata=save_dset_metadata,
        save_file_metadata=save_file_metadata,
    )


def get_FOMO_saving_config(save_as_tensor: bool) -> SavingConfig:
    return SavingConfig(
        bidsify=False,
        save_as_tensor=save_as_tensor,
        tensor_dtype="float32",
        save_dset_metadata=False,
        save_file_metadata=True,
    )
