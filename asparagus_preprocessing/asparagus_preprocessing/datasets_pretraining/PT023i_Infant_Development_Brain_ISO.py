from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_iso_preprocessing_config,
)
from asparagus_preprocessing.datasets_pretraining.PT023_Infant_Development_Brain import (
    process,
)
from asparagus_preprocessing.paths import get_source_path


def main(
    path: str = get_source_path(),
    subdir: str = "Infant_Development_Brain",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_iso_preprocessing_config()

    process(
        saving_config=saving_config,
        preprocessing_config=preprocessing_config,
        path=path,
        subdir=subdir,
        processes=processes,
        task_name="PT023i_Infant_Development_Brain_ISO",
    )
