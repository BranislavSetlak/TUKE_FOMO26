from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_iso_preprocessing_config,
)
from asparagus_preprocessing.datasets_pretraining.PT026_MICA_MICs import process
from asparagus_preprocessing.paths import get_source_path


def main(
    path: str = get_source_path(),
    subdir: str = "MICA-MICs",
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
        task_name="PT026i_MICA_MICs_ISO",
    )
