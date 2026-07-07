import os
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO300K_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.metadata_generation import (
    combine_datasets_without_splits,
    generate_dataset_json,
)
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import prepare_target_dir
from asparagus_preprocessing.utils.saving import enhanced_save_json


def main(
    path: str = "",
    subdir: str = "",
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=False,
):
    dataset_collection = [
        "PT001i",
        "PT002i",
        "PT003i",
        "PT004i",
        "PT005i",
        "PT006i",
        "PT007i",
        "PT008i",
        "PT009i",
        "PT010i",
        "PT011i",
        "PT012i",
        "PT013i",
        "PT014i",
        "PT015i",
        "PT016i",
        "PT017i",
        "PT018i",
        "PT019i",
        "PT020i",
        "PT021i",
        "PT022i",
        "PT023i",
        "PT024i",
        "PT025i",
        "PT026i",
        "PT027i",
        "PT028i",
        "PT029i",
        "PT030i",
        "PT031i",
        "PT032i",
        "PT033i",
        "PT034i",
        "PT035i",
        "PT036i",
    ]
    dataset_config = DatasetConfig(
        task_name="PT900i_FOMO300K_ISO",
        n_classes=1,
        n_modalities=1,
        in_extensions=[],
        split=None,
        patterns_exclusion=[],
        patterns_DWI=[],
        patterns_PET=[],
        patterns_perfusion=[],
        patterns_m0=[],
        patterns_bidsify=[],
        df_columns=[],
    )
    preprocessing_config = get_noresampling_preprocessing_config()
    saving_config = get_FOMO300K_saving_config(
        save_as_tensor=save_as_tensor,
        save_dset_metadata=save_dset_metadata,
        bidsify=bidsify,
    )

    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    prepare_target_dir(target_dir, saving_config.save_as_tensor)

    all_dataset_json, all_files = combine_datasets_without_splits(dataset_collection)

    generate_dataset_json(
        os.path.join(target_dir, "dataset.json"),
        dataset_name=dataset_config.task_name,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        dataset_config=dataset_config,
        metadata={
            "dataset_collection_jsons": all_dataset_json,
            "dataset_collection": dataset_collection,
            "final_files": len(all_files),
        },
    )
    enhanced_save_json(all_files, os.path.join(target_dir, "paths.json"))


if __name__ == "__main__":
    args = asparagus_parser.parse_args()
    main(
        processes=args.num_workers,
        bidsify=args.bidsify,
        save_dset_metadata=args.save_dset_metadata,
        save_as_tensor=args.save_as_tensor,
    )
