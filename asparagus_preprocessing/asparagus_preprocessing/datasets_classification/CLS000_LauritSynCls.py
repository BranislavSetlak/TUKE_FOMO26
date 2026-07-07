import os
import torch
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.paths import get_data_path, get_raw_labels_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.metadata_generation import (
    postprocess_standard_dataset,
)
from asparagus_preprocessing.utils.path import prepare_target_dir
from asparagus_preprocessing.utils.process_case import preprocess_case_without_label
from asparagus_preprocessing.utils.saving import save_clsreg_data_and_metadata
from dataclasses import asdict
from itertools import repeat
from multiprocessing.pool import Pool


def main(
    path=None,
    subdir=None,
    processes=12,
    bidsify=False,
    save_dset_metadata=False,
    save_as_tensor=True,
):
    dataset_config = DatasetConfig(
        task_name="CLS000_LauritSynCLS",
        in_extensions=[".pt"],
        n_classes=5,
        n_modalities=1,
        split="split_40_10_50",
        patterns_exclusion=["func", "dwi"],
    )
    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config(n_modalities=dataset_config.n_modalities)

    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    raw_labels_target_dir = os.path.join(get_raw_labels_path(), dataset_config.task_name)

    prepare_target_dir(target_dir, saving_config.save_as_tensor)
    os.makedirs(raw_labels_target_dir, exist_ok=True)

    torch.manual_seed(421890)

    p = Pool(processes)
    p.starmap(
        generate_random_clscase,
        zip(
            range(250),
            repeat(preprocessing_config),
            repeat(saving_config),
            repeat(target_dir),
            repeat(raw_labels_target_dir),
        ),
    )
    p.close()
    p.join()

    postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=[],
        source_files_DWI=[],
        source_files_PET=[],
        source_files_Perf=[],
        source_files_excluded=[],
        processes=processes,
    )


def generate_random_clscase(i, preprocessing_config, saving_config, target_dir, raw_labels_target_dir):
    dims = (
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
    )
    data = torch.FloatTensor(*dims).uniform_(-1.1e16, 1.1e16).numpy()

    label = torch.randint(
        low=0,
        high=5,
        size=(1,),
    )

    image, image_props = preprocess_case_without_label(images=[data], **asdict(preprocessing_config), strict=False)
    save_clsreg_data_and_metadata(
        data=image,
        label=label,
        metadata=image_props,
        data_path=os.path.join(target_dir, f"case_{i:03d}"),
        saving_config=saving_config,
        table_path=os.path.join(raw_labels_target_dir, f"case_{i:03d}.csv"),
    )


if __name__ == "__main__":
    main()
