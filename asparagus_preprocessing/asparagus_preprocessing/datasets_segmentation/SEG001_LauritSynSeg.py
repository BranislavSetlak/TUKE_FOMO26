import nibabel as nib
import numpy as np
import os
import torch
from asparagus.functional.task_conversion_and_preprocessing import (
    postprocess_standard_dataset,
    preprocess_case_with_label,
)
from asparagus.paths import get_data_path, get_source_labels_path
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_iso_preprocessing_config,
)
from asparagus_preprocessing.utils.saving import save_pickle
from dataclasses import asdict
from itertools import repeat
from multiprocessing.pool import Pool


def convert(processes=10):
    task_name = "Dataset_SEG001_LauritSynSeg"
    file_suffix = ".pt"
    n_modalities = 1
    n_classes = 5
    preprocessing_config = get_iso_preprocessing_config(modalities=n_modalities)

    target_dir = os.path.join(get_data_path(), task_name)
    raw_labels_target_dir = os.path.join(get_source_labels_path(), task_name)

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(raw_labels_target_dir, exist_ok=True)

    torch.manual_seed(421890)

    p = Pool(processes)
    p.starmap_async(
        generate_random_segcase,
        zip(
            range(250),
            repeat(preprocessing_config),
            repeat(target_dir),
            repeat(raw_labels_target_dir),
        ),
    )
    p.close()
    p.join()

    postprocess_standard_dataset(
        target_dir=target_dir,
        file_suffix=file_suffix,
        task_name=task_name,
        DWI_patterns=[],
        PET_patterns=[],
        exclusion_patterns=[],
        source_files_standard=[],
        source_files_DWI=[],
        source_files_PET=[],
        source_files_excluded=[],
        preprocessing_config=preprocessing_config,
        processes=processes,
        n_classes=n_classes,
        n_modalities=n_modalities,
    )


def generate_random_segcase(i, preprocessing_config, target_dir, raw_labels_target_dir):
    dims = (
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
        torch.randint(low=60, high=280, size=(1,)),
    )
    data = torch.FloatTensor(*dims).uniform_(-1.1e16, 1.1e16).numpy()

    seg = torch.randint(
        low=0,
        high=5,
        size=dims,
    ).numpy()

    images, seg, properties = preprocess_case_with_label(
        images=[data],
        label=seg,
        **asdict(preprocessing_config),
        strict=False,
    )
    torch.save(
        torch.cat([torch.tensor(np.array(images)), torch.tensor(seg).unsqueeze(0)]),
        os.path.join(target_dir, f"LauritSynSeg_{i}.pt"),
    )

    nib.save(
        nib.Nifti1Image(np.array(seg, dtype=np.float32), affine=np.eye(4)),
        os.path.join(raw_labels_target_dir, f"LauritSynSeg_{i}_label.nii.gz"),
    )

    save_pickle(properties, os.path.join(target_dir, f"LauritSynSeg_{i}.pkl"))


if __name__ == "__main__":
    convert()
