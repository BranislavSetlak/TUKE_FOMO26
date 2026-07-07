import logging
import nibabel as nib
import os
from asparagus_preprocessing.configs.preprocessing_presets import get_FOMO_saving_config, get_noresampling_preprocessing_config
from asparagus_preprocessing.paths import get_data_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import simple_recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import simple_postprocess_standard_dataset
from asparagus_preprocessing.utils.mp import process_dataset_without_table
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir
from asparagus_preprocessing.utils.process_case import preprocess_case_with_label
from asparagus_preprocessing.utils.saving import save_data_and_metadata, save_raw_label
from dataclasses import asdict


def process_sample(file_in, file_out, dataset_config, preprocess_config, saving_config):
    # Skip already-processed cases (makes reruns safe)
    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return
    logging.info(f"processing {file_in}")

    # Load images
    # file_in is the "anchor" file found by recursive_find_and_group_files.
    # Derive the paths to your other modalities and the label from it.
    
    # Single-modality example:
    image1 = nib.load(file_in)
    
    # Multi-modality example (derive companion files from the anchor path):
    # image2 = nib.load(file_in.replace("_T1w", "_FLAIR"))
    # image3 = nib.load(file_in.replace("_T1w", "_T2w"))

    # Load label
    # Derive the label path from file_in using string replacement.
    label_path = file_in.replace("_T1w", "_seg")  # adjust to match your naming
    source_label = nib.load(label_path)

    # Optional label remapping
    # Use this if some label values need to be merged or discarded.
    # Example: remap class 2 (artefact) back to 0 (background)
    # label_data = source_label.get_fdata()
    # label_data[label_data == 2] = 0
    # source_label = nib.Nifti1Image(label_data, source_label.affine, source_label.header)

    # Preprocess
    # Pass all modalities as a list; label is processed consistently with them.
    image, label, image_props = preprocess_case_with_label(
        images=[image1],  # add image2, image3, ... for multi-modality
        label=source_label,
        **asdict(preprocess_config),
        strict=False,  # True = crash on error, False = log and skip
    )
    image_props["src_label_path"] = label_path

    # Save
    save_data_and_metadata(
        data=image + [label],  # images first, label last
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
    )
    save_raw_label(file_out, label_path)


def main(
    path: str = get_source_path(),
    subdir: str = "MyDataset",  # folder name inside get_source_path()
    processes: int = 12,
    bidsify: bool = False,
    save_dset_metadata: bool = False,
    save_as_tensor: bool = True,
):
    dataset_config = DatasetConfig(
        task_name="SEG0XX_MyDataset",  # unique name; used as the output folder name
        in_extensions=[".nii.gz"],  # file extensions to search for
        n_classes=2,  # number of foreground classes + 1 for background
        n_modalities=1,  # number of input image channels per case
        split="split_40_10_50",  # split function used (see utils/splitting.py)
        patterns_exclusion=["seg", "mask"],  # path substrings: matching files are skipped
    )

    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config(n_modalities=dataset_config.n_modalities)

    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    prepare_target_dir(target_dir, saving_config.save_as_tensor)

    files_standard, files_excluded = simple_recursive_find_and_group_files(
        source_dir,
        extensions=dataset_config.in_extensions,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )

    files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, dataset_config.in_extensions)

    process_dataset_without_table(
        process_fn=process_sample,
        files_in=files_standard,
        files_out=files_standard_out,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        processes=processes,
    )

    simple_postprocess_standard_dataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        target_dir=target_dir,
        source_files_standard=files_standard,
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
