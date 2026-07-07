import logging
import nibabel as nib
import os
import pandas as pd
from asparagus_preprocessing.configs.preprocessing_presets import get_FOMO_saving_config, get_noresampling_preprocessing_config
from asparagus_preprocessing.paths import get_data_path, get_raw_labels_path, get_source_path
from asparagus_preprocessing.utils.dataclasses import DatasetConfig
from asparagus_preprocessing.utils.detect import simple_recursive_find_and_group_files
from asparagus_preprocessing.utils.metadata_generation import simple_postprocess_standard_dataset
from asparagus_preprocessing.utils.mp import process_dataset_with_table
from asparagus_preprocessing.utils.parser import asparagus_parser
from asparagus_preprocessing.utils.path import get_image_output_paths, prepare_target_dir
from asparagus_preprocessing.utils.process_case import preprocess_case_without_label
from asparagus_preprocessing.utils.saving import save_clsreg_data_and_metadata
from dataclasses import asdict


def process_sample(file_in, file_out, dataset_config, preprocessing_config, saving_config, table_in, table_out):
    # Skip already-processed cases
    if (os.path.exists(file_out + ".pt") and saving_config.save_as_tensor) or (
        os.path.exists(file_out + ".nii.gz") and not saving_config.save_as_tensor
    ):
        return
    logging.info(f"processing {file_in}")

    # Load image
    image = nib.load(file_in)
    # Multi-modality: derive companion paths from file_in, e.g.:
    # image2 = nib.load(file_in.replace("_T1w", "_FLAIR"))

    # Extract subject ID from file path
    # Adjust the parsing to match your dataset's filename convention, e.g.:
    #   subject_id = int(re.search(r"sub-(\d+)", file_in).group(1))
    #   subject_id = os.path.basename(file_in).split("_T1w")[0].replace("sub-", "")
    subject_id = ...  # extract from file_in

    # Look up the label in the labels table
    # table_in is the DataFrame loaded in main() below.
    # Adjust the column names to match your CSV/TSV/Excel file.
    row = table_in[table_in["subject_id_column"] == subject_id]
    if row.empty:
        logging.warning(f"No label found for subject {subject_id}, skipping")
        return
    text_label = row["label_column"].item()

    # Map raw label values to integer class indices (0-indexed)
    label_map = {"classA": 0, "classB": 1}  # adjust to your classes
    if text_label not in label_map:
        logging.warning(f"Unexpected label '{text_label}' for subject {subject_id}, skipping")
        return
    label = label_map[text_label]

    # Preprocess and save
    image, image_props = preprocess_case_without_label(
        images=[image],  # add image2, ... for multi-modality
        **asdict(preprocessing_config),
        strict=False,
    )
    save_clsreg_data_and_metadata(
        data=image,
        label=label,
        metadata=image_props,
        data_path=file_out,
        saving_config=saving_config,
        table_path=table_out,
    )


def main(
    path: str = get_source_path(),
    subdir: str = "MyDataset",  # folder name inside get_source_path()
    processes: int = 12,
    bidsify: bool = False,
    save_dset_metadata: bool = False,
    save_as_tensor: bool = True,
):
    dataset_config = DatasetConfig(
        task_name="CLS0XX_MyDataset",  # unique name; used as the output folder name
        in_extensions=[".nii.gz"],  # file extensions to search for
        n_classes=2,  # number of classes (here we have ClassA and ClassB)
        n_modalities=1,  # number of input image channels per case
        split="split_40_10_50",  # splitting fn used (See asparagus_preprocessing.utils.splitting)
        patterns_exclusion=["func", "dwi"],  # path substrings: matching files are skipped
    )

    saving_config = get_FOMO_saving_config(save_as_tensor=save_as_tensor)
    preprocessing_config = get_noresampling_preprocessing_config(n_modalities=dataset_config.n_modalities)

    source_dir = os.path.join(path, subdir)
    target_dir = os.path.join(get_data_path(), dataset_config.task_name)
    table_out = os.path.join(get_raw_labels_path(), dataset_config.task_name, "labels.csv")
    prepare_target_dir(target_dir, saving_config.save_as_tensor)

    # Load the demographics / labels table.
    # Adjust the filename and any preprocessing (column renaming, type casting, etc.)
    table_in = pd.read_csv(os.path.join(source_dir, "participants.tsv"), sep="\t")
    # table_in = pd.read_excel(os.path.join(source_dir, "participants.xlsx"), sheet_name="Sheet1")

    files_standard, files_excluded = simple_recursive_find_and_group_files(
        source_dir,
        extensions=dataset_config.in_extensions,
        patterns_exclusion=dataset_config.patterns_exclusion,
        processes=processes,
    )
    files_standard_out = get_image_output_paths(files_standard, source_dir, target_dir, dataset_config.in_extensions)

    process_dataset_with_table(
        process_fn=process_sample,
        files_in=files_standard,
        files_out=files_standard_out,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        saving_config=saving_config,
        table_in=table_in,
        table_out=table_out,
        processes=processes,
    )

    if saving_config.bidsify or saving_config.save_dset_metadata:
        raise NotImplementedError

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
