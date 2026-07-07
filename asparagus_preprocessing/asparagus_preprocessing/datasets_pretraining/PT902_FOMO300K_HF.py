#!/usr/bin/env python3
"""
Convert FOMO300K HF-downloaded MRI datasets to asparagus_preprocessing format.

Default:       Unzips into output_dir, converts .nii.gz -> .pt + .pkl,
               removes .nii.gz, writes dataset.json + paths.json per dataset
               and a combined FOMO300K collection at the output root.

--unzip_only:  Unzips in-place in input_dir (if not already unzipped),
               writes dataset.json + paths.json pointing to .nii.gz
               directly in input_dir. No output_dir needed.

Works whether data is still zipped or already unzipped.

Usage:
    python PT902_FOMO300K_HF.py --input_dir /path/to/hf_data --output_dir /path/to/output
    python PT902_FOMO300K_HF.py --input_dir /path/to/hf_data --output_dir /path/to/output --num_workers 24
    python PT902_FOMO300K_HF.py --input_dir /path/to/hf_data --unzip_only
"""

import argparse
import logging
import nibabel as nib
import numpy as np
import os
import shutil
import torch
import zipfile
from asparagus_preprocessing.configs.preprocessing_presets import (
    get_FOMO_saving_config,
    get_noresampling_preprocessing_config,
)
from asparagus_preprocessing.utils.detect import recursive_find_files
from asparagus_preprocessing.utils.metadata_generation import generate_dataset_json
from asparagus_preprocessing.utils.saving import enhanced_save_json, save_pickle
from multiprocessing import Pool
from time import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def unzip_all(root_dir, target_dir=None):
    """Unzip all .zip files. If target_dir is None, extract in-place next to the zip."""
    zips = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".zip"):
                zips.append(os.path.join(dirpath, f))
    if not zips:
        return
    logging.info(f"Found {len(zips)} zip files")
    for z in zips:
        if target_dir is None:
            dest = os.path.dirname(z)
        else:
            rel = os.path.relpath(z, root_dir)
            dest = os.path.join(target_dir, os.path.dirname(rel))
        # Check if already extracted by looking for the folder the zip would create
        zip_stem = os.path.splitext(os.path.basename(z))[0]
        expected = os.path.join(dest, zip_stem)
        if os.path.isdir(expected):
            continue
        try:
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest)
        except Exception as e:
            logging.error(f"Failed to unzip {z}: {e}")


def convert_single_nifti(args):
    nifti_path, output_path, saving_config = args
    ext = ".pt"

    if os.path.isfile(output_path + ext):
        if not saving_config.save_file_metadata or os.path.isfile(output_path + ".pkl"):
            return output_path + ext

    try:
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj, dtype=np.float32)[None]  # add channel dim -> (1, X, Y, Z)
        orientation = "".join(nib.aff2axcodes(img.affine))
        spacing = list(img.header.get_zooms()[:3])
        metadata = {
            "nifti_metadata": {
                "original_orientation": orientation,
                "original_spacing": spacing,
                "original_size": list(img.shape[:3]),
                "affine": img.affine.tolist(),
                "header": img.header.structarr.tobytes(),
                "final_direction": orientation,
            },
            "original_spacing": spacing,
            "original_size": list(img.shape[:3]),
            "original_orientation": orientation,
            "new_size": list(data.shape),
            "new_spacing": spacing,
            "new_direction": orientation,
            "crop_to_nonzero": False,
            "foreground_locations": [],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(torch.from_numpy(data), output_path + ext)

        if saving_config.save_file_metadata:
            save_pickle(metadata, output_path + ".pkl")

        return output_path + ext

    except Exception as e:
        logging.error(f"Error {nifti_path}: {e}")
        return None


def copy_metadata_files(source_dir, target_dir):
    if source_dir == target_dir:
        return
    for f in os.listdir(source_dir):
        if f.endswith(".tsv") or f.lower().startswith("license"):
            src = os.path.join(source_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(target_dir, f))


def process_dataset(dataset_name, input_dir, output_dir, saving_config, preprocessing_config, num_workers, unzip_only):
    """
    Process a single dataset (or sub-dataset).

    Assumes unzipping has ALREADY been done by the caller.

    Returns list of output file paths (.nii.gz if unzip_only, .pt otherwise).
    """
    source_dir = os.path.join(input_dir, dataset_name)
    if not os.path.isdir(source_dir):
        logging.warning(f"Directory not found: {source_dir}")
        return []

    if unzip_only:
        # json files point to .nii.gz in input_dir
        target_dir = source_dir

        niftis = recursive_find_files(target_dir, extensions=[".nii.gz"])
        if not niftis:
            logging.warning(f"No NIfTI files in {dataset_name}")
            return []

        output_paths = sorted(niftis)
        logging.info(f"{dataset_name}: {len(output_paths)} nifti files")
    else:
        # Convert to .pt in output_dir
        target_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)

        # Look for niftis in output_dir first (already unzipped there),
        # fall back to source_dir
        niftis = recursive_find_files(target_dir, extensions=[".nii.gz"])
        if not niftis:
            niftis = recursive_find_files(source_dir, extensions=[".nii.gz"])
        if not niftis:
            logging.warning(f"No NIfTI files in {dataset_name}")
            return []

        logging.info(f"{dataset_name}: {len(niftis)} nifti files")

        work = [(p, p[:-7], saving_config) for p in niftis]

        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = pool.map(convert_single_nifti, work)
        else:
            results = [convert_single_nifti(w) for w in work]

        output_paths = sorted([r for r in results if r is not None])
        logging.info(f"{dataset_name}: {len(output_paths)}/{len(niftis)} converted")

        for p in niftis:
            if os.path.isfile(p):
                os.remove(p)

    copy_metadata_files(source_dir, target_dir)
    enhanced_save_json(output_paths, os.path.join(target_dir, "paths.json"))
    generate_dataset_json(
        output_file=os.path.join(target_dir, "dataset.json"),
        dataset_name=dataset_name,
        metadata={"files_target_directory_total": len(output_paths)},
        saving_config=saving_config,
        preprocessing_config=preprocessing_config,
    )

    return output_paths


def collect_paths(root_dir, unzip_only):
    """Scan the directory for the correct file type and return sorted paths."""
    ext = [".nii.gz"] if unzip_only else [".pt"]
    paths = recursive_find_files(root_dir, extensions=ext)
    return sorted(paths)


def main(**kwargs):
    # Trigger error if kwargs are passed to main as in asp_process
    if kwargs:
        raise RuntimeError(
            "PT902_FOMO300K_HF must be run as a standalone script, not via asp_process. "
            "Use: python PT902_FOMO300K_HF.py --input_dir ... [--output_dir ... | --unzip_only]"
        )
    parser = argparse.ArgumentParser(description="Convert HF MRI datasets to asparagus format")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument(
        "--unzip_only", action="store_true", default=False, help="Only unzip in-place and create json files, no .pt conversion"
    )
    args = parser.parse_args()

    if bool(args.unzip_only) == bool(args.output_dir):
        parser.error("Provide exactly one of --output_dir (convert to .pt) or --unzip_only (unzip in place)")

    saving_config = get_FOMO_saving_config(save_as_tensor=not args.unzip_only)
    # Script skip preprocessing, only records config as metadata.
    preprocessing_config = get_noresampling_preprocessing_config()

    # Where json files and final data live
    json_root = args.input_dir if args.unzip_only else args.output_dir

    datasets = sorted(
        [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith("PT")]
    )
    logging.info(f"Found {len(datasets)} datasets, unzip_only={args.unzip_only}")

    t0 = time()
    all_files = []

    for dataset_name in datasets:
        source = os.path.join(args.input_dir, dataset_name)

        # ---- Unzip BEFORE discovering sub-datasets ----
        if args.unzip_only:
            unzip_all(source)
        else:
            unzip_all(source, os.path.join(args.output_dir, dataset_name))

        # Now discover sub-datasets (directories will exist after unzipping)
        sub_datasets = sorted([d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d)) and d.startswith("ds")])

        if sub_datasets:
            logging.info(f"{dataset_name}: {len(sub_datasets)} sub-datasets")
            ds_paths = []
            for i, sub_ds in enumerate(sub_datasets):
                paths = process_dataset(
                    os.path.join(dataset_name, sub_ds),
                    args.input_dir,
                    args.output_dir,
                    saving_config,
                    preprocessing_config,
                    args.num_workers,
                    args.unzip_only,
                )
                ds_paths.extend(paths)
                if (i + 1) % 100 == 0:
                    logging.info(f"  {i + 1}/{len(sub_datasets)} sub-datasets done")

            # Write parent-level paths.json by scanning actual files on disk
            ds_target_dir = os.path.join(json_root, dataset_name)
            os.makedirs(ds_target_dir, exist_ok=True)
            ds_paths = collect_paths(ds_target_dir, args.unzip_only)
            enhanced_save_json(ds_paths, os.path.join(ds_target_dir, "paths.json"))
            copy_metadata_files(source, ds_target_dir)
            generate_dataset_json(
                output_file=os.path.join(ds_target_dir, "dataset.json"),
                dataset_name=dataset_name,
                metadata={
                    "files_target_directory_total": len(ds_paths),
                    "n_sub_datasets": len(sub_datasets),
                },
                saving_config=saving_config,
                preprocessing_config=preprocessing_config,
            )
            all_files.extend(ds_paths)
        else:
            paths = process_dataset(
                dataset_name,
                args.input_dir,
                args.output_dir,
                saving_config,
                preprocessing_config,
                args.num_workers,
                args.unzip_only,
            )
            all_files.extend(paths)

    # Copy .tsv and license files into collection root
    copy_metadata_files(args.input_dir, json_root)
    for dataset_name in datasets:
        source = os.path.join(args.input_dir, dataset_name)
        copy_metadata_files(source, json_root)

    # Combined FOMO300K collection — scan actual files as ground truth
    all_files = collect_paths(json_root, args.unzip_only)
    enhanced_save_json(all_files, os.path.join(json_root, "paths.json"))
    generate_dataset_json(
        output_file=os.path.join(json_root, "dataset.json"),
        dataset_name="FOMO300K",
        metadata={
            "dataset_collection": datasets,
            "final_files": len(all_files),
        },
        saving_config=saving_config,
        preprocessing_config=preprocessing_config,
    )

    logging.info(f"Done in {time() - t0:.0f}s — {len(all_files)} total files")


if __name__ == "__main__":
    main()
