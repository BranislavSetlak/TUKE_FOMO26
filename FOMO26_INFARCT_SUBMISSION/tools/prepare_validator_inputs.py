#!/usr/bin/env python3
"""Create valid local Task 1 NIfTIs because the supplied ZIP has LFS pointers."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def _volume(seed: int, modality: str, shape=(80, 88, 72)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1)[:, None, None, None] / 2
    scale = np.asarray(shape, dtype=np.float32)[:, None, None, None] / 2.4
    radius = np.sqrt(np.sum(((grid - center) / scale) ** 2, axis=0))
    brain = radius < 1.0
    base = {
        "flair": 500.0,
        "adc": 900.0,
        "dwi": 650.0,
        "swi": 350.0,
        "t2s": 400.0,
    }[modality]
    data = np.zeros(shape, dtype=np.float32)
    data[brain] = base + 80.0 * rng.standard_normal(int(brain.sum())).astype(np.float32)
    lesion = (
        ((grid[0] - shape[0] * 0.63) / 7.0) ** 2
        + ((grid[1] - shape[1] * 0.46) / 6.0) ** 2
        + ((grid[2] - shape[2] * 0.54) / 5.0) ** 2
    ) < 1.0
    if modality == "dwi":
        data[lesion] += 300.0
    elif modality == "adc":
        data[lesion] -= 250.0
    elif modality == "flair":
        data[lesion] += 120.0
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    affine = np.diag([1.1, 1.1, 1.5, 1.0]).astype(np.float32)
    subjects = (("synthetic_swi", "swi", 101), ("synthetic_t2s", "t2s", 202))
    for subject, optional, seed in subjects:
        directory = root / subject
        directory.mkdir(parents=True, exist_ok=True)
        for offset, modality in enumerate(("flair", "dwi", "adc", optional)):
            image = nib.Nifti1Image(_volume(seed + offset, modality), affine)
            image.set_qform(affine, code=1)
            image.set_sform(affine, code=1)
            nib.save(image, directory / f"{modality}.nii.gz")

    manifest = root / "manifest.yaml"
    manifest.write_text(
        "task1:\n"
        "  synthetic_swi:\n"
        "    inputs:\n"
        "      flair: synthetic_swi/flair.nii.gz\n"
        "      dwi: synthetic_swi/dwi.nii.gz\n"
        "      adc: synthetic_swi/adc.nii.gz\n"
        "      swi: synthetic_swi/swi.nii.gz\n"
        "  synthetic_t2s:\n"
        "    inputs:\n"
        "      flair: synthetic_t2s/flair.nii.gz\n"
        "      dwi: synthetic_t2s/dwi.nii.gz\n"
        "      adc: synthetic_t2s/adc.nii.gz\n"
        "      t2s: synthetic_t2s/t2s.nii.gz\n",
        encoding="utf-8",
    )
    print(f"VALIDATOR_INPUTS_READY manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
