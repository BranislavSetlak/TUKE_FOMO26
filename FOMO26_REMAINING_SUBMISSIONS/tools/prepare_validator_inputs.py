#!/usr/bin/env python3
"""Generate valid deterministic NIfTIs for every remaining official validator task."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def volume(seed: int, shape: tuple[int, int, int], scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1)[:, None, None, None] / 2
    radius = np.sqrt(np.sum(((grid - center) / (np.asarray(shape)[:, None, None, None] / 2.3)) ** 2, axis=0))
    brain = radius < 1.0
    data = np.zeros(shape, dtype=np.float32)
    data[brain] = scale + 0.12 * scale * rng.standard_normal(int(brain.sum())).astype(np.float32)
    return data


def save(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    shape = (80, 88, 72)
    affine = np.asarray([[-1.1, 0, 0, 80], [0, 1.2, 0, -40], [0, 0, 1.5, -50], [0, 0, 0, 1]], dtype=np.float64)

    for case, optional, seed in (("case_swi", "swi", 20), ("case_t2s", "t2s", 30)):
        for offset, (name, intensity) in enumerate((("flair", 500), ("dwi", 700), (optional, 350))):
            save(root / "task2" / case / f"{name}.nii.gz", volume(seed + offset, shape, intensity), affine)
    save(root / "task3" / "case_001" / "t1.nii.gz", volume(40, shape, 600), affine)
    save(root / "task4" / "case_001" / "t2.nii.gz", volume(50, shape, 800), affine)
    save(root / "task5" / "case_001" / "t1.nii.gz", volume(60, shape, 650), affine)
    save(root / "task6" / "case_001" / "input.nii.gz", volume(70, shape, 500), affine)
    save(root / "task6" / "case_002" / "input.nii.gz", volume(71, (84, 76, 68), 900), np.diag([1.3, 1.1, 1.7, 1.0]))

    manifest = root / "manifest.yaml"
    manifest.write_text(
        "task2:\n"
        "  case_swi:\n    inputs:\n      flair: task2/case_swi/flair.nii.gz\n      dwi: task2/case_swi/dwi.nii.gz\n      swi: task2/case_swi/swi.nii.gz\n"
        "  case_t2s:\n    inputs:\n      flair: task2/case_t2s/flair.nii.gz\n      dwi: task2/case_t2s/dwi.nii.gz\n      t2s: task2/case_t2s/t2s.nii.gz\n"
        "task3:\n  case_001:\n    inputs:\n      t1: task3/case_001/t1.nii.gz\n"
        "task4:\n  case_001:\n    inputs:\n      t2: task4/case_001/t2.nii.gz\n"
        "task5:\n  case_001:\n    inputs:\n      t1: task5/case_001/t1.nii.gz\n"
        "task6_and_7:\n"
        "  case_001:\n    inputs:\n      input: task6/case_001/input.nii.gz\n"
        "  case_002:\n    inputs:\n      input: task6/case_002/input.nii.gz\n",
        encoding="utf-8",
    )
    print(f"ALL_VALIDATOR_INPUTS_READY manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
