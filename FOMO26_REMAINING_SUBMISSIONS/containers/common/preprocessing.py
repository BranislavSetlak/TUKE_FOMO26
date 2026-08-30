"""Preprocessing matched to the FOMO26 TUKE fine-tuning pipelines."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to


CLSREG_SIZE = (96, 96, 96)


def load_nifti(path: str | Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    original = nib.load(str(path))
    if len(original.shape) == 4 and original.shape[-1] == 1:
        data = np.asarray(original.dataobj, dtype=np.float32)[..., 0]
        original = nib.Nifti1Image(data, original.affine, original.header)
    if len(original.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI at {path}, got {original.shape}")
    return original, nib.as_closest_canonical(original)


def _arrays(paths: list[str]) -> tuple[np.ndarray, nib.Nifti1Image, nib.Nifti1Image, list[bool]]:
    loaded = [load_nifti(path) for path in paths]
    original, reference = loaded[0]
    aligned = [reference]
    resampled = [False]
    for _source_original, source in loaded[1:]:
        same = source.shape == reference.shape and np.allclose(
            source.affine, reference.affine, rtol=1e-4, atol=1e-3
        )
        aligned.append(source if same else resample_from_to(source, reference, order=3))
        resampled.append(not same)
    values = []
    for image in aligned:
        array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        values.append(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(values), original, reference, resampled


def volume_znorm(channels: np.ndarray) -> np.ndarray:
    mask = channels[0] != 0
    for channel in channels[1:]:
        mask |= channel > 0
    if not np.any(mask):
        raise ValueError("Input contains no non-background voxels")
    output = channels.copy()
    for index in range(output.shape[0]):
        values = output[index][mask]
        mean = float(values.mean(dtype=np.float64))
        std = max(float(values.std(dtype=np.float64)), 1e-8)
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Non-finite intensity statistics in channel {index}")
        output[index][mask] = (values - mean) / std
    return output


def _pad(channels: np.ndarray, minimum: tuple[int, int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pads = []
    for current, wanted in zip(channels.shape[1:], minimum, strict=True):
        missing = max(0, wanted - current)
        pads.append((missing // 2, missing - missing // 2))
    pad_spec = [(0, 0), *pads]
    value = float(channels.min())
    return np.pad(channels, pad_spec, mode="constant", constant_values=value), pads


def _center_crop(channels: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    slices = []
    for current, wanted in zip(channels.shape[1:], size, strict=True):
        start = max(0, (current - wanted) // 2)
        slices.append(slice(start, start + wanted))
    return channels[(slice(None), *slices)]


def load_clsreg_tensor(path: str) -> tuple[torch.Tensor, dict]:
    channels, _original, _canonical, resampled = _arrays([path])
    channels = volume_znorm(channels)
    channels, _pads = _pad(channels, CLSREG_SIZE)
    channels = _center_crop(channels, CLSREG_SIZE)
    channels = np.ascontiguousarray(channels, dtype=np.float32)
    if channels.shape != (1, *CLSREG_SIZE) or not np.isfinite(channels).all():
        raise RuntimeError(f"Invalid cls/reg tensor shape={channels.shape}")
    return torch.from_numpy(channels).unsqueeze(0), {
        "output_shape": [1, *channels.shape],
        "resampled": resampled,
    }


def load_segmentation_tensor(paths: list[str], roi_size: tuple[int, int, int]) -> tuple[torch.Tensor, dict]:
    channels, original, canonical, resampled = _arrays(paths)
    channels = volume_znorm(channels)
    original_canonical_shape = tuple(int(v) for v in channels.shape[1:])
    channels, pads = _pad(channels, roi_size)
    channels = np.ascontiguousarray(channels, dtype=np.float32)
    if not np.isfinite(channels).all():
        raise RuntimeError("Non-finite segmentation tensor")
    context = {
        "original": original,
        "canonical": canonical,
        "canonical_shape": original_canonical_shape,
        "pads": pads,
        "resampled": resampled,
        "tensor_shape": [1, *channels.shape],
    }
    return torch.from_numpy(channels).unsqueeze(0), context


def remove_padding(array: np.ndarray, context: dict) -> np.ndarray:
    slices = []
    for axis, (left, _right) in enumerate(context["pads"]):
        size = context["canonical_shape"][axis]
        slices.append(slice(left, left + size))
    return np.asarray(array[tuple(slices)])


def save_original_grid_mask(mask_canonical: np.ndarray, context: dict, output: str | Path) -> None:
    original = context["original"]
    canonical = context["canonical"]
    if tuple(mask_canonical.shape) != tuple(canonical.shape):
        raise ValueError(f"Mask shape {mask_canonical.shape} != canonical shape {canonical.shape}")
    canonical_mask = nib.Nifti1Image(mask_canonical.astype(np.uint8), canonical.affine)
    restored = resample_from_to(canonical_mask, (original.shape, original.affine), order=0)
    restored_data = np.rint(np.asarray(restored.dataobj)).astype(np.uint8)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(restored_data, original.affine, header)
    result.set_qform(original.affine, code=int(original.header["qform_code"]) or 1)
    result.set_sform(original.affine, code=int(original.header["sform_code"]) or 1)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(destination))
