"""Task 1 preprocessing matched to CLS002_FOMO26_Infarct training."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to


TARGET_SIZE = (96, 96, 96)


def _load_canonical(path: str | Path) -> nib.Nifti1Image:
    image = nib.load(str(path))
    if len(image.shape) == 4 and image.shape[-1] == 1:
        data = np.asarray(image.dataobj, dtype=np.float32)[..., 0]
        image = nib.Nifti1Image(data, image.affine, image.header)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI at {path}, got shape {image.shape}")
    return nib.as_closest_canonical(image)


def _center_pad_crop(
    array: np.ndarray, target: tuple[int, int, int], pad_value: float
) -> np.ndarray:
    pad_width = []
    for current, wanted in zip(array.shape, target, strict=True):
        missing = max(0, wanted - current)
        pad_width.append((missing // 2, missing // 2 + missing % 2))
    if any(left or right for left, right in pad_width):
        array = np.pad(array, pad_width, mode="constant", constant_values=pad_value)

    slices = []
    for current, wanted in zip(array.shape, target, strict=True):
        start = max(0, (current - wanted) // 2)
        slices.append(slice(start, start + wanted))
    return np.asarray(array[tuple(slices)], dtype=np.float32)


def _volume_wise_znorm(channels: np.ndarray) -> np.ndarray:
    # This mirrors gardening_tools Torch_Normalize used at validation time:
    # the common foreground mask starts with nonzero FLAIR and is extended by
    # positive values in the remaining modalities. Background is left alone.
    mask = channels[0] != 0
    for channel in channels[1:]:
        mask |= channel > 0
    if not np.any(mask):
        raise ValueError("All three required modalities contain only background")

    output = channels.copy()
    for index in range(output.shape[0]):
        values = output[index][mask]
        mean = float(values.mean(dtype=np.float64))
        std = float(values.std(dtype=np.float64))
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Non-finite intensity statistics in channel {index}")
        std = max(std, 1e-8)
        output[index][mask] = (values - mean) / std
    return output


def load_infarct_tensor(flair: str, adc: str, dwi: str) -> tuple[torch.Tensor, dict]:
    """Return `[1, 3, 96, 96, 96]` in trained FLAIR/ADC/DWI order."""

    images = [_load_canonical(path) for path in (flair, adc, dwi)]
    reference = images[0]
    aligned = [reference]
    resampled = [False]
    for image in images[1:]:
        same_grid = image.shape == reference.shape and np.allclose(
            image.affine, reference.affine, rtol=1e-4, atol=1e-3
        )
        aligned.append(image if same_grid else resample_from_to(image, reference, order=3))
        resampled.append(not same_grid)

    arrays = []
    for image in aligned:
        array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(array)
    channels = np.stack(arrays, axis=0)
    channels = _volume_wise_znorm(channels)
    # Torch_Pad(pad_value="min") uses the minimum over the full CXYZ tensor,
    # not an independent minimum for each channel.
    pad_value = float(channels.min())
    channels = np.stack(
        [_center_pad_crop(channel, TARGET_SIZE, pad_value) for channel in channels]
    )
    channels = np.ascontiguousarray(channels, dtype=np.float32)

    if channels.shape != (3, *TARGET_SIZE) or not np.isfinite(channels).all():
        raise RuntimeError(f"Invalid preprocessed tensor: shape={channels.shape}")
    metadata = {
        "input_shapes": [list(image.shape) for image in images],
        "resampled_to_flair": resampled,
        "output_shape": [1, *channels.shape],
        "channel_order": ["flair", "adc", "dwi"],
        "channel_min": [float(value) for value in channels.min(axis=(1, 2, 3))],
        "channel_max": [float(value) for value in channels.max(axis=(1, 2, 3))],
    }
    return torch.from_numpy(channels).unsqueeze(0), metadata
