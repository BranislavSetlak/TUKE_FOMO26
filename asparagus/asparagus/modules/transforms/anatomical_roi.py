"""Leakage-safe fixed anatomical ROI utilities for 3-D segmentation."""

from collections.abc import Sequence

import torch.nn.functional as F


def validate_normalized_center(normalized_center: Sequence[float]) -> tuple[float, float, float]:
    center = tuple(float(value) for value in normalized_center)
    if len(center) != 3:
        raise ValueError(f"Expected a 3-D normalized center, got {center}")
    if any(value < 0.0 or value > 1.0 for value in center):
        raise ValueError(f"Normalized center values must be in [0, 1], got {center}")
    return center


def validate_roi_size(roi_size: Sequence[int]) -> tuple[int, int, int]:
    size = tuple(int(value) for value in roi_size)
    if len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError(f"Expected three positive ROI dimensions, got {size}")
    return size


def fixed_roi_slices(
    spatial_shape: Sequence[int],
    roi_size: Sequence[int],
    normalized_center: Sequence[float],
) -> tuple[slice, slice, slice]:
    """Return a clamped fixed-size ROI around a normalized anatomical center."""

    shape = tuple(int(value) for value in spatial_shape)
    size = validate_roi_size(roi_size)
    center = validate_normalized_center(normalized_center)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3-D spatial shape, got {shape}")
    if any(image_dim < roi_dim for image_dim, roi_dim in zip(shape, size)):
        raise ValueError(
            f"Image shape {shape} is smaller than ROI {size}; pad before applying the ROI"
        )

    slices = []
    for image_dim, roi_dim, center_fraction in zip(shape, size, center):
        center_index = center_fraction * max(image_dim - 1, 0)
        start = int(round(center_index - (roi_dim - 1) / 2.0))
        start = min(max(start, 0), image_dim - roi_dim)
        slices.append(slice(start, start + roi_dim))
    return tuple(slices)


class Torch_FixedNormalizedCrop:
    """Crop image and label using one fold-specific normalized ROI center."""

    def __init__(
        self,
        roi_size: Sequence[int],
        normalized_center: Sequence[float],
        data_key: str = "image",
        label_key: str = "label",
    ):
        self.roi_size = validate_roi_size(roi_size)
        self.normalized_center = validate_normalized_center(normalized_center)
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, data_dict: dict) -> dict:
        image = data_dict[self.data_key]
        slices = fixed_roi_slices(
            spatial_shape=image.shape[1:],
            roi_size=self.roi_size,
            normalized_center=self.normalized_center,
        )
        data_dict[self.data_key] = image[(slice(None), *slices)]
        label = data_dict.get(self.label_key)
        if label is not None:
            data_dict[self.label_key] = label[(slice(None), *slices)]
        return data_dict


class Torch_ResizeImageAndLabel:
    """Resize a 3-D image and its categorical label to the same spatial size."""

    def __init__(
        self,
        target_size: Sequence[int],
        data_key: str = "image",
        label_key: str = "label",
    ):
        self.target_size = validate_roi_size(target_size)
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, data_dict: dict) -> dict:
        image = data_dict[self.data_key]
        if image.ndim != 4:
            raise ValueError(
                f"Expected image shape [C, X, Y, Z], got {tuple(image.shape)}"
            )
        if tuple(image.shape[1:]) != self.target_size:
            image_dtype = image.dtype
            data_dict[self.data_key] = F.interpolate(
                image.unsqueeze(0).float(),
                size=self.target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).to(dtype=image_dtype)

        label = data_dict.get(self.label_key)
        if label is not None:
            if label.ndim != 4:
                raise ValueError(
                    f"Expected label shape [C, X, Y, Z], got {tuple(label.shape)}"
                )
            if tuple(label.shape[1:]) != self.target_size:
                label_dtype = label.dtype
                data_dict[self.label_key] = F.interpolate(
                    label.unsqueeze(0).float(),
                    size=self.target_size,
                    mode="nearest",
                ).squeeze(0).to(dtype=label_dtype)
        return data_dict
