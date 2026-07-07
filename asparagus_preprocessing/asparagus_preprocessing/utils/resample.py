import logging
import numpy as np
import pandas as pd
from asparagus_preprocessing.utils.normalize import normalizer
from numpy.typing import NDArray
from skimage.transform import resize


def resample_and_normalize_case(
    case: list,
    target_size,
    norm_op: str,
    intensities: list = None,
    label: np.ndarray = None,
    allow_missing_modalities: bool = False,
    resample_method: str = "yucca",
):
    assert resample_method in ["nnunet", "yucca"], "resample_method must be either 'nnunet' or 'yucca'"

    # Normalize and Transpose images to target view.
    # Transpose labels to target view.
    assert len(case) == len(norm_op), (
        "number of images and "
        "normalization  operations does not match. \n"
        f"len(images) == {len(case)} \n"
        f"len(norm_op) == {len(norm_op)} \n"
    )

    for i in range(len(case)):
        image = case[i]
        assert image is not None
        if image.size == 0:
            assert allow_missing_modalities is True, "missing modality and allow_missing_modalities is not enabled"
        else:
            # Normalize
            if intensities is not None:
                case[i] = normalizer(image, scheme=norm_op[i], intensities=intensities[i])
            else:
                case[i] = normalizer(image, scheme=norm_op[i])

            # Resample to target shape and spacing
            try:
                if resample_method == "nnunet":
                    case[i] = resize(case[i], output_shape=target_size, order=3, mode="edge")
                elif resample_method == "yucca":
                    case[i] = resize(case[i], output_shape=target_size, order=3)
            except OverflowError:
                logging.error("Unexpected values in either shape or image for resize")

    if label is not None:
        try:
            if resample_method == "nnunet":
                label = resize_segmentation(label, new_shape=target_size, order=3)
            elif resample_method == "yucca":
                label = resize(label, output_shape=target_size, order=0, anti_aliasing=False)

        except OverflowError:
            logging.error("Unexpected values in either shape or label for resize")
        return case, label

    return case


def resize_segmentation(
    segmentation: NDArray,
    new_shape: tuple[int, ...],
    order: int = 3,
) -> NDArray:
    """
    Resize a segmentation map while preserving label integrity.

    Converts segmentation to one-hot encoding, resizes each channel, and converts
    back to segmentation map. This prevents interpolation artifacts (e.g., [0, 0, 2]
    becoming [0, 1, 2]).

    Args:
        segmentation: Input segmentation map (multi-dimensional array).
        new_shape: Target shape for resizing.
        order: Interpolation order. 0 for nearest neighbor, 1-5 for higher orders.
            See skimage documentation for details. Defaults to 3 (cubic).

    Returns:
        Resized segmentation map with same dtype as input.

    Raises:
        AssertionError: If new_shape dimensionality doesn't match segmentation.
    """
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        ).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        unique_labels = np.sort(pd.unique(segmentation.ravel()))
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(
                mask.astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
